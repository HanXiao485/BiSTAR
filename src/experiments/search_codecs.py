import copy
import time
from pathlib import Path

import numpy as np

from src.bistar.reconstruction.interpolation import InterpolationReconstructor
from src.bistar.reconstruction.previous_frame import PreviousFrameReconstructor
from src.bistar.transport.atlas import decode_atlas_gop, encode_atlas_gop
from src.bistar.transport.packetizer import Packetizer
from src.bistar.transport.patch_budget import PatchBudgetController
from src.bistar.transport.serializer import deserialize_packet
from src.codecs import H264Adapter, H265Adapter
from src.datasets.frame_reader import read_rgb_frames
from src.experiments.atlas_runner import SELECTORS
from src.experiments.runner import write_csv


def _selector(codec, seed):
    name = codec["selector"]["type"]
    if name not in SELECTORS:
        raise ValueError("Unknown selector: {}".format(name))
    if name == "random":
        return SELECTORS[name](patch_size=int(codec["patch_size"]), seed=int(seed))
    return SELECTORS[name](patch_size=int(codec["patch_size"]))


def _reconstruct(sparse, mask, codec, get_model, previous):
    kind = codec["reconstruction"]["type"]
    if kind == "videomae":
        adapter = get_model()
        expected = int(adapter.config["temporal"]["num_frames"])
        if len(sparse) > expected:
            raise ValueError("GOP cannot exceed VideoMAE temporal.num_frames")
        # Model-only padding is never transmitted or billed as fractional bytes.
        clip = sparse + [sparse[-1]] * (expected - len(sparse))
        return list(adapter.reconstruct_clip(clip, mask)[:len(sparse)])
    classes = {"previous_frame": PreviousFrameReconstructor, "interpolation": InterpolationReconstructor}
    if kind not in classes:
        raise ValueError("Unknown reconstruction: {}".format(kind))
    reconstructor = classes[kind](int(codec["patch_size"]))
    output = []
    for frame in sparse:
        previous = reconstructor.reconstruct(frame, mask, previous)
        output.append(previous)
    return output


def _encode_sparse_gop(clip, selection, codec, fps, sequence_number, start):
    packetizer = Packetizer(patch_size=int(codec["patch_size"]), **codec["payload_encoding"])
    sparse, wires, accounting = [], [], []
    for offset, frame in enumerate(clip):
        _, wire, sizes = packetizer.packetize(
            frame, selection["selected_indices"], sequence_number, start + offset,
            int((start + offset) / fps * 1e9),
        )
        sparse.append(packetizer.sparse_frame(deserialize_packet(wire)))
        wires.append(wire)
        accounting.append(sizes)
    return {"sparse_frames": sparse, "wires": wires,
            **{key: sum(row[key] for row in accounting)
               for key in ("payload_bytes", "metadata_bytes", "total_bytes")}}


def _match_sparse_budget(clip, current, previous, gaze, codec, target, fps, seed, maximum):
    best = None
    lower, upper = 1, maximum
    visited = set()
    # Rate-only probing uses no ROI quality and no model inference.
    while lower <= upper:
        budget = (lower + upper) // 2
        visited.add(budget)
        selection = _selector(codec, seed).select(current, previous, gaze, budget)
        encoded = _encode_sparse_gop(clip, selection, codec, fps, 0, 0)
        actual = encoded["total_bytes"] * 8 / (len(clip) / fps) / 1000
        item = (abs(actual - target), budget, selection, encoded)
        if best is None or item[:2] < best[:2]:
            best = item
        if actual < target:
            lower = budget + 1
        else:
            upper = budget - 1
    for budget in {1, maximum, max(1, upper), min(maximum, lower)} - visited:
        selection = _selector(codec, seed).select(current, previous, gaze, budget)
        encoded = _encode_sparse_gop(clip, selection, codec, fps, 0, 0)
        actual = encoded["total_bytes"] * 8 / (len(clip) / fps) / 1000
        item = (abs(actual - target), budget, selection, encoded)
        if item[:2] < best[:2]:
            best = item
    return best[2], best[3]


def sparse_sequence(frames, fps, candidate, output_dir, target, get_model, gaze_points,
                    reconstruct=True, budget_target=None, use_gaze=False):
    method, codec = candidate["method"], candidate["config"]["codec"]
    patch_size = int(codec["patch_size"])
    gop_size = int(codec.get("gop_size", 16))
    rows, columns = frames[0].shape[0] // patch_size, frames[0].shape[1] // patch_size
    maximum = rows * columns
    budget_target = target if budget_target is None else budget_target
    controller = PatchBudgetController(codec["selector"], codec.get("encoder", {}).get("bitrate_kbps", 100),
                                       maximum, target_kbps=budget_target)
    encoder = copy.deepcopy(codec.get("encoder", {}))
    encoder.update({"mode": "bitrate", "bitrate_kbps": target})
    previous_source = previous_decoded = None
    reconstructed, masks, gop_rows = [], [], []
    total = payload = metadata = 0
    encode_seconds = inference_seconds = 0.0
    latencies = []
    for gop_id, start in enumerate(range(0, len(frames), gop_size)):
        clip = frames[start:start + gop_size]
        gaze = gaze_points[start] if use_gaze else None
        if gaze is None:
            gaze = (frames[0].shape[1] / 2, frames[0].shape[0] / 2, 1.0)
        seed = int(codec["selector"].get("seed", 0)) + gop_id
        began = time.perf_counter()
        if method == "bistar" and not codec["selector"].get("patch_budget_mode", True):
            selection, encoded = _match_sparse_budget(
                clip, clip[0], previous_source, gaze, codec, target, fps, seed, maximum
            )
            encoded = _encode_sparse_gop(clip, selection, codec, fps, 0, start)
        else:
            selection = _selector(codec, seed).select(clip[0], previous_source, gaze, controller.current)
            if method == "bistar_atlas":
                encoded = encode_atlas_gop(clip, selection["selected_indices"], fps, patch_size,
                                           output_dir / "bitstreams", encoder,
                                           gop_id=gop_id, start_frame_id=start,
                                           timestamp_ns=int(start / fps * 1e9))
                received = decode_atlas_gop(encoded["wire"])
                if received["total_bytes"] != encoded["total_bytes"]:
                    raise RuntimeError("Receiver and sender Atlas byte accounting differ")
                encoded["sparse_frames"] = received["sparse_frames"]
                encoded["visible_mask"] = received["visible_mask"]
            else:
                encoded = _encode_sparse_gop(clip, selection, codec, fps, 0, start)
        transport_seconds = time.perf_counter() - began
        encode_seconds += transport_seconds
        inference = 0.0
        if reconstruct:
            began = time.perf_counter()
            receiver_mask = encoded.get("visible_mask", selection["visible_mask"])
            output = _reconstruct(encoded["sparse_frames"], receiver_mask, codec,
                                  get_model, previous_decoded)
            inference = time.perf_counter() - began
            inference_seconds += inference
            previous_decoded = output[-1]
            reconstructed.extend(output)
            masks.extend([receiver_mask] * len(clip))
            wire_dir = output_dir / "wire"
            wire_dir.mkdir(parents=True, exist_ok=True)
            if method == "bistar_atlas":
                (wire_dir / "gop_{:06d}.bin".format(gop_id)).write_bytes(encoded["wire"])
            else:
                for offset, wire in enumerate(encoded["wires"]):
                    (wire_dir / "frame_{:06d}.bin".format(start + offset)).write_bytes(wire)
        total += encoded["total_bytes"]
        payload += encoded["payload_bytes"]
        metadata += encoded["metadata_bytes"]
        processing_ms = (transport_seconds + inference) * 1000
        latencies.extend([(len(clip) - 1 - offset) / fps * 1000 + processing_ms for offset in range(len(clip))])
        gop_rows.append({"gop_id": gop_id, "start_frame": start, "valid_frames": len(clip),
                         "visible_patch_count": len(selection["selected_indices"]),
                         "payload_bytes": encoded["payload_bytes"], "metadata_bytes": encoded["metadata_bytes"],
                         "total_bytes": encoded["total_bytes"], "processing_ms": processing_ms,
                         "algorithmic_wait_ms": (len(clip) - 1) / fps * 1000})
        controller.update(encoded["total_bytes"], len(clip), fps, budget_target)
        previous_source = clip[-1]
    return {"reconstructed": reconstructed, "visible_masks": masks,
            "total_bytes": total, "payload_bytes": payload, "metadata_bytes": metadata,
            "actual_bitrate_kbps": total * 8 / (len(frames) / fps) / 1000,
            "encode_decode_time_sec": encode_seconds, "inference_time_sec": inference_seconds,
            "mean_latency_ms": float(np.mean(latencies)), "p95_latency_ms": float(np.percentile(latencies, 95)),
            "latency_scope": "offline_gop_wait_plus_full_processing_excludes_network_display",
            "visible_patch_counts": [row["visible_patch_count"] for row in gop_rows],
            "gop_results": gop_rows}


def evaluate_candidate(frames, source, fps, preprocessing, candidate, output_dir, target,
                       options, get_model, gaze_points, use_gaze=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    method = candidate["method"]
    trials = []
    if method in ("h264", "h265"):
        adapter = (H264Adapter if method == "h264" else H265Adapter)()
        requested = float(target)
        best = None
        for iteration in range(1, int(options["max_match_iterations"]) + 1):
            cfg = copy.deepcopy(candidate["config"])
            cfg["preprocessing"] = preprocessing
            cfg["codec"]["encoder"].update({"mode": "bitrate", "bitrate_kbps": requested,
                                             "maxrate_kbps": requested, "bufsize_kbps": requested * 2})
            encoded = adapter.encode_sequence(source, output_dir / "match_{}".format(iteration), cfg)
            error = abs(encoded["actual_bitrate_kbps"] - target) / target * 100
            trials.append({"iteration": iteration, "requested_kbps": requested,
                           "actual_kbps": encoded["actual_bitrate_kbps"], "error_percent": error})
            if best is None or error < best[0]:
                best = (error, encoded, cfg, requested)
            if error <= options["tolerance_percent"]:
                break
            requested *= max(0.5, min(2.0, target / encoded["actual_bitrate_kbps"]))
        error, encoded, cfg, requested = best
        decoded = adapter.decode_sequence(Path(encoded["bitstream_path"]), output_dir / "decoded", cfg)
        reconstructed = read_rgb_frames(Path(decoded["decoded_path"]))
        mask = np.ones((frames[0].shape[0] // 16, frames[0].shape[1] // 16), dtype=bool)
        result = {**encoded, "reconstructed": reconstructed, "visible_masks": [mask] * len(frames),
                  "mean_latency_ms": None, "p95_latency_ms": None,
                  "latency_scope": "whole_sequence_offline_codec_runtime_only",
                  "decode_time_sec": decoded["decode_time_sec"]}
    else:
        requested = float(target)
        if method == "bistar_atlas":
            best = None
            for iteration in range(1, int(options["max_match_iterations"]) + 1):
                probe = sparse_sequence(frames, fps, candidate, output_dir / "rate_search" / str(iteration),
                                        requested, get_model, gaze_points, reconstruct=False,
                                        budget_target=target, use_gaze=use_gaze)
                error = abs(probe["actual_bitrate_kbps"] - target) / target * 100
                trials.append({"iteration": iteration, "requested_kbps": requested,
                               "actual_kbps": probe["actual_bitrate_kbps"], "error_percent": error})
                if best is None or error < best[0]:
                    best = (error, requested)
                if error <= options["tolerance_percent"]:
                    break
                requested *= max(0.5, min(2.0, target / probe["actual_bitrate_kbps"]))
            requested = best[1]
        result = sparse_sequence(frames, fps, candidate, output_dir / "final", requested,
                                 get_model, gaze_points, reconstruct=False, budget_target=target,
                                 use_gaze=use_gaze)
        error = abs(result["actual_bitrate_kbps"] - target) / target * 100
        # Unmatched configurations cannot be ranked and do not need neural inference.
        if error <= options["tolerance_percent"]:
            result = sparse_sequence(frames, fps, candidate, output_dir / "final", requested,
                                     get_model, gaze_points, budget_target=target, use_gaze=use_gaze)
        if "gop_results" in result:
            write_csv(result["gop_results"], output_dir / "gop_results.csv")
    result["status"] = ("matched" if abs(result["actual_bitrate_kbps"] - target) / target * 100
                        <= options["tolerance_percent"] else "bitrate_unreachable")
    result["requested_kbps"] = requested
    result["match_iterations"] = len(trials) or 1
    write_csv(trials, output_dir / "rate_search.csv")
    return result
