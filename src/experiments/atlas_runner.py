import copy
import time
from pathlib import Path

import numpy as np

from src.bistar.reconstruction.videomae_adapter import VideoMAEAdapter
from src.bistar.selector import (
    EdgeSelector,
    FullSelector,
    GazeEdgeSelector,
    GazeSelector,
    GazeTemporalSelector,
    RandomSelector,
    StructuralSelector,
    TemporalSSIMSelector,
)
from src.bistar.transport.atlas import decode_atlas_gop, encode_atlas_gop
from src.bistar.transport.patch_budget import PatchBudgetController
from src.common.config import load_yaml
from src.metrics.image_quality import frame_metrics


SELECTORS = {
    "random": RandomSelector,
    "gaze_only": GazeSelector,
    "temporal_only": TemporalSSIMSelector,
    "edge_only": EdgeSelector,
    "structural_only": StructuralSelector,
    "gaze_temporal": GazeTemporalSelector,
    "gaze_edge": GazeEdgeSelector,
    "bistar_full": FullSelector,
}


def load_atlas_config(path):
    config = load_yaml(path)
    if "codec" not in config:
        raise ValueError("Atlas configuration requires codec")
    codec = config["codec"]
    required = ("patch_size", "gop_size", "selector", "encoder", "control")
    missing = [key for key in required if key not in codec]
    if missing:
        raise ValueError("Atlas configuration missing: {}".format(", ".join(missing)))
    if codec["control"].get("mask_encoding") != "bitmask":
        raise ValueError("Atlas control.mask_encoding must be bitmask")
    if codec["encoder"].get("b_frames", 0) != 0:
        raise ValueError("Atlas real-time encoder requires b_frames=0")
    selector = codec["selector"]
    if "patch_budget_mode" in selector and not isinstance(selector["patch_budget_mode"], bool):
        raise ValueError("Atlas selector.patch_budget_mode must be true or false")
    return config


def evaluate_atlas_sequence(frames, fps, atlas_config, output_dir, target_kbps=None,
                            selector_type=None, visible_patch_count=None, model_adapter=None,
                            lpips_model=None, compute_metrics=True, sequence_number=0,
                            reconstruct=True, budget_target_kbps=None):
    codec = atlas_config["codec"] if "codec" in atlas_config else atlas_config
    patch_size = int(codec["patch_size"])
    gop_size = int(codec["gop_size"])
    selector_name = selector_type or codec["selector"]["type"]
    selector = SELECTORS[selector_name](patch_size=patch_size)
    encoder = copy.deepcopy(codec["encoder"])
    if target_kbps is not None:
        encoder["mode"] = "bitrate"
        encoder["bitrate_kbps"] = float(target_kbps)
    control_target_kbps = (
        budget_target_kbps if budget_target_kbps is not None else
        target_kbps if target_kbps is not None else codec["encoder"].get("bitrate_kbps", 100)
    )
    image_height, image_width = frames[0].shape[:2]
    patch_count = (image_height // patch_size) * (image_width // patch_size)
    controller = PatchBudgetController(
        codec["selector"], codec["encoder"].get("bitrate_kbps", 100), patch_count,
        target_kbps=control_target_kbps,
        force_fixed=visible_patch_count is not None,
    )
    if visible_patch_count is not None:
        controller.current = int(visible_patch_count)
    if reconstruct:
        model_adapter = model_adapter or VideoMAEAdapter(load_yaml("configs/models/videomae.yaml"))
        if model_adapter.model is None:
            model_adapter.load_model()
    reconstructed_frames = []
    metric_rows = []
    frame_latencies_ms = []
    total_bytes = payload_bytes = metadata_bytes = 0
    previous_source = None
    budgets = []
    selection_hits = np.zeros((image_height // patch_size, image_width // patch_size), dtype=np.float64)
    gop_rows = []
    for gop_id, start in enumerate(range(0, len(frames), gop_size)):
        valid = frames[start:start + gop_size]
        valid_count = len(valid)
        current = valid[0]
        gaze = (current.shape[1] / 2.0, current.shape[0] / 2.0, 1.0)
        budget = controller.current
        selection = selector.select(current, previous_source, gaze, budget)
        budgets.append(len(selection["selected_indices"]))
        selection_hits += selection["visible_mask"] * valid_count
        encode_started = time.perf_counter()
        encoded = encode_atlas_gop(
            valid, selection["selected_indices"], fps, patch_size,
            Path(output_dir) / "atlas_bitstreams", encoder,
            sequence_id=sequence_number, gop_id=gop_id, start_frame_id=start,
            timestamp_ns=int(start / fps * 1e9),
        )
        received = decode_atlas_gop(encoded["wire"])
        if received["total_bytes"] != encoded["total_bytes"]:
            raise RuntimeError("Receiver and sender Atlas byte accounting differ")
        encode_decode_seconds = time.perf_counter() - encode_started
        reconstructed = []
        inference_seconds = 0.0
        if reconstruct:
            inference_started = time.perf_counter()
            model_input = received["sparse_frames"] + [received["sparse_frames"][-1]] * (
                gop_size - valid_count)
            reconstructed = model_adapter.reconstruct_clip(model_input, received["visible_mask"])
            inference_seconds = time.perf_counter() - inference_started
        gop_total = received["total_bytes"]
        gop_payload = received["payload_bytes"]
        gop_metadata = received["metadata_bytes"]
        total_bytes += gop_total
        payload_bytes += gop_payload
        metadata_bytes += gop_metadata
        reconstructed_frames.extend(reconstructed[:valid_count])
        for offset in range(valid_count):
            wait_ms = (valid_count - 1 - offset) / fps * 1000.0
            processing_ms = (encode_decode_seconds + inference_seconds) / valid_count * 1000.0
            frame_latencies_ms.append(wait_ms + processing_ms)
            if compute_metrics and reconstruct:
                metric_rows.append(frame_metrics(valid[offset], reconstructed[offset], lpips_model))
        gop_rows.append({
            "gop_id": gop_id,
            "start_frame_id": start,
            "valid_frames": valid_count,
            "visible_patch_count": len(selection["selected_indices"]),
            "payload_bytes": gop_payload,
            "metadata_bytes": gop_metadata,
            "total_bytes": gop_total,
            "algorithmic_wait_ms": (valid_count - 1) / fps * 1000.0,
            "processing_ms": (encode_decode_seconds + inference_seconds) * 1000.0,
            "atlas_width": encoded["layout"]["width"],
            "atlas_height": encoded["layout"]["height"],
            "patch_budget_strategy": controller.strategy,
            "tail_transport_policy": "valid_frames_only",
        })
        controller.update(gop_total, valid_count, fps, control_target_kbps)
        previous_source = valid[-1]
    duration = len(frames) / fps
    return {
        "reconstructed": reconstructed_frames,
        "frame_metrics": metric_rows,
        "frame_latencies_ms": frame_latencies_ms,
        "gop_results": gop_rows,
        "payload_bytes": payload_bytes,
        "metadata_bytes": metadata_bytes,
        "total_bytes": total_bytes,
        "actual_bitrate_kbps": total_bytes * 8.0 / duration / 1000.0,
        "mean_latency_ms": float(np.mean(frame_latencies_ms)),
        "p95_latency_ms": float(np.percentile(frame_latencies_ms, 95)),
        "visible_patch_count": float(np.mean(budgets)),
        "visible_patch_counts": budgets,
        "patch_budget_strategy": controller.strategy,
        "selection_frequency": selection_hits / len(frames),
        "selector": selector_name,
        "transport": "h264_patch_atlas",
        "evaluation_mode": "chunked_causal_selection",
    }


def match_atlas_bitrate(frames, fps, atlas_config, output_dir, target_kbps,
                        tolerance_percent, max_iterations, model_adapter,
                        lpips_model=None, compute_metrics=True, sequence_number=0):
    requested = float(target_kbps)
    iteration = 0
    for iteration in range(1, int(max_iterations) + 1):
        probe = evaluate_atlas_sequence(
            frames, fps, atlas_config, Path(output_dir) / "rate_search" / str(iteration),
            target_kbps=requested, compute_metrics=False, sequence_number=sequence_number,
            reconstruct=False, budget_target_kbps=target_kbps,
        )
        error = abs(probe["actual_bitrate_kbps"] - target_kbps) / target_kbps * 100.0
        if error <= tolerance_percent:
            break
        requested *= max(0.5, min(2.0, target_kbps / probe["actual_bitrate_kbps"]))
    result = evaluate_atlas_sequence(
        frames, fps, atlas_config, Path(output_dir) / "final",
        target_kbps=requested, model_adapter=model_adapter, lpips_model=lpips_model,
        compute_metrics=compute_metrics, sequence_number=sequence_number,
        budget_target_kbps=target_kbps,
    )
    result["requested_kbps"] = requested
    result["match_iterations"] = iteration
    result["status"] = (
        "matched" if abs(result["actual_bitrate_kbps"] - target_kbps) / target_kbps * 100.0
        <= tolerance_percent else "bitrate_unreachable"
    )
    return result
