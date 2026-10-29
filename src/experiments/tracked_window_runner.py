import copy
import time
from pathlib import Path

import numpy as np

from src.bistar.reconstruction.videomae_adapter import VideoMAEAdapter
from src.bistar.transport.tracked_window import (
    decode_tracked_window_gop,
    encode_tracked_window_gop,
)
from src.common.config import load_yaml
from src.datasets.task_roi import frame_annotations, read_task_annotations
from src.metrics.image_quality import frame_metrics


def load_tracked_window_config(path):
    config = load_yaml(path)
    if "codec" not in config:
        raise ValueError("Tracked-window configuration requires codec")
    codec = config["codec"]
    required = ("window", "encoder", "reconstruction")
    missing = [key for key in required if key not in codec]
    if missing:
        raise ValueError("Tracked-window configuration missing: {}".format(", ".join(missing)))
    if codec["reconstruction"].get("type") != "videomae":
        raise ValueError("Tracked-window reconstruction must be videomae")
    if int(codec["encoder"].get("b_frames", 0)) != 0:
        raise ValueError("Tracked-window real-time encoder requires b_frames=0")
    return config


def focus_points_from_annotations(frame_count, width, height, gaze_points=None,
                                  roi_masks=None, missing_policy="frame_center"):
    points = []
    gaze_points = gaze_points or [None] * frame_count
    for index in range(frame_count):
        gaze = gaze_points[index] if index < len(gaze_points) else None
        if gaze is not None:
            points.append((float(gaze[0]), float(gaze[1])))
            continue
        if roi_masks is not None and index < len(roi_masks) and np.asarray(roi_masks[index]).any():
            ys, xs = np.nonzero(roi_masks[index])
            points.append((float(xs.mean()), float(ys.mean())))
            continue
        if missing_policy == "frame_center":
            points.append((width / 2.0, height / 2.0))
        elif missing_policy == "hold_last" and points:
            points.append(points[-1])
        else:
            raise ValueError("Visual focus is missing for frame {}".format(index))
    return points


def resolve_experiment_focus(experiment_config, sequence_id, frames, project_root):
    """Resolve gaze -> ROI center -> configured fallback for one sequence."""
    roi = experiment_config.get("roi")
    gaze = masks = None
    provenance = ["configured_fallback"]
    if roi and roi.get("annotations"):
        annotations = read_task_annotations(
            Path(project_root) / roi["annotations"] if not Path(roi["annotations"]).is_absolute()
            else Path(roi["annotations"]),
            roi.get("coordinate_space", "normalized"),
        )
        masks, gaze, provenance = frame_annotations(
            annotations, sequence_id, range(len(frames)), frames[0].shape[0], frames[0].shape[1],
            roi.get("coordinate_space", "normalized"),
        )
    policy = experiment_config.get("tracked_window", {}).get("missing_focus_policy", "frame_center")
    points = focus_points_from_annotations(
        len(frames), frames[0].shape[1], frames[0].shape[0], gaze, masks, policy)
    return points, provenance, masks


def evaluate_tracked_window_sequence(frames, focus_points, fps, window_config, output_dir,
                                     target_kbps=None, model_adapter=None, lpips_model=None,
                                     compute_metrics=True, sequence_number=0, reconstruct=True):
    codec = copy.deepcopy(window_config["codec"] if "codec" in window_config else window_config)
    clip_size = int(codec["reconstruction"].get("window_frames", 16))
    if target_kbps is not None:
        codec["encoder"]["mode"] = "bitrate"
        codec["encoder"]["bitrate_kbps"] = float(target_kbps)
    if reconstruct:
        model_adapter = model_adapter or VideoMAEAdapter(load_yaml("configs/models/videomae.yaml"))
        if model_adapter.model is None:
            model_adapter.load_model()
    reconstructed_frames, metric_rows, gop_rows = [], [], []
    total_bytes = payload_bytes = metadata_bytes = 0
    latencies = []
    selection_hits = np.zeros(frames[0].shape[:2], dtype=np.float64)
    for gop_id, start in enumerate(range(0, len(frames), clip_size)):
        valid = frames[start:start + clip_size]
        valid_focus = focus_points[start:start + len(valid)]
        began = time.perf_counter()
        sent = encode_tracked_window_gop(
            valid, valid_focus, fps, codec, Path(output_dir) / "bitstreams",
            sequence_id=sequence_number, gop_id=gop_id, start_frame_id=start,
            timestamp_ns=int(start / fps * 1e9),
        )
        received = decode_tracked_window_gop(sent["wire"])
        transport_seconds = time.perf_counter() - began
        if received["total_bytes"] != sent["total_bytes"]:
            raise RuntimeError("Tracked-window sender and receiver accounting differ")
        inference_seconds = 0.0
        reconstructed = []
        if reconstruct:
            sparse = list(received["sparse_frames"])
            masks = np.asarray(received["visible_pixel_masks"])
            if len(sparse) < clip_size:
                padding = clip_size - len(sparse)
                sparse += [sparse[-1].copy() for _ in range(padding)]
                masks = np.concatenate([masks, np.repeat(masks[-1:], padding, axis=0)])
            began = time.perf_counter()
            reconstructed = model_adapter.reconstruct_dynamic_clip(sparse, masks)
            inference_seconds = time.perf_counter() - began
            reconstructed_frames.extend(list(reconstructed[:len(valid)]))
        total_bytes += received["total_bytes"]
        payload_bytes += received["payload_bytes"]
        metadata_bytes += received["metadata_bytes"]
        selection_hits += received["visible_pixel_masks"].sum(axis=0)
        processing_ms = (transport_seconds + inference_seconds) * 1000.0 / len(valid)
        for offset in range(len(valid)):
            wait_ms = (len(valid) - 1 - offset) / fps * 1000.0
            latencies.append(wait_ms + processing_ms)
            if compute_metrics and reconstruct:
                metric_rows.append(frame_metrics(valid[offset], reconstructed[offset], lpips_model))
        gop_rows.append({
            "gop_id": gop_id, "start_frame_id": start, "valid_frames": len(valid),
            "payload_bytes": received["payload_bytes"],
            "metadata_bytes": received["metadata_bytes"], "total_bytes": received["total_bytes"],
            "window_width": int(codec["window"]["width"]),
            "window_height": int(codec["window"]["height"]),
            "mean_window_motion_px": float(np.mean([
                np.hypot(b[0] - a[0], b[1] - a[1])
                for a, b in zip(received["positions"], received["positions"][1:])
            ])) if len(received["positions"]) > 1 else 0.0,
            "algorithmic_wait_ms": (len(valid) - 1) / fps * 1000.0,
            "processing_ms": (transport_seconds + inference_seconds) * 1000.0,
            "tail_transport_policy": "valid_frames_only",
        })
    duration = len(frames) / fps
    return {
        "reconstructed": reconstructed_frames, "frame_metrics": metric_rows,
        "gop_results": gop_rows, "payload_bytes": payload_bytes,
        "metadata_bytes": metadata_bytes, "total_bytes": total_bytes,
        "actual_bitrate_kbps": total_bytes * 8.0 / duration / 1000.0,
        "mean_latency_ms": float(np.mean(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "selection_frequency_pixels": selection_hits / len(frames),
        "transport": "tracked_window_video",
        "evaluation_mode": "receiver_closed_dynamic_mask",
    }


def match_tracked_window_bitrate(frames, focus_points, fps, window_config, output_dir,
                                 target_kbps, tolerance_percent, max_iterations,
                                 model_adapter, lpips_model=None, compute_metrics=True,
                                 sequence_number=0):
    requested = float(target_kbps)
    for iteration in range(1, int(max_iterations) + 1):
        probe = evaluate_tracked_window_sequence(
            frames, focus_points, fps, window_config,
            Path(output_dir) / "rate_search" / str(iteration), target_kbps=requested,
            compute_metrics=False, reconstruct=False, sequence_number=sequence_number,
        )
        error = abs(probe["actual_bitrate_kbps"] - target_kbps) / target_kbps * 100.0
        if error <= tolerance_percent:
            break
        requested *= max(0.5, min(2.0, target_kbps / probe["actual_bitrate_kbps"]))
    result = evaluate_tracked_window_sequence(
        frames, focus_points, fps, window_config, Path(output_dir) / "final",
        target_kbps=requested, model_adapter=model_adapter, lpips_model=lpips_model,
        compute_metrics=compute_metrics, sequence_number=sequence_number,
    )
    result["requested_kbps"] = requested
    result["match_iterations"] = iteration
    result["status"] = "matched" if (
        abs(result["actual_bitrate_kbps"] - target_kbps) / target_kbps * 100.0
        <= tolerance_percent) else "bitrate_unreachable"
    return result
