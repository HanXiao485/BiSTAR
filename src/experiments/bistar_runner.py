import time

import numpy as np

from src.bistar.reconstruction.videomae_adapter import VideoMAEAdapter
from src.bistar.selector import FullSelector
from src.bistar.transport.packetizer import Packetizer
from src.bistar.transport.serializer import deserialize_packet
from src.common.config import load_yaml
from src.metrics.image_quality import create_lpips_model, frame_metrics


def encode_sparse_clip(frames, budget, sequence_number=0, fps=15, patch_size=16, payload_encoding=None):
    """Run selector and real serialization without exposing source frames to the decoder."""
    selector = FullSelector(patch_size=patch_size)
    selected = selector.select(frames[-1], frames[-2] if len(frames) > 1 else None,
                               (frames[-1].shape[1] / 2, frames[-1].shape[0] / 2, 1.0), budget)
    packetizer = Packetizer(patch_size=patch_size, **(payload_encoding or {"mode": "png", "quality": 90}))
    sparse_frames, accounting = [], []
    for frame_id, frame in enumerate(frames):
        _, wire, sizes = packetizer.packetize(frame, selected["selected_indices"], sequence_number, frame_id,
                                               int(frame_id / fps * 1e9))
        received = deserialize_packet(wire)
        sparse_frames.append(packetizer.sparse_frame(received)); accounting.append(sizes)
    return {"sparse_frames": sparse_frames, "visible_mask": selected["visible_mask"], "accounting": accounting,
            "selected_indices": selected["selected_indices"]}


def select_budget_for_rate(frames, target_kbps, fps, patch_size=16, payload_encoding=None):
    best = None
    for budget in range(1, (frames[0].shape[0] // patch_size) * (frames[0].shape[1] // patch_size) + 1):
        encoded = encode_sparse_clip(frames, budget, fps=fps, patch_size=patch_size, payload_encoding=payload_encoding)
        total = sum(item["total_bytes"] for item in encoded["accounting"])
        rate = total * 8.0 / (len(frames) / fps) / 1000.0
        error = abs(rate - target_kbps)
        if best is None or error < best[0]: best = (error, budget, rate, encoded)
        if rate >= target_kbps: break
    return {"budget": best[1], "actual_kbps": best[2], "encoded": best[3],
            "status": "matched" if best[0] / target_kbps <= 0.05 else "bitrate_unreachable"}


def evaluate_videomae_clip(frames, encoded, model_config_path="configs/models/videomae.yaml", adapter=None, lpips_model=None):
    adapter = adapter or VideoMAEAdapter(load_yaml(model_config_path)); started = time.perf_counter()
    reconstructed = adapter.reconstruct_clip(encoded["sparse_frames"], encoded["visible_mask"])
    latency = time.perf_counter() - started
    lpips_model = lpips_model or create_lpips_model("cpu")
    rows = [frame_metrics(reference, output, lpips_model) for reference, output in zip(frames, reconstructed)]
    payload = sum(item["payload_bytes"] for item in encoded["accounting"])
    metadata = sum(item["metadata_bytes"] for item in encoded["accounting"])
    return {"reconstructed": reconstructed, "frame_metrics": rows, "payload_bytes": payload,
            "metadata_bytes": metadata, "total_bytes": payload + metadata, "inference_time_sec": latency,
            "inference_ms_per_frame": latency * 1000 / len(frames), "gpu_peak_memory_bytes": _gpu_peak_memory()}


def _gpu_peak_memory():
    try:
        import torch
        return int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    except ImportError:
        return 0
