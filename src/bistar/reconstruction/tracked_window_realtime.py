"""Causal VideoMAE reconstruction policies for tracked-window transport."""

import time

import numpy as np


def _history_clip(sparse_frames, visible_masks, frame_id, history_frames):
    """Return only decoder-available frames ending at ``frame_id``."""
    first = max(0, frame_id - history_frames + 1)
    indices = list(range(first, frame_id + 1))
    indices = [indices[0]] * (history_frames - len(indices)) + indices
    return ([sparse_frames[index] for index in indices],
            np.asarray([visible_masks[index] for index in indices], dtype=bool))


def reconstruct_causal_rolling(adapter, sparse_frames, visible_masks, history_frames=16):
    """Reconstruct frame t from [t-history+1, t], never from a future frame."""
    reconstructed, latencies = [], []
    for frame_id in range(len(sparse_frames)):
        clip, masks = _history_clip(
            sparse_frames, visible_masks, frame_id, int(history_frames))
        began = time.perf_counter()
        output = adapter.reconstruct_dynamic_clip(clip, masks)
        latencies.append(time.perf_counter() - began)
        reconstructed.append(output[-1])
    return reconstructed, {
        "submitted_jobs": len(reconstructed),
        "completed_jobs": len(reconstructed),
        "dropped_jobs": 0,
        "frames_without_background": 0,
        "mean_background_age_frames": 0.0,
        "max_background_age_frames": 0.0,
        "model_latency_ms": float(np.mean(latencies) * 1000.0),
        "p95_model_latency_ms": float(np.percentile(latencies, 95) * 1000.0),
        "critical_path_model_latency_ms": float(np.mean(latencies) * 1000.0),
    }


def _schedule_latest_jobs(frame_ids, durations, fps):
    """Simulate one worker with a one-item pending queue and keep-latest policy."""
    completed = {}
    current = None
    pending = None
    duration_by_id = dict(zip(frame_ids, durations))
    for frame_id in frame_ids:
        arrival = frame_id / float(fps)
        while current is not None and current[0] <= arrival:
            finish, job_id = current
            completed[job_id] = finish
            if pending is None:
                current = None
                break
            current = (finish + duration_by_id[pending], pending)
            pending = None
        if current is None:
            current = (arrival + duration_by_id[frame_id], frame_id)
        else:
            pending = frame_id
    while current is not None:
        finish, job_id = current
        completed[job_id] = finish
        if pending is None:
            current = None
        else:
            current = (finish + duration_by_id[pending], pending)
            pending = None
    return completed


def reconstruct_async_fast_path(adapter, sparse_frames, visible_masks, fps,
                                history_frames=16, inference_stride=2):
    """Display the current ROI immediately and refresh the background asynchronously."""
    stride = int(inference_stride)
    if stride < 1:
        raise ValueError("Async inference_stride must be positive")
    triggers = list(range(stride - 1, len(sparse_frames), stride))
    predictions, durations = {}, []
    for frame_id in triggers:
        clip, masks = _history_clip(
            sparse_frames, visible_masks, frame_id, int(history_frames))
        began = time.perf_counter()
        output = adapter.reconstruct_dynamic_clip(clip, masks)
        durations.append(time.perf_counter() - began)
        predictions[frame_id] = output[-1]
    completions = _schedule_latest_jobs(triggers, durations, fps)
    by_finish = sorted((finish, frame_id) for frame_id, finish in completions.items())
    background = np.zeros_like(sparse_frames[0])
    latest = None
    cursor = 0
    reconstructed, ages = [], []
    for frame_id in range(len(sparse_frames)):
        deadline = frame_id / float(fps)
        while cursor < len(by_finish) and by_finish[cursor][0] <= deadline:
            latest = by_finish[cursor][1]
            cursor += 1
        canvas = predictions[latest].copy() if latest is not None else background.copy()
        mask = visible_masks[frame_id]
        canvas[mask] = sparse_frames[frame_id][mask]
        reconstructed.append(canvas)
        ages.append(None if latest is None else frame_id - latest)
    finite_ages = [age for age in ages if age is not None]
    return reconstructed, {
        "submitted_jobs": len(triggers),
        "completed_jobs": len(completions),
        "dropped_jobs": len(triggers) - len(completions),
        "frames_without_background": sum(age is None for age in ages),
        "mean_background_age_frames": float(np.mean(finite_ages)) if finite_ages else None,
        "max_background_age_frames": float(np.max(finite_ages)) if finite_ages else None,
        "model_latency_ms": float(np.mean(durations) * 1000.0) if durations else 0.0,
        "p95_model_latency_ms": float(np.percentile(durations, 95) * 1000.0) if durations else 0.0,
        "critical_path_model_latency_ms": 0.0,
    }
