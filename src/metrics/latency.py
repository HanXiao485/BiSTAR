import numpy as np


def latency_summary(values_ms):
    a = np.asarray(values_ms, dtype=float)
    if a.size == 0: raise ValueError("At least one latency value is required")
    return {"mean_latency_ms": float(a.mean()), "p50_latency_ms": float(np.percentile(a, 50)),
            "p95_latency_ms": float(np.percentile(a, 95)), "p99_latency_ms": float(np.percentile(a, 99))}


def decoded_frame_latency_ms(frame):
    if frame.capture_timestamp_ns is None or frame.display_ready_timestamp_ns is None:
        raise ValueError("DecodedFrame is missing end-to-end timestamps")
    return (frame.display_ready_timestamp_ns - frame.capture_timestamp_ns) / 1e6
