import numpy as np


def bitrate_kbps(num_bytes, duration_sec):
    if duration_sec <= 0: raise ValueError("duration must be positive")
    return num_bytes * 8.0 / duration_sec / 1000.0


def communication_metrics(frame_bytes, fps):
    values = np.asarray(frame_bytes, dtype=float); duration = len(values) / fps
    bins = [values[i:i + int(fps)].sum() * 8 / 1000 for i in range(0, len(values), int(fps))]
    return {"mean_bitrate_kbps": bitrate_kbps(int(values.sum()), duration), "peak_1s_kbps": float(max(bins, default=0)),
            "p95_1s_kbps": float(np.percentile(bins, 95)) if bins else 0.0}


def bitrate_breakdown(payload_bytes, metadata_bytes, fps):
    if len(payload_bytes) != len(metadata_bytes):
        raise ValueError("Payload and metadata series must have equal length")
    payload = communication_metrics(payload_bytes, fps); metadata = communication_metrics(metadata_bytes, fps)
    total = communication_metrics((np.asarray(payload_bytes) + np.asarray(metadata_bytes)).tolist(), fps)
    return {"payload_bitrate_kbps": payload["mean_bitrate_kbps"],
            "metadata_bitrate_kbps": metadata["mean_bitrate_kbps"],
            "total_bitrate_kbps": total["mean_bitrate_kbps"], "peak_1s_kbps": total["peak_1s_kbps"],
            "p95_1s_kbps": total["p95_1s_kbps"]}
