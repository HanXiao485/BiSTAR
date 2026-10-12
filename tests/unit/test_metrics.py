import numpy as np

from src.metrics.image_quality import roi_metrics
from src.metrics.latency import latency_summary
from src.metrics.temporal_quality import residual_difference


def test_roi_and_temporal_metrics():
    a = np.zeros((32, 32, 3), np.uint8); b = a.copy(); b[8:24, 8:24] = 20
    roi = np.zeros((32, 32), bool); roi[8:24, 8:24] = True
    values = roi_metrics(a, b, roi)
    assert values["roi_psnr"] > 0 and values["roi_ssim"] < 1
    assert residual_difference(a, b, a, a) > 0


def test_latency_percentiles():
    values = latency_summary([1, 2, 3, 100])
    assert values["p50_latency_ms"] == 2.5 and values["p99_latency_ms"] > values["p95_latency_ms"]
