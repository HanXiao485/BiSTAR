import math
from pathlib import Path

import numpy as np
from skimage.metrics import structural_similarity


def psnr(reference, reconstructed):
    mse = np.mean((reference.astype(np.float64) - reconstructed.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def ssim(reference, reconstructed):
    return float(structural_similarity(reference, reconstructed, channel_axis=2, data_range=255))


def ms_ssim(reference, reconstructed):
    scores, a, b = [], reference, reconstructed
    for _ in range(3):
        scores.append(ssim(a, b))
        if min(a.shape[:2]) < 16: break
        a = a[::2, ::2]; b = b[::2, ::2]
    return float(np.prod(np.maximum(scores, 1e-12) ** (1.0 / len(scores))))


def lpips_distance(reference, reconstructed, model=None):
    if model is None:
        # Explicit deterministic fallback for lightweight CPU runs, labelled separately by callers.
        return float(np.mean(np.abs(reference.astype(np.float32) - reconstructed.astype(np.float32))) / 255.0)
    import torch
    a = torch.from_numpy(reference.transpose(2, 0, 1)).float()[None] / 127.5 - 1
    b = torch.from_numpy(reconstructed.transpose(2, 0, 1)).float()[None] / 127.5 - 1
    with torch.no_grad(): return float(model(a, b).item())


def frame_metrics(reference, reconstructed, lpips_model=None):
    return {"psnr": psnr(reference, reconstructed), "ssim": ssim(reference, reconstructed),
            "ms_ssim": ms_ssim(reference, reconstructed), "lpips": lpips_distance(reference, reconstructed, lpips_model),
            "lpips_backend": "lpips" if lpips_model is not None else "normalized_l1_fallback"}


def create_lpips_model(device="cpu", network="squeeze"):
    """Load the standard AlexNet LPIPS metric once per experiment."""
    import torch
    torch.hub.set_dir(str(Path(__file__).resolve().parents[2] / "checkpoints/torch/hub"))
    import lpips
    model = lpips.LPIPS(net=network, verbose=False).to(device)
    model.eval()
    return model


def roi_metrics(reference, reconstructed, roi_mask, lpips_model=None):
    mask = np.asarray(roi_mask, dtype=bool)
    if mask.shape != reference.shape[:2] or not mask.any():
        raise ValueError("ROI mask must match the frame and contain at least one pixel")
    ys, xs = np.where(mask); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    ref_crop = reference[y0:y1, x0:x1].copy(); rec_crop = reconstructed[y0:y1, x0:x1].copy()
    crop_mask = mask[y0:y1, x0:x1]
    ref_crop[~crop_mask] = 0; rec_crop[~crop_mask] = 0
    values = frame_metrics(ref_crop, rec_crop, lpips_model)
    return {"roi_psnr": values["psnr"], "roi_ssim": values["ssim"], "roi_lpips": values["lpips"]}
