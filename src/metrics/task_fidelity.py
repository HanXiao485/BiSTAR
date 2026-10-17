import numpy as np
from skimage.metrics import structural_similarity

from .spatial_quality import PatchQualityAccumulator


def task_frame_metrics(reference, reconstructed, roi_mask, patch_size=16,
                       psnr_threshold=30.0, ssim_threshold=0.9, visible_mask=None):
    """Measure independent task ROIs; retention means explicit quality thresholds."""
    roi_mask = np.asarray(roi_mask, dtype=bool)
    if roi_mask.shape != reference.shape[:2] or not roi_mask.any():
        raise ValueError("Task ROI must be nonempty and match frame dimensions")
    error = reference.astype(np.float64) - reconstructed.astype(np.float64)
    mse = float(np.mean(error[roi_mask] ** 2))
    _, similarity = structural_similarity(
        reference, reconstructed, channel_axis=2, data_range=255, full=True
    )
    similarity = similarity.mean(axis=2) if similarity.ndim == 3 else similarity
    height, width = roi_mask.shape
    if height % patch_size or width % patch_size:
        raise ValueError("Task metric grid requires dimensions divisible by patch_size")
    weights = roi_mask.reshape(height // patch_size, patch_size, width // patch_size, patch_size).sum((1, 3))
    patch_mse = (error ** 2).mean(axis=2) * roi_mask
    patch_mse = patch_mse.reshape(height // patch_size, patch_size, width // patch_size, patch_size).sum((1, 3))
    patch_mse = patch_mse / np.maximum(weights, 1)
    patch_ssim = (similarity * roi_mask).reshape(
        height // patch_size, patch_size, width // patch_size, patch_size
    ).sum((1, 3)) / np.maximum(weights, 1)
    retained = (patch_mse <= 255.0 ** 2 * 10.0 ** (-psnr_threshold / 10.0)) & (patch_ssim >= ssim_threshold)
    coverage = None
    if visible_mask is not None:
        coverage = float(np.sum(weights * visible_mask) / weights.sum())
    return {
        "roi_mse": mse,
        "roi_psnr": float(10 * np.log10(255.0 ** 2 / max(mse, 1e-12))),
        "roi_ssim": float(similarity[roi_mask].mean()),
        "roi_normalized_l1": float(np.abs(error[roi_mask]).mean() / 255.0),
        "critical_patch_retention": float(np.sum(weights * retained) / weights.sum()),
        "roi_visible_coverage": coverage,
        "roi_pixels": int(roi_mask.sum()),
    }


def evaluate_task_sequence(reference, reconstructed, masks, metadata, visible_masks=None,
                           patch_size=16, psnr_threshold=30.0, ssim_threshold=0.9,
                           valid_frames=None):
    if not (len(reference) == len(reconstructed) == len(masks)):
        raise ValueError("Source, reconstruction and ROI frame counts must match")
    spatial = PatchQualityAccumulator(reference[0].shape[0], reference[0].shape[1], patch_size)
    if valid_frames is None:
        valid_frames = [True] * len(reference)
    if len(valid_frames) != len(reference):
        raise ValueError("ROI validity and source frame counts must match")
    frame_rows = []
    for index, (source, decoded, roi, valid) in enumerate(zip(reference, reconstructed, masks, valid_frames)):
        visible = visible_masks[index] if visible_masks is not None else None
        values = (task_frame_metrics(source, decoded, roi, patch_size,
                                     psnr_threshold, ssim_threshold, visible)
                  if valid else {key: None for key in (
                      "roi_mse", "roi_psnr", "roi_ssim", "roi_normalized_l1",
                      "critical_patch_retention", "roi_visible_coverage", "roi_pixels")})
        frame_rows.append({**metadata, "local_frame_id": index, "roi_valid": bool(valid), **values})
        spatial.update(source, decoded, visible)
    valid_rows = [row for row in frame_rows if row["roi_valid"]]
    if not valid_rows:
        raise ValueError("No valid task ROI frames")
    mean = {key: float(np.mean([row[key] for row in valid_rows]))
            for key in ("roi_ssim", "roi_normalized_l1", "critical_patch_retention")}
    mean["roi_psnr"] = float(10 * np.log10(255.0 ** 2 / max(
        np.mean([row["roi_mse"] for row in valid_rows]), 1e-12)))
    mean["roi_visible_coverage"] = (
        float(np.mean([row["roi_visible_coverage"] for row in valid_rows]))
        if visible_masks is not None else None
    )
    mean["roi_valid_frames"] = len(valid_rows)
    mean["roi_valid_fraction"] = len(valid_rows) / len(frame_rows)
    return mean, frame_rows, spatial.rows_for(metadata)
