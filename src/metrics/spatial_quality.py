import numpy as np
from skimage.metrics import structural_similarity


class PatchQualityAccumulator:
    """Aggregate spatial quality without storing one row per frame and patch."""

    def __init__(self, image_height, image_width, patch_size):
        if image_height % patch_size or image_width % patch_size:
            raise ValueError("Frame dimensions must be divisible by patch_size")
        self.patch_size = int(patch_size)
        self.rows = image_height // patch_size
        self.columns = image_width // patch_size
        self.squared_error = np.zeros((self.rows, self.columns), dtype=np.float64)
        self.absolute_error = np.zeros((self.rows, self.columns), dtype=np.float64)
        self.ssim_sum = np.zeros((self.rows, self.columns), dtype=np.float64)
        self.visible_frames = np.zeros((self.rows, self.columns), dtype=np.float64)
        self.frame_count = 0

    def update(self, reference, reconstructed, visible_mask=None):
        patch = self.patch_size
        error = reference.astype(np.float64) - reconstructed.astype(np.float64)
        squared = (error ** 2).reshape(self.rows, patch, self.columns, patch, 3).mean((1, 3, 4))
        absolute = np.abs(error).reshape(self.rows, patch, self.columns, patch, 3).mean((1, 3, 4))
        _, ssim_map = structural_similarity(
            reference, reconstructed, channel_axis=2, data_range=255, full=True
        )
        if ssim_map.ndim == 3:
            ssim_map = ssim_map.mean(axis=2)
        patch_ssim = ssim_map.reshape(self.rows, patch, self.columns, patch).mean((1, 3))
        self.squared_error += squared
        self.absolute_error += absolute
        self.ssim_sum += patch_ssim
        if visible_mask is not None:
            mask = np.asarray(visible_mask, dtype=bool)
            if mask.shape != (self.rows, self.columns):
                raise ValueError("Visible mask shape does not match patch grid")
            self.visible_frames += mask
        self.frame_count += 1

    def rows_for(self, metadata, selection_frequency=None):
        if self.frame_count < 1:
            return []
        mse = self.squared_error / self.frame_count
        psnr = np.full_like(mse, np.inf)
        nonzero = mse > 0
        psnr[nonzero] = 10.0 * np.log10(255.0 ** 2 / mse[nonzero])
        frequency = (self.visible_frames / self.frame_count if selection_frequency is None
                     else np.asarray(selection_frequency, dtype=np.float64))
        result = []
        for row in range(self.rows):
            for column in range(self.columns):
                result.append({
                    **metadata,
                    "patch_row": row,
                    "patch_column": column,
                    "patch_index": row * self.columns + column,
                    "patch_psnr": float(psnr[row, column]),
                    "patch_ssim": float(self.ssim_sum[row, column] / self.frame_count),
                    "patch_normalized_l1": float(
                        self.absolute_error[row, column] / self.frame_count / 255.0
                    ),
                    "selection_frequency": float(frequency[row, column]),
                })
        return result
