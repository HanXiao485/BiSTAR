import cv2
import numpy as np

from .base import Reconstructor
from .previous_frame import pixel_mask


class InterpolationReconstructor(Reconstructor):
    def __init__(self, patch_size=16): self.patch_size = patch_size

    def reconstruct(self, sparse_frame, visible_mask, previous_frame=None):
        mask = pixel_mask(visible_mask, self.patch_size)
        if not mask.any(): return np.zeros_like(sparse_frame) if previous_frame is None else previous_frame.copy()
        result = sparse_frame.copy(); missing = (~mask).astype(np.uint8) * 255
        for channel in range(3): result[:, :, channel] = cv2.inpaint(result[:, :, channel], missing, 3, cv2.INPAINT_TELEA)
        return result
