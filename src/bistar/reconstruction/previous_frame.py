import numpy as np

from .base import Reconstructor


def pixel_mask(visible_mask, patch_size):
    return np.repeat(np.repeat(visible_mask, patch_size, axis=0), patch_size, axis=1)


class PreviousFrameReconstructor(Reconstructor):
    def __init__(self, patch_size=16): self.patch_size = patch_size

    def reconstruct(self, sparse_frame, visible_mask, previous_frame=None):
        # Only sparse_frame and decoder state are accepted, preventing hidden ground-truth access.
        base = np.zeros_like(sparse_frame) if previous_frame is None else previous_frame.copy()
        mask = pixel_mask(visible_mask, self.patch_size)
        base[mask] = sparse_frame[mask]
        return base
