import numpy as np


class MaskAdapter:
    """Convert BiSTAR visible=True spatial masks to VideoMAE hidden=True token masks.

    Tokens are flattened in upstream VideoMAE's time-major order: T, H, W.
    """
    def __init__(self, num_frames=16, tubelet_size=2):
        if num_frames % tubelet_size: raise ValueError("num_frames must be divisible by tubelet_size")
        self.temporal_tokens = num_frames // tubelet_size

    def build(self, visible_mask_2d):
        visible = np.asarray(visible_mask_2d)
        if visible.ndim != 2 or visible.dtype != np.bool_: raise ValueError("visible mask must be a 2-D bool array")
        return np.broadcast_to(~visible, (self.temporal_tokens,) + visible.shape).copy().reshape(-1)
