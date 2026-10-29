import numpy as np


class MaskAdapter:
    """Convert BiSTAR visible=True spatial masks to VideoMAE hidden=True token masks.

    Tokens are flattened in upstream VideoMAE's time-major order: T, H, W.
    """
    def __init__(self, num_frames=16, tubelet_size=2):
        if num_frames % tubelet_size: raise ValueError("num_frames must be divisible by tubelet_size")
        self.num_frames = int(num_frames)
        self.tubelet_size = int(tubelet_size)
        self.temporal_tokens = num_frames // tubelet_size

    def build(self, visible_mask_2d):
        visible = np.asarray(visible_mask_2d)
        if visible.ndim != 2 or visible.dtype != np.bool_: raise ValueError("visible mask must be a 2-D bool array")
        return np.broadcast_to(~visible, (self.temporal_tokens,) + visible.shape).copy().reshape(-1)

    def build_dynamic(self, visible_masks_3d):
        """Build a token mask for per-frame visibility.

        A tubelet is visible only when its spatial patch is available in every
        constituent frame; transmitted pixels are copied back after inference.
        """
        visible = np.asarray(visible_masks_3d)
        if visible.ndim != 3 or visible.dtype != np.bool_:
            raise ValueError("dynamic visible mask must be a 3-D bool array")
        if visible.shape[0] != self.num_frames:
            raise ValueError("dynamic visible mask has incompatible temporal length")
        token_visible = visible.reshape(
            self.temporal_tokens, self.tubelet_size, *visible.shape[1:]).all(axis=1)
        return (~token_visible).reshape(-1)
