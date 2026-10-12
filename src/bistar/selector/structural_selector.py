import numpy as np

from .base import PatchSelector
from .edge_selector import EdgeSelector
from .temporal_ssim_selector import TemporalSSIMSelector


class StructuralSelector(PatchSelector):
    def __init__(self, patch_size=16, **kwargs):
        super().__init__(patch_size); self.temporal = TemporalSSIMSelector(patch_size=patch_size); self.edge = EdgeSelector(patch_size=patch_size)

    def select(self, current_frame, previous_frame=None, gaze_point=None, budget=None):
        a = self.temporal.select(current_frame, previous_frame); b = self.edge.select(current_frame, previous_frame)
        scores = a["scores"] + b["scores"]
        mask = self.top_k(scores, budget) if budget is not None else np.logical_or(a["visible_mask"], b["visible_mask"])
        return self._result(mask, scores, {"selector": "structural"})
