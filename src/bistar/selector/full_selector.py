import numpy as np

from .base import PatchSelector
from .gaze_selector import GazeSelector
from .structural_selector import StructuralSelector


class FullSelector(PatchSelector):
    def __init__(self, patch_size=16, **kwargs):
        super().__init__(patch_size); self.gaze = GazeSelector(patch_size=patch_size); self.structural = StructuralSelector(patch_size=patch_size)

    def select(self, current_frame, previous_frame=None, gaze_point=None, budget=None):
        gaze = self.gaze.select(current_frame, previous_frame, gaze_point); structural = self.structural.select(current_frame, previous_frame)
        scores = structural["scores"] + gaze["scores"] / max(current_frame.shape[:2])
        mask = self.top_k(scores, budget) if budget is not None else np.logical_or(gaze["visible_mask"], structural["visible_mask"])
        return self._result(mask, scores, {"selector": "bistar_full"})
