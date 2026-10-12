import numpy as np

from .base import PatchSelector
from .edge_selector import EdgeSelector
from .gaze_selector import GazeSelector
from .temporal_ssim_selector import TemporalSSIMSelector


def _normalize(scores):
    scores = np.asarray(scores, dtype=np.float32)
    span = float(scores.max() - scores.min())
    return np.zeros_like(scores) if span == 0 else (scores - scores.min()) / span


class _GazeCombined(PatchSelector):
    component_class = None
    label = ""

    def __init__(self, patch_size=16, **kwargs):
        super().__init__(patch_size); self.gaze = GazeSelector(patch_size=patch_size); self.component = self.component_class(patch_size=patch_size)

    def select(self, current_frame, previous_frame=None, gaze_point=None, budget=None):
        gaze = self.gaze.select(current_frame, previous_frame, gaze_point)
        component = self.component.select(current_frame, previous_frame)
        scores = _normalize(gaze["scores"]) + _normalize(component["scores"])
        mask = self.top_k(scores, budget) if budget is not None else np.logical_or(gaze["visible_mask"], component["visible_mask"])
        return self._result(mask, scores, {"selector": self.label})


class GazeTemporalSelector(_GazeCombined):
    component_class = TemporalSSIMSelector
    label = "gaze_temporal"


class GazeEdgeSelector(_GazeCombined):
    component_class = EdgeSelector
    label = "gaze_edge"
