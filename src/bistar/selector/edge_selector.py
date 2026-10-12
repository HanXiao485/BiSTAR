import cv2
import numpy as np

from .base import PatchSelector


class EdgeSelector(PatchSelector):
    def __init__(self, threshold_low=50, threshold_high=150, patch_size=16):
        super().__init__(patch_size); self.low = threshold_low; self.high = threshold_high

    def select(self, current_frame, previous_frame=None, gaze_point=None, budget=None):
        ph, pw = self._shape(current_frame)
        edges = cv2.Canny(cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY), self.low, self.high).astype(np.float32) / 255.0
        scores = edges.reshape(ph, self.patch_size, pw, self.patch_size).mean((1, 3))
        mask = self.top_k(scores, budget) if budget is not None else scores > 0
        return self._result(mask, scores, {"selector": "edge"})
