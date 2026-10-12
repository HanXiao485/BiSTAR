import numpy as np

from .base import PatchSelector


class GazeSelector(PatchSelector):
    def __init__(self, radius_pixels=48, fallback="center", patch_size=16):
        super().__init__(patch_size); self.radius_pixels = radius_pixels; self.fallback = fallback

    def select(self, current_frame, previous_frame=None, gaze_point=None, budget=None):
        ph, pw = self._shape(current_frame); h, w = current_frame.shape[:2]
        valid = gaze_point is not None and (len(gaze_point) < 3 or gaze_point[2] > 0)
        x, y = (gaze_point[0], gaze_point[1]) if valid else (w / 2, h / 2)
        yy, xx = np.mgrid[0:ph, 0:pw]; cx = (xx + .5) * self.patch_size; cy = (yy + .5) * self.patch_size
        scores = -np.sqrt((cx - x) ** 2 + (cy - y) ** 2)
        mask = self.top_k(scores, budget) if budget is not None else (-scores <= self.radius_pixels)
        return self._result(mask, scores, {"selector": "gaze", "gaze": [float(x), float(y)]})
