import cv2
import numpy as np
from skimage.metrics import structural_similarity

from .base import PatchSelector


class TemporalSSIMSelector(PatchSelector):
    def __init__(self, threshold=0.15, top_k=None, patch_size=16):
        super().__init__(patch_size); self.threshold = threshold; self.fixed_top_k = top_k

    def select(self, current_frame, previous_frame=None, gaze_point=None, budget=None):
        ph, pw = self._shape(current_frame)
        if previous_frame is None: scores = np.ones((ph, pw), dtype=np.float32)
        else:
            current_gray = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
            previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_RGB2GRAY)
            scores = np.empty((ph, pw), dtype=np.float32)
            for row in range(ph):
                for col in range(pw):
                    y, x = row * self.patch_size, col * self.patch_size
                    a = current_gray[y:y+self.patch_size, x:x+self.patch_size]
                    b = previous_gray[y:y+self.patch_size, x:x+self.patch_size]
                    scores[row, col] = 1.0 - structural_similarity(a, b, data_range=255)
        count = budget if budget is not None else self.fixed_top_k
        mask = self.top_k(scores, count) if count is not None else scores >= self.threshold
        return self._result(mask, scores, {"selector": "temporal_ssim", "score": "one_minus_patch_ssim"})
