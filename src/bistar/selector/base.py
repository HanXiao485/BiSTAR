from abc import ABC, abstractmethod

import numpy as np


class PatchSelector(ABC):
    def __init__(self, patch_size=16):
        self.patch_size = patch_size

    def _shape(self, frame):
        h, w = frame.shape[:2]
        if h % self.patch_size or w % self.patch_size:
            raise ValueError("Frame dimensions must be divisible by patch_size")
        return h // self.patch_size, w // self.patch_size

    def _result(self, mask, scores=None, metadata=None):
        indices = np.flatnonzero(mask.reshape(-1)).astype(int).tolist()
        return {"visible_mask": mask.astype(bool), "selected_indices": indices, "scores": scores, "metadata": metadata or {}}

    @staticmethod
    def top_k(scores, budget):
        flat = scores.reshape(-1)
        count = max(0, min(int(budget), flat.size))
        mask = np.zeros(flat.size, dtype=bool)
        if count: mask[np.argpartition(flat, -count)[-count:]] = True
        return mask.reshape(scores.shape)

    @abstractmethod
    def select(self, current_frame, previous_frame=None, gaze_point=None, budget=None):
        raise NotImplementedError
