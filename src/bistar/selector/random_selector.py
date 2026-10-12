import numpy as np

from .base import PatchSelector


class RandomSelector(PatchSelector):
    def __init__(self, num_visible_patches=40, seed=0, patch_size=16):
        super().__init__(patch_size); self.num_visible_patches = num_visible_patches; self.rng = np.random.default_rng(seed)

    def select(self, current_frame, previous_frame=None, gaze_point=None, budget=None):
        shape = self._shape(current_frame); count = min(budget or self.num_visible_patches, shape[0] * shape[1])
        mask = np.zeros(shape[0] * shape[1], dtype=bool)
        mask[self.rng.choice(mask.size, count, replace=False)] = True
        return self._result(mask.reshape(shape), metadata={"selector": "random"})
