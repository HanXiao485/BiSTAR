from abc import ABC, abstractmethod


class Reconstructor(ABC):
    @abstractmethod
    def reconstruct(self, sparse_frame, visible_mask, previous_frame=None):
        raise NotImplementedError
