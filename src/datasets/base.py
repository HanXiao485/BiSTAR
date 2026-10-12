from abc import ABC, abstractmethod


class VideoDataset(ABC):
    @abstractmethod
    def sequences(self):
        raise NotImplementedError
