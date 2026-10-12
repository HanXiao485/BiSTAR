from pathlib import Path

from .base import VideoDataset
from .manifest import read_manifest


class ManifestVideoDataset(VideoDataset):
    def __init__(self, manifest: Path, split: str):
        self.rows = read_manifest(manifest, split)

    def sequences(self):
        return iter(self.rows)
