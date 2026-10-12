import numpy as np

from src.bistar.reconstruction.videomae_adapter import VideoMAEAdapter
from src.common.config import load_yaml


def test_official_checkpoint_forward():
    config = load_yaml("configs/models/videomae.yaml"); adapter = VideoMAEAdapter(config)
    frames = [np.full((224, 224, 3), 127 + (i % 2), np.uint8) for i in range(16)]
    visible = np.zeros((14, 14), dtype=bool); visible.reshape(-1)[:20] = True
    output = adapter.reconstruct_clip(frames, visible)
    assert output.shape == (16, 224, 224, 3) and output.dtype == np.uint8
    assert np.isfinite(output).all() and output.min() >= 0 and output.max() <= 255
