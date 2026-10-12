import numpy as np
import torch

from src.bistar.reconstruction.videomae_adapter import VideoMAEAdapter
from src.common.config import load_yaml


class DummyModel(torch.nn.Module):
    def __init__(self): super().__init__(); self.anchor = torch.nn.Parameter(torch.zeros(1))
    def forward(self, clip, mask):
        return torch.zeros((clip.shape[0], int(mask[0].sum()), 1536), device=clip.device)


def test_hidden_pixels_do_not_affect_videomae_input_or_output():
    adapter = VideoMAEAdapter(load_yaml("configs/models/videomae.yaml")); adapter.model = DummyModel()
    visible = np.zeros((14, 14), dtype=bool); visible[0, 0] = True
    a = np.zeros((16, 224, 224, 3), dtype=np.uint8); b = np.full_like(a, 255)
    a[:, :16, :16] = 80; b[:, :16, :16] = 80
    out_a = adapter.reconstruct_clip(list(a), visible); out_b = adapter.reconstruct_clip(list(b), visible)
    assert np.array_equal(out_a, out_b)
    assert np.array_equal(out_a[:, :16, :16], a[:, :16, :16])
