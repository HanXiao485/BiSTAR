import numpy as np

from src.bistar.reconstruction.mask_adapter import MaskAdapter


def test_shape_order_and_semantics():
    visible = np.zeros((14, 14), dtype=bool); visible[2, 3] = True
    mask = MaskAdapter(16, 2).build(visible)
    assert mask.shape == (1568,)
    cube = mask.reshape(8, 14, 14)
    assert not cube[:, 2, 3].any()
    assert cube[:, 0, 0].all()
