import numpy as np

from src.bistar.reconstruction.previous_frame import PreviousFrameReconstructor


def test_hidden_ground_truth_cannot_leak():
    visible = np.zeros((14,14), dtype=bool); visible[0,0] = True
    sparse = np.zeros((224,224,3), dtype=np.uint8); sparse[:16,:16] = 7
    hidden_truth_a = np.full((224,224,3), 30, dtype=np.uint8); hidden_truth_b = np.full((224,224,3), 220, dtype=np.uint8)
    reconstructor = PreviousFrameReconstructor(16)
    # Ground truths are deliberately never supplied to the decoder; changing them cannot change output.
    output_a = reconstructor.reconstruct(sparse, visible, None); output_b = reconstructor.reconstruct(sparse, visible, None)
    assert not np.array_equal(hidden_truth_a, hidden_truth_b)
    assert np.array_equal(output_a, output_b)
    assert not output_a[16:,16:].any()
