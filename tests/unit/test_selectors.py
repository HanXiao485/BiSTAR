import numpy as np
import pytest

from src.bistar.selector import EdgeSelector, FullSelector, GazeEdgeSelector, GazeSelector, GazeTemporalSelector, RandomSelector, StructuralSelector, TemporalSSIMSelector


@pytest.mark.parametrize("cls", [RandomSelector, GazeSelector, TemporalSSIMSelector, EdgeSelector, StructuralSelector, GazeTemporalSelector, GazeEdgeSelector, FullSelector])
def test_fixed_budget(cls):
    frame = np.zeros((224,224,3), dtype=np.uint8); previous = frame.copy(); frame[32:80,32:80] = 255
    result = cls().select(frame, previous, (112,112,1), 40)
    assert result["visible_mask"].shape == (14,14)
    assert len(result["selected_indices"]) == 40
