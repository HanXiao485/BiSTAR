import numpy as np
import pytest

from src.bistar.transport.bitrate_accounting import bitrate_breakdown, bitrate_kbps
from src.experiments.bistar_runner import resolve_patch_budget_options, select_budget_for_rate


def test_known_bitrate(): assert abs(bitrate_kbps(10000, 1.0) - 80.0) < 1e-6


def test_payload_metadata_breakdown():
    values = bitrate_breakdown([1000] * 15, [100] * 15, 15)
    assert values["payload_bitrate_kbps"] == 120.0
    assert values["metadata_bitrate_kbps"] == 12.0
    assert values["total_bitrate_kbps"] == 132.0


def test_fixed_visible_patch_budget():
    frames = [np.full((32, 32, 3), value, dtype=np.uint8) for value in range(16)]
    selected = select_budget_for_rate(
        frames, target_kbps=100, fps=15, patch_size=16, fixed_visible_patch_count=3
    )
    assert selected["budget_mode"] == "fixed"
    assert selected["budget"] == 3
    assert len(selected["encoded"]["selected_indices"]) == 3


def test_fixed_visible_patch_budget_range_is_validated():
    frames = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(16)]
    with pytest.raises(ValueError, match="between 1 and 4"):
        select_budget_for_rate(
            frames, target_kbps=100, fps=15, patch_size=16, fixed_visible_patch_count=5
        )


def test_patch_budget_mode_is_boolean():
    assert resolve_patch_budget_options({"bistar": {"patch_budget_mode": True, "visible_patch_count": 40}}) == (True, 40)
    assert resolve_patch_budget_options({"bistar": {"patch_budget_mode": False, "visible_patch_count": 40}}) == (False, None)
    with pytest.raises(ValueError, match="true or false"):
        resolve_patch_budget_options({"bistar": {"patch_budget_mode": "fixed", "visible_patch_count": 40}})
