from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .types import QPContext, QPDecision, SequenceInfo


class QPMapProvider(ABC):
    required_features = ()

    def reset(self, sequence_info: SequenceInfo, config: dict) -> None:
        self.sequence_info = sequence_info
        self.config = config

    @abstractmethod
    def select(self, context: QPContext) -> QPDecision:
        raise NotImplementedError

    def update(self, feedback: dict) -> None:
        return None

    def close(self) -> None:
        return None


class UniformProvider(QPMapProvider):
    def select(self, context: QPContext) -> QPDecision:
        value = float(self.config.get("delta", 0.0))
        return QPDecision("delta", np.full(context.grid_shape, value, dtype=np.float32),
                          context.frame_id, metadata={"provider": "uniform"})


class FixedROIProvider(QPMapProvider):
    def select(self, context: QPContext) -> QPDecision:
        if not context.roi_available or context.roi_coverage is None:
            if self.config.get("missing_roi_policy", "uniform") != "uniform":
                raise ValueError("ROI is unavailable and missing_roi_policy is not uniform")
            value = float(self.config.get("missing_roi_delta", 0.0))
            return QPDecision("delta", np.full(context.grid_shape, value, dtype=np.float32),
                              context.frame_id,
                              metadata={"provider": "fixed_roi", "roi_fallback": True})
        coverage = np.asarray(context.roi_coverage, dtype=np.float32)
        if coverage.shape != context.grid_shape:
            raise ValueError("ROI coverage shape {} does not match {}".format(
                coverage.shape, context.grid_shape))
        roi_delta = float(self.config["roi_delta"])
        background = self.config.get("non_roi", {})
        policy = background.get("policy", "inherit")
        if policy not in ("inherit", "fixed_delta"):
            raise ValueError("fixed_roi non_roi.policy must be inherit or fixed_delta")
        background_delta = 0.0 if policy == "inherit" else float(background.get("delta", 0.0))
        mixed = self.config.get("mixed_block_policy", "any_overlap")
        if mixed == "any_overlap":
            values = np.where(coverage > 0, roi_delta, background_delta)
        elif mixed == "coverage_threshold":
            threshold = float(self.config.get("coverage_threshold", 0.5))
            if not 0 <= threshold <= 1:
                raise ValueError("coverage_threshold must be in [0, 1]")
            values = np.where(coverage >= threshold, roi_delta, background_delta)
        elif mixed == "soft_blend":
            values = coverage * roi_delta + (1.0 - coverage) * background_delta
        else:
            raise ValueError("Unknown mixed_block_policy: {}".format(mixed))
        return QPDecision("delta", values.astype(np.float32), context.frame_id,
                          metadata={"provider": "fixed_roi", "roi_fallback": False})


class ExternalMapProvider(QPMapProvider):
    def reset(self, sequence_info: SequenceInfo, config: dict) -> None:
        super().reset(sequence_info, config)
        path = Path(config.get("external_map_path") or "")
        if not path.is_file():
            raise ValueError("external_map_path does not exist: {}".format(path))
        loaded = np.load(path, allow_pickle=False)
        try:
            if isinstance(loaded, np.ndarray):
                maps = loaded
                frame_ids = np.arange(len(maps), dtype=np.int64)
            else:
                if "maps" not in loaded or "frame_ids" not in loaded:
                    raise ValueError("External NPZ requires maps and frame_ids arrays")
                maps = loaded["maps"]
                frame_ids = loaded["frame_ids"]
            if maps.ndim != 3 or len(frame_ids) != len(maps):
                raise ValueError("External maps must have shape [frames, rows, columns]")
            if len(set(int(value) for value in frame_ids)) != len(frame_ids):
                raise ValueError("External map contains duplicate frame_ids")
            if tuple(maps.shape[1:]) != sequence_info.grid_shape:
                raise ValueError("External map grid {} does not match {}".format(
                    tuple(maps.shape[1:]), sequence_info.grid_shape))
            self.maps: Dict[int, np.ndarray] = {
                int(frame_id): np.asarray(values, dtype=np.float32)
                for frame_id, values in zip(frame_ids, maps)
            }
        finally:
            if hasattr(loaded, "close"):
                loaded.close()

    def select(self, context: QPContext) -> QPDecision:
        if context.frame_id not in self.maps:
            raise ValueError("External map is missing source frame {}".format(context.frame_id))
        return QPDecision("delta", self.maps[context.frame_id].copy(), context.frame_id,
                          metadata={"provider": "external_map"})


PROVIDERS = {
    "uniform": UniformProvider,
    "fixed_roi": FixedROIProvider,
    "external_map": ExternalMapProvider,
}


def create_provider(config: dict) -> QPMapProvider:
    name = config.get("provider")
    if name not in PROVIDERS:
        raise ValueError("Unknown QP map provider: {}".format(name))
    return PROVIDERS[name]()


def validate_decision(decision: QPDecision, context: QPContext, limits) -> np.ndarray:
    if decision.mode != "delta":
        raise ValueError("This backend supports delta QP maps only")
    if int(decision.frame_id) != int(context.frame_id):
        raise ValueError("QP decision frame_id does not match source frame")
    values = np.asarray(decision.values, dtype=np.float32)
    if values.shape != context.grid_shape:
        raise ValueError("QP map shape {} does not match {}".format(values.shape, context.grid_shape))
    if not np.isfinite(values).all():
        raise ValueError("QP map contains NaN or infinite values")
    lower, upper = float(limits[0]), float(limits[1])
    if lower > upper:
        raise ValueError("delta_limits lower bound exceeds upper bound")
    if np.any(values < lower) or np.any(values > upper):
        raise ValueError("QP map contains values outside delta_limits [{}, {}]".format(lower, upper))
    return np.ascontiguousarray(values)
