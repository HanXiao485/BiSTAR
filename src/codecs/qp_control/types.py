from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class SequenceInfo:
    sequence_id: str
    width: int
    height: int
    padded_width: int
    padded_height: int
    fps: float
    num_frames: int
    target_bitrate_kbps: Optional[float]

    @property
    def grid_shape(self) -> Tuple[int, int]:
        return ((self.height + 15) // 16, (self.width + 15) // 16)


@dataclass(frozen=True)
class QPContext:
    sequence_id: str
    frame_id: int
    pts: int
    time_base: Tuple[int, int]
    original_size: Tuple[int, int]
    padded_size: Tuple[int, int]
    grid_shape: Tuple[int, int]
    target_bitrate_kbps: Optional[float]
    roi_coverage: Optional[np.ndarray] = None
    roi_available: bool = False
    rgb: Optional[np.ndarray] = None
    rgb_available: bool = False
    base_qp: Optional[float] = None
    base_qp_available: bool = False


@dataclass
class QPDecision:
    mode: str
    values: np.ndarray
    frame_id: int
    coordinate_version: str = "source-raster-mb16-v1"
    metadata: Dict[str, Any] = field(default_factory=dict)
