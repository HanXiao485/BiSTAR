from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class FrameRecord:
    sequence_id: str
    frame_id: int
    timestamp_ns: int
    rgb: np.ndarray


@dataclass
class EncodedPacket:
    sequence_id: str
    frame_id: int
    payload: bytes
    payload_bytes: int
    metadata_bytes: int
    total_bytes: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecodedFrame:
    sequence_id: str
    frame_id: int
    rgb: np.ndarray
    capture_timestamp_ns: Optional[int] = None
    encode_start_timestamp_ns: Optional[int] = None
    encode_end_timestamp_ns: Optional[int] = None
    send_timestamp_ns: Optional[int] = None
    receive_timestamp_ns: Optional[int] = None
    decode_start_timestamp_ns: Optional[int] = None
    decode_end_timestamp_ns: Optional[int] = None
    display_ready_timestamp_ns: Optional[int] = None
    payload_bytes: int = 0
    metadata_bytes: int = 0
    total_bytes: int = 0
    method_metadata: Dict[str, Any] = field(default_factory=dict)
