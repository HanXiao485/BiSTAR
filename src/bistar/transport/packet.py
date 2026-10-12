from dataclasses import dataclass
from typing import List


@dataclass
class BiSTARPacket:
    version: int
    sequence_id: int
    frame_id: int
    timestamp_ns: int
    image_width: int
    image_height: int
    patch_size: int
    selected_indices: List[int]
    payload_codec: str
    payload: bytes
