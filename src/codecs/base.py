from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict


class CodecAdapter(ABC):
    @abstractmethod
    def encode_sequence(self, input_video: Path, output_dir: Path, config: dict) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def decode_sequence(self, bitstream_path: Path, output_dir: Path, config: dict) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def probe(self, output_path: Path) -> Dict[str, Any]:
        raise NotImplementedError
