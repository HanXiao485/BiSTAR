import subprocess

import numpy as np
import pytest

from src.codecs import H264Adapter, H265Adapter
from src.common.config import load_codec_config
from src.datasets.frame_reader import read_rgb_frames, write_rgb_video


@pytest.mark.parametrize("name,adapter", [("h264", H264Adapter), ("h265", H265Adapter)])
def test_encode_decode(tmp_path, name, adapter):
    source = tmp_path / "source.mkv"
    subprocess.run(["ffmpeg","-y","-f","lavfi","-i","testsrc2=size=224x224:rate=15","-t","1","-c:v","ffv1",str(source)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    cfg = load_codec_config("configs/codecs/{}.yaml".format(name)); cfg["preprocessing"] = {"target_width":224,"target_height":224,"target_fps":15}
    encoded = adapter().encode_sequence(source, tmp_path/name, cfg); decoded = adapter().decode_sequence(encoded["bitstream_path"], tmp_path/name/"decoded", cfg)
    assert encoded["total_bytes"] > 0 and encoded["num_frames"] == 15
    assert decoded["num_frames"] == 15 and decoded["width"] == 224 and decoded["height"] == 224
    records = adapter().decoded_frames("test", decoded, encoded)
    assert [record.frame_id for record in records] == list(range(15))
    assert sum(record.total_bytes for record in records) == encoded["total_bytes"]


def test_write_reconstruction_video(tmp_path):
    frames = np.stack([np.full((224, 224, 3), value, dtype=np.uint8) for value in (0, 64, 128)])
    output = tmp_path / "reconstructed.mp4"
    result = write_rgb_video(frames, output, fps=15)
    decoded = read_rgb_frames(output)
    assert output.stat().st_size == result["bytes"] > 0
    assert result["num_frames"] == len(decoded) == 3
    assert decoded[0].shape == (224, 224, 3)
