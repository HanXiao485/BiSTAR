import subprocess

import pytest

from src.codecs import H264Adapter, H265Adapter
from src.common.config import load_codec_config


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
