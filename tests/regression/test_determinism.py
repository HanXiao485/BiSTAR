import subprocess

from src.codecs import H264Adapter
from src.common.config import load_codec_config


def test_h264_size_is_deterministic(tmp_path):
    source=tmp_path/"source.mkv"; subprocess.run(["ffmpeg","-y","-f","lavfi","-i","testsrc2=size=224x224:rate=15","-t","0.5","-c:v","ffv1",str(source)],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    cfg=load_codec_config("configs/codecs/h264.yaml"); cfg["preprocessing"]={"target_width":224,"target_height":224,"target_fps":15}
    a=H264Adapter().encode_sequence(source,tmp_path/"a",cfg); b=H264Adapter().encode_sequence(source,tmp_path/"b",cfg)
    assert a["total_bytes"] == b["total_bytes"]
