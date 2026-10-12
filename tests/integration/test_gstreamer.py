import shutil
import subprocess

import pytest


@pytest.mark.parametrize("encoder,parser", [("x264enc", "h264parse"), ("x265enc", "h265parse")])
def test_gstreamer_codec_pipeline(encoder, parser):
    if not shutil.which("gst-launch-1.0"): pytest.skip("GStreamer is not installed")
    command = ["gst-launch-1.0", "-q", "videotestsrc", "num-buffers=5", "!",
               "video/x-raw,width=224,height=224,framerate=15/1", "!", "videoconvert", "!",
               encoder, "tune=zerolatency", "!", parser, "!", "fakesink"]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
