from .ffmpeg_common import FFmpegAdapter


class H264Adapter(FFmpegAdapter):
    codec_name = "libx264"
    muxer = "h264"
    extension = "h264"
