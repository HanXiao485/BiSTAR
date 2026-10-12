from .ffmpeg_common import FFmpegAdapter


class H265Adapter(FFmpegAdapter):
    codec_name = "libx265"
    muxer = "hevc"
    extension = "hevc"
