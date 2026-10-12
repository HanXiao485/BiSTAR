import json
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

from .base import CodecAdapter
from src.common.types import DecodedFrame
from src.datasets.frame_reader import read_rgb_frames


def shell_join(command):
    return " ".join(shlex.quote(str(part)) for part in command)


def actual_bitrate_kbps(total_bytes: int, duration_sec: float) -> float:
    if duration_sec <= 0:
        raise ValueError("duration_sec must be positive")
    return total_bytes * 8.0 / duration_sec / 1000.0


class FFmpegAdapter(CodecAdapter):
    codec_name = ""
    muxer = ""
    extension = ""

    def _run(self, command: List[str], log_path: Path) -> float:
        started = time.perf_counter()
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("$ " + shell_join(command) + "\n\nSTDOUT\n" + result.stdout + "\nSTDERR\n" + result.stderr, encoding="utf-8")
        if result.returncode != 0:
            raise RuntimeError("FFmpeg failed; see {}".format(log_path))
        return time.perf_counter() - started

    def probe(self, output_path: Path) -> Dict[str, Any]:
        command = ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                   "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames,duration:format=duration", "-of", "json", str(output_path)]
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise RuntimeError("ffprobe failed for {}: {}".format(output_path, result.stderr))
        data = json.loads(result.stdout)
        stream = data["streams"][0]
        duration = float(stream.get("duration") or data.get("format", {}).get("duration") or 0)
        rate = stream.get("avg_frame_rate", "0/1").split("/")
        fps = float(rate[0]) / float(rate[1]) if float(rate[1]) else 0.0
        frames = int(stream.get("nb_read_frames") or round(duration * fps))
        return {"width": int(stream["width"]), "height": int(stream["height"]), "fps": fps, "num_frames": frames, "duration_sec": duration}

    def encode_sequence(self, input_video: Path, output_dir: Path, config: dict) -> Dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        codec = config["codec"]
        encoder = codec["encoder"]
        prep = config.get("preprocessing", {})
        fps, width, height = prep.get("target_fps", 15), prep.get("target_width", 224), prep.get("target_height", 224)
        output = output_dir / (input_video.stem + "." + codec["output"].get("extension", self.extension))
        command = ["ffmpeg", "-y", "-i", str(input_video), "-an", "-vf", "fps={},scale={}:{},format={}".format(fps, width, height, codec["pixel_format"]),
                   "-c:v", encoder.get("codec_name", self.codec_name), "-preset", str(encoder["preset"])]
        mode = encoder["mode"]
        if mode == "bitrate":
            command += ["-b:v", "{}k".format(encoder["bitrate_kbps"])]
            if encoder.get("maxrate_kbps") is not None: command += ["-maxrate", "{}k".format(encoder["maxrate_kbps"])]
            if encoder.get("bufsize_kbps") is not None: command += ["-bufsize", "{}k".format(encoder["bufsize_kbps"])]
        elif mode == "crf":
            command += ["-crf", str(encoder["crf"])]
        else:
            raise ValueError("Unsupported encoder mode: {}".format(mode))
        command += ["-g", str(encoder["gop_size"]), "-bf", str(encoder["b_frames"])]
        if encoder.get("tune"): command += ["-tune", str(encoder["tune"])]
        command += ["-f", codec["output"]["container"], str(output)]
        elapsed = self._run(command, output_dir / "encode.log")
        probe = self.probe(output)
        size = output.stat().st_size
        duration = probe["num_frames"] / fps
        return {"bitstream_path": str(output), "num_frames": probe["num_frames"], "duration_sec": duration,
                "payload_bytes": size, "metadata_bytes": 0, "total_bytes": size,
                "actual_bitrate_kbps": actual_bitrate_kbps(size, duration), "encode_time_sec": elapsed,
                "ffmpeg_command": shell_join(command)}

    def decode_sequence(self, bitstream_path: Path, output_dir: Path, config: dict) -> Dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / "decoded.mkv"
        command = ["ffmpeg", "-y", "-i", str(bitstream_path), "-an", "-c:v", "ffv1", "-pix_fmt", "rgb24", str(output)]
        elapsed = self._run(command, output_dir / "decode.log")
        probe = self.probe(output)
        probe.update({"decoded_path": str(output), "decode_time_sec": elapsed, "ffmpeg_command": shell_join(command)})
        return probe

    def decoded_frames(self, sequence_id, decode_result, encode_result, capture_timestamps_ns=None):
        frames = read_rgb_frames(Path(decode_result["decoded_path"]))
        count = len(frames); payload_base, remainder = divmod(int(encode_result["payload_bytes"]), count)
        per_frame_decode_ns = int(decode_result["decode_time_sec"] * 1e9 / count)
        records = []
        for frame_id, rgb in enumerate(frames):
            payload = payload_base + (1 if frame_id < remainder else 0)
            capture = capture_timestamps_ns[frame_id] if capture_timestamps_ns else None
            records.append(DecodedFrame(sequence_id, frame_id, rgb, capture_timestamp_ns=capture,
                                        decode_start_timestamp_ns=frame_id * per_frame_decode_ns,
                                        decode_end_timestamp_ns=(frame_id + 1) * per_frame_decode_ns,
                                        display_ready_timestamp_ns=(frame_id + 1) * per_frame_decode_ns,
                                        payload_bytes=payload, metadata_bytes=0, total_bytes=payload,
                                        method_metadata={"codec": self.codec_name}))
        return records
