import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from .ffmpeg_common import FFmpegAdapter, actual_bitrate_kbps, shell_join
from .qp_control import QPContext, SequenceInfo, create_provider
from .qp_control.providers import validate_decision


class H264BlockQPAdapter(FFmpegAdapter):
    """Optional libx264 adapter using per-picture macroblock quant_offsets."""

    codec_name = "libx264_block_qp"
    muxer = "h264"
    extension = "h264"

    @staticmethod
    def capabilities():
        return {
            "modes": ["delta"], "block_size": 16, "pixel_formats": ["yuv420p"],
            "base_qp_available_before_select": False, "applied_block_qp_trace": False,
            "frame_average_crf_trace": True, "reorder_supported": False,
        }

    @staticmethod
    def _x264_version():
        result = subprocess.run(["pkg-config", "--modversion", "x264"], text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout.strip() if result.returncode == 0 else None

    def _bridge_path(self, codec):
        configured = codec.get("backend", {}).get("bridge_path", "build/x264_qp_bridge")
        path = Path(configured)
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.is_file():
            raise RuntimeError(
                "x264 QP bridge is unavailable at {}. Run scripts/build_x264_qp_bridge.sh".format(path))
        return path

    @staticmethod
    def _roi_coverage(frame_id, width, height, config):
        roi = config.get("roi", {})
        rectangles = roi.get("per_frame_rectangles", {}).get(str(frame_id), roi.get("rectangles", []))
        if not rectangles:
            return None, False
        rows, columns = (height + 15) // 16, (width + 15) // 16
        coverage = np.zeros((rows, columns), dtype=np.float32)
        for rectangle in rectangles:
            if len(rectangle) != 4:
                raise ValueError("Each ROI rectangle must be [x, y, width, height]")
            x, y, rect_width, rect_height = (float(value) for value in rectangle)
            if roi.get("coordinate_space", "pixels") == "normalized":
                x, rect_width = x * width, rect_width * width
                y, rect_height = y * height, rect_height * height
            left, top = max(0.0, x), max(0.0, y)
            right, bottom = min(float(width), x + rect_width), min(float(height), y + rect_height)
            if right <= left or bottom <= top:
                continue
            for row in range(max(0, int(top) // 16), min(rows, (int(np.ceil(bottom)) + 15) // 16)):
                for column in range(max(0, int(left) // 16), min(columns, (int(np.ceil(right)) + 15) // 16)):
                    block_left, block_top = column * 16, row * 16
                    block_right, block_bottom = min(width, block_left + 16), min(height, block_top + 16)
                    overlap = max(0.0, min(right, block_right) - max(left, block_left)) * max(
                        0.0, min(bottom, block_bottom) - max(top, block_top))
                    valid_area = (block_right - block_left) * (block_bottom - block_top)
                    coverage[row, column] = max(coverage[row, column], overlap / valid_area)
        return coverage, True

    def encode_sequence(self, input_video: Path, output_dir: Path, config: dict):
        output_dir.mkdir(parents=True, exist_ok=True)
        codec, prep = config["codec"], config.get("preprocessing", {})
        encoder, qp = codec["encoder"], codec["qp_control"]
        width, height = int(prep.get("target_width", 224)), int(prep.get("target_height", 224))
        fps = int(prep.get("target_fps", 15))
        if codec["pixel_format"] != "yuv420p" or width % 2 or height % 2:
            raise ValueError("h264_block_qp requires even-sized 8-bit yuv420p input")
        if int(qp.get("block_size", 16)) != 16 or qp.get("mode", "delta") != "delta":
            raise ValueError("h264_block_qp supports delta mode with 16x16 macroblocks only")
        raw_path = output_dir / "input.yuv"
        ffmpeg_command = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(input_video),
            "-an", "-vf", "fps={},scale={}:{},format=yuv420p".format(fps, width, height),
            "-f", "rawvideo", str(raw_path),
        ]
        started = time.perf_counter()
        result = subprocess.run(ffmpeg_command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise RuntimeError("Raw-frame preparation failed: {}".format(result.stderr.strip()))
        frame_bytes = width * height * 3 // 2
        raw_size = raw_path.stat().st_size
        if raw_size == 0 or raw_size % frame_bytes:
            raise RuntimeError("Prepared YUV stream is empty or truncated")
        num_frames = raw_size // frame_bytes
        info = SequenceInfo(input_video.stem, width, height, width, height, fps, num_frames,
                            float(encoder["bitrate_kbps"]) if encoder.get("mode") == "bitrate" else None)
        provider_cfg = dict(qp)
        external = provider_cfg.get("external_map_path")
        if external and not Path(external).is_absolute():
            provider_cfg["external_map_path"] = str(Path.cwd() / external)
        provider = create_provider(provider_cfg)
        provider.reset(info, provider_cfg)
        maps_path, requested_trace = output_dir / "qp_maps.f32", output_dir / "qp_requested.csv"
        checksum = hashlib.sha256()
        fallback_count = 0
        clipped_count = 0
        limits = qp.get("delta_limits", [-12.0, 12.0])
        try:
            with maps_path.open("wb") as maps, requested_trace.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["frame_id", "row", "col", "coverage", "requested_delta",
                                 "requested_absolute", "base_qp", "applied_qp", "available"])
                for frame_id in range(num_frames):
                    coverage, available = self._roi_coverage(frame_id, width, height, qp)
                    context = QPContext(info.sequence_id, frame_id, frame_id, (1, fps),
                                        (width, height), (width, height), info.grid_shape,
                                        info.target_bitrate_kbps, coverage, available)
                    decision = provider.select(context)
                    values = validate_decision(decision, context, limits)
                    fallback_count += int(bool(decision.metadata.get("roi_fallback", False)))
                    payload = values.astype("<f4", copy=False).tobytes(order="C")
                    maps.write(payload); checksum.update(payload)
                    for row in range(values.shape[0]):
                        for column in range(values.shape[1]):
                            writer.writerow([frame_id, row, column,
                                             None if coverage is None else float(coverage[row, column]),
                                             float(values[row, column]), None, None, None, False])
        finally:
            provider.close()
        output = output_dir / (input_video.stem + "." + codec["output"].get("extension", "h264"))
        frame_trace = output_dir / "x264_frame_trace.csv"
        bridge = self._bridge_path(codec)
        bridge_command = [
            str(bridge), "--input", str(raw_path), "--maps", str(maps_path),
            "--output", str(output), "--trace", str(frame_trace), "--width", str(width),
            "--height", str(height), "--fps", str(fps), "--bitrate-kbps", str(encoder["bitrate_kbps"]),
            "--preset", str(encoder.get("preset", "medium")), "--tune", str(encoder.get("tune") or "none"),
            "--keyint", str(encoder.get("gop_size", 30)), "--bframes", str(encoder.get("b_frames", 0)),
        ]
        if int(encoder.get("b_frames", 0)) != 0:
            raise ValueError("h264_block_qp currently requires b_frames=0 for source-frame alignment")
        bridge_result = subprocess.run(bridge_command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (output_dir / "encode.log").write_text(
            "$ " + shell_join(ffmpeg_command) + "\n" + result.stderr + "\n$ " +
            shell_join(bridge_command) + "\n" + bridge_result.stderr, encoding="utf-8")
        if bridge_result.returncode != 0:
            raise RuntimeError("x264 QP bridge failed; see {}".format(output_dir / "encode.log"))
        elapsed = time.perf_counter() - started
        probe = self.probe(output)
        size = output.stat().st_size
        duration = num_frames / float(fps)
        metadata = {
            "provider": qp["provider"], "backend": "libx264_quant_offsets",
            "backend_version": self._x264_version(),
            "backend_capabilities": self.capabilities(), "qp_map_sha256": checksum.hexdigest(),
            "qp_trace_path": str(requested_trace), "frame_trace_path": str(frame_trace),
            "applied_block_qp_available": False, "roi_fallback_count": fallback_count,
            "clipped_value_count": clipped_count,
        }
        (output_dir / "qp_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        raw_path.unlink(missing_ok=True)
        return {"bitstream_path": str(output), "num_frames": probe["num_frames"],
                "duration_sec": duration, "payload_bytes": size, "metadata_bytes": 0,
                "total_bytes": size, "actual_bitrate_kbps": actual_bitrate_kbps(size, duration),
                "encode_time_sec": elapsed, "ffmpeg_command": shell_join(bridge_command), **metadata}
