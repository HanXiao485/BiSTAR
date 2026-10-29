import struct
import subprocess
from pathlib import Path

import numpy as np


MAGIC = b"BTWN"
VERSION = 1
HEADER = struct.Struct("!4sBIIIQHHHHHBBII")
COORDINATE = struct.Struct("!HH")
CODECS = {"libx264": (1, "h264"), "libx265": (2, "hevc")}
CODEC_IDS = {value[0]: value[1] for value in CODECS.values()}


def _aligned_origin(center, size, limit, alignment):
    maximum = limit - size
    raw = min(maximum, max(0, int(round(center - size / 2.0))))
    if alignment > 1:
        raw = int(round(raw / alignment)) * alignment
        raw = min(maximum, max(0, raw))
    return raw


def track_window_positions(focus_points, frame_width, frame_height, window_width,
                           window_height, alignment=2, deadband_pixels=8):
    """Track focus points with a deadband while keeping each point in the window."""
    if not (0 < window_width <= frame_width and 0 < window_height <= frame_height):
        raise ValueError("Tracked window must fit inside the source frame")
    positions = []
    previous = None
    for point in focus_points:
        cx, cy = float(point[0]), float(point[1])
        target = (_aligned_origin(cx, window_width, frame_width, alignment),
                  _aligned_origin(cy, window_height, frame_height, alignment))
        if previous is None:
            current = target
        else:
            x, y = previous
            left = x + deadband_pixels
            right = x + window_width - deadband_pixels
            top = y + deadband_pixels
            bottom = y + window_height - deadband_pixels
            if cx < left or cx >= right:
                x = target[0]
            if cy < top or cy >= bottom:
                y = target[1]
            current = (x, y)
        positions.append(current)
        previous = current
    return positions


def _encode_video(frames, fps, path, encoder):
    codec = str(encoder.get("codec_name", "libx264"))
    if codec not in CODECS:
        raise ValueError("Tracked-window codec must be libx264 or libx265")
    _, container = CODECS[codec]
    height, width = frames[0].shape[:2]
    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s:v", "{}x{}".format(width, height),
        "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", codec,
        "-frames:v", str(len(frames)), "-vsync", "0",
        "-preset", str(encoder.get("preset", "fast")),
        "-tune", str(encoder.get("tune", "zerolatency")),
        "-g", str(encoder.get("gop_size", len(frames))),
        "-keyint_min", str(encoder.get("gop_size", len(frames))),
        "-sc_threshold", "0", "-bf", str(encoder.get("b_frames", 0)),
    ]
    mode = encoder.get("mode", "crf")
    if mode == "crf":
        command += ["-crf", str(encoder.get("crf", 22))]
    elif mode == "bitrate":
        rate = float(encoder["bitrate_kbps"])
        command += ["-b:v", "{}k".format(rate), "-maxrate", "{}k".format(rate),
                    "-bufsize", "{}k".format(max(1.0, rate * 2.0))]
    elif mode == "qp":
        command += ["-qp", str(encoder["qp"])]
    else:
        raise ValueError("Unsupported tracked-window rate-control mode: {}".format(mode))
    command += ["-pix_fmt", str(encoder.get("pixel_format", "yuv420p")),
                "-f", container, str(path)]
    result = subprocess.run(command, input=np.stack(frames).tobytes(), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
    return path.read_bytes(), command


def encode_tracked_window_gop(frames, focus_points, fps, config, output_dir,
                              sequence_id=0, gop_id=0, start_frame_id=0,
                              timestamp_ns=0):
    if not frames or len(frames) != len(focus_points):
        raise ValueError("Frames and focus points must be nonempty and aligned")
    codec = config["codec"] if "codec" in config else config
    height, width = frames[0].shape[:2]
    window = codec["window"]
    window_width, window_height = int(window["width"]), int(window["height"])
    positions = track_window_positions(
        focus_points, width, height, window_width, window_height,
        int(window.get("alignment", 2)), int(window.get("deadband_pixels", 8)),
    )
    crops = [frame[y:y + window_height, x:x + window_width].copy()
             for frame, (x, y) in zip(frames, positions)]
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    encoder = codec["encoder"]
    codec_id, extension = CODECS[str(encoder.get("codec_name", "libx264"))]
    path = output_dir / "seq{:04d}_gop{:06d}.{}".format(sequence_id, gop_id, extension)
    payload, command = _encode_video(crops, fps, path, encoder)
    coordinates = b"".join(COORDINATE.pack(x, y) for x, y in positions)
    header = HEADER.pack(
        MAGIC, VERSION, int(sequence_id), int(gop_id), int(start_frame_id), int(timestamp_ns),
        width, height, window_width, window_height, len(frames), codec_id, 0,
        len(coordinates), len(payload),
    )
    wire = header + coordinates + payload
    path.with_suffix(path.suffix + ".wire").write_bytes(wire)
    return {"wire": wire, "total_bytes": len(wire), "payload_bytes": len(payload),
            "metadata_bytes": len(header) + len(coordinates), "positions": positions,
            "ffmpeg_command": command, "bitstream_path": str(path)}


def decode_tracked_window_gop(wire):
    if len(wire) < HEADER.size:
        raise ValueError("Truncated tracked-window header")
    values = HEADER.unpack_from(wire)
    (magic, version, sequence_id, gop_id, start_frame_id, timestamp_ns, frame_width,
     frame_height, window_width, window_height, frame_count, codec_id, _flags,
     coordinate_bytes, payload_bytes) = values
    if magic != MAGIC or version != VERSION:
        raise ValueError("Unsupported tracked-window packet")
    if coordinate_bytes != frame_count * COORDINATE.size:
        raise ValueError("Tracked-window coordinate table size mismatch")
    if len(wire) != HEADER.size + coordinate_bytes + payload_bytes:
        raise ValueError("Tracked-window packet length mismatch")
    if codec_id not in CODEC_IDS:
        raise ValueError("Unknown tracked-window codec identifier")
    offset = HEADER.size
    positions = [COORDINATE.unpack_from(wire, offset + i * COORDINATE.size)
                 for i in range(frame_count)]
    payload = wire[offset + coordinate_bytes:]
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", CODEC_IDS[codec_id],
               "-i", "pipe:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    result = subprocess.run(command, input=payload, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
    frame_bytes = window_width * window_height * 3
    if len(result.stdout) != frame_count * frame_bytes:
        raise RuntimeError("Tracked-window decoded frame count mismatch")
    crops = np.frombuffer(result.stdout, dtype=np.uint8).reshape(
        frame_count, window_height, window_width, 3).copy()
    sparse_frames = np.zeros((frame_count, frame_height, frame_width, 3), dtype=np.uint8)
    visible_masks = np.zeros((frame_count, frame_height, frame_width), dtype=bool)
    for index, (x, y) in enumerate(positions):
        if x + window_width > frame_width or y + window_height > frame_height:
            raise ValueError("Tracked-window coordinates exceed the source frame")
        sparse_frames[index, y:y + window_height, x:x + window_width] = crops[index]
        visible_masks[index, y:y + window_height, x:x + window_width] = True
    return {"sequence_id": sequence_id, "gop_id": gop_id, "start_frame_id": start_frame_id,
            "timestamp_ns": timestamp_ns, "sparse_frames": list(sparse_frames),
            "visible_pixel_masks": visible_masks, "positions": positions,
            "payload_bytes": payload_bytes, "metadata_bytes": HEADER.size + coordinate_bytes,
            "total_bytes": len(wire)}
