import math
import struct
import subprocess
from pathlib import Path

import numpy as np


MAGIC = b"BATL"
HEADER = struct.Struct("!4sBIIIQHHHBBHII")


def encode_patch_mask(selected_indices, patch_count):
    mask = bytearray((int(patch_count) + 7) // 8)
    for index in selected_indices:
        index = int(index)
        if index < 0 or index >= patch_count:
            raise ValueError("Patch index out of range: {}".format(index))
        mask[index // 8] |= 1 << (index % 8)
    return bytes(mask)


def decode_patch_mask(data, patch_count):
    expected = (int(patch_count) + 7) // 8
    if len(data) != expected:
        raise ValueError("Patch mask has {} bytes, expected {}".format(len(data), expected))
    return [index for index in range(patch_count) if data[index // 8] & (1 << (index % 8))]


def atlas_shape(visible_count, patch_size):
    if visible_count < 1:
        raise ValueError("Atlas requires at least one visible patch")
    columns = int(math.ceil(math.sqrt(visible_count)))
    rows = int(math.ceil(visible_count / columns))
    return rows, columns, rows * patch_size, columns * patch_size


def build_atlas_frames(frames, selected_indices, patch_size):
    rows, columns, height, width = atlas_shape(len(selected_indices), patch_size)
    image_columns = frames[0].shape[1] // patch_size
    atlases = []
    for frame in frames:
        atlas = np.zeros((height, width, 3), dtype=np.uint8)
        for slot, index in enumerate(selected_indices):
            source_row, source_column = divmod(int(index), image_columns)
            atlas_row, atlas_column = divmod(slot, columns)
            source_y, source_x = source_row * patch_size, source_column * patch_size
            atlas_y, atlas_x = atlas_row * patch_size, atlas_column * patch_size
            atlas[atlas_y:atlas_y + patch_size, atlas_x:atlas_x + patch_size] = (
                frame[source_y:source_y + patch_size, source_x:source_x + patch_size]
            )
        atlases.append(atlas)
    return atlases, {"rows": rows, "columns": columns, "height": height, "width": width}


def restore_sparse_frames(atlas_frames, selected_indices, image_height, image_width, patch_size):
    _, atlas_columns, _, _ = atlas_shape(len(selected_indices), patch_size)
    image_columns = image_width // patch_size
    sparse_frames = []
    for atlas in atlas_frames:
        sparse = np.zeros((image_height, image_width, 3), dtype=np.uint8)
        for slot, index in enumerate(selected_indices):
            source_row, source_column = divmod(int(index), image_columns)
            atlas_row, atlas_column = divmod(slot, atlas_columns)
            atlas_y, atlas_x = atlas_row * patch_size, atlas_column * patch_size
            source_y, source_x = source_row * patch_size, source_column * patch_size
            sparse[source_y:source_y + patch_size, source_x:source_x + patch_size] = (
                atlas[atlas_y:atlas_y + patch_size, atlas_x:atlas_x + patch_size]
            )
        sparse_frames.append(sparse)
    return sparse_frames


def _encode_h264(frames, fps, output_path, encoder):
    height, width = frames[0].shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "{}x{}".format(width, height),
        "-r", str(fps), "-i", "pipe:0", "-frames:v", str(len(frames)), "-vsync", "0",
        "-an", "-c:v", encoder.get("codec_name", "libx264"),
        "-preset", str(encoder.get("preset", "ultrafast")), "-tune", str(encoder.get("tune", "zerolatency")),
        "-bf", "0", "-g", str(len(frames)), "-keyint_min", str(len(frames)), "-sc_threshold", "0",
    ]
    if encoder.get("mode", "bitrate") == "bitrate":
        bitrate = float(encoder["bitrate_kbps"])
        command += ["-b:v", "{}k".format(bitrate), "-maxrate", "{}k".format(bitrate),
                    "-bufsize", "{}k".format(max(1.0, bitrate * 2.0))]
    else:
        command += ["-crf", str(encoder.get("crf", 23))]
    command += ["-f", "h264", str(output_path)]
    payload = b"".join(np.ascontiguousarray(frame, dtype=np.uint8).tobytes() for frame in frames)
    result = subprocess.run(command, input=payload, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError("Atlas encode failed: {}".format(result.stderr.decode("utf-8", errors="replace")))
    return output_path.read_bytes()


def _decode_h264(payload, frame_count, height, width):
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "h264", "-i", "pipe:0",
        "-frames:v", str(frame_count), "-vsync", "0",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    result = subprocess.run(command, input=payload, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError("Atlas decode failed: {}".format(result.stderr.decode("utf-8", errors="replace")))
    frame_bytes = height * width * 3
    if len(result.stdout) != frame_count * frame_bytes:
        raise RuntimeError("Decoded atlas frame count differs from encoded frame count")
    return [
        np.frombuffer(result.stdout[offset:offset + frame_bytes], np.uint8).reshape(height, width, 3).copy()
        for offset in range(0, len(result.stdout), frame_bytes)
    ]


def encode_atlas_gop(frames, selected_indices, fps, patch_size, output_dir, encoder,
                     sequence_id=0, gop_id=0, start_frame_id=0, timestamp_ns=0):
    if not frames:
        raise ValueError("Cannot encode an empty GOP")
    image_height, image_width = frames[0].shape[:2]
    patch_count = (image_height // patch_size) * (image_width // patch_size)
    mask = encode_patch_mask(selected_indices, patch_count)
    atlases, layout = build_atlas_frames(frames, selected_indices, patch_size)
    bitstream_path = Path(output_dir) / "gop_{:06d}.h264".format(gop_id)
    payload = _encode_h264(atlases, fps, bitstream_path, encoder)
    header = HEADER.pack(
        MAGIC, 1, int(sequence_id), int(gop_id), int(start_frame_id), int(timestamp_ns),
        int(image_width), int(image_height), int(patch_size), int(layout["columns"]),
        int(layout["rows"]), int(len(frames)), int(len(mask)), int(len(payload)),
    )
    wire = header + mask + payload
    decoded_atlases = _decode_h264(payload, len(frames), layout["height"], layout["width"])
    sparse_frames = restore_sparse_frames(
        decoded_atlases, selected_indices, image_height, image_width, patch_size
    )
    return {
        "wire": wire,
        "payload": payload,
        "payload_bytes": len(payload),
        "metadata_bytes": len(header) + len(mask),
        "total_bytes": len(wire),
        "sparse_frames": sparse_frames,
        "visible_mask": np.asarray(
            [index in set(selected_indices) for index in range(patch_count)], dtype=bool
        ).reshape(image_height // patch_size, image_width // patch_size),
        "selected_indices": list(selected_indices),
        "layout": layout,
        "bitstream_path": str(bitstream_path),
    }


def decode_atlas_gop(wire):
    if len(wire) < HEADER.size:
        raise ValueError("Truncated atlas packet")
    values = HEADER.unpack_from(wire)
    (magic, version, sequence_id, gop_id, start_frame_id, timestamp_ns,
     image_width, image_height, patch_size, atlas_columns, atlas_rows,
     frame_count, mask_size, payload_size) = values
    if magic != MAGIC or version != 1:
        raise ValueError("Invalid atlas packet")
    expected = HEADER.size + mask_size + payload_size
    if len(wire) != expected:
        raise ValueError("Atlas packet length mismatch")
    mask = wire[HEADER.size:HEADER.size + mask_size]
    payload = wire[HEADER.size + mask_size:]
    patch_count = (image_height // patch_size) * (image_width // patch_size)
    indices = decode_patch_mask(mask, patch_count)
    atlas_height, atlas_width = atlas_rows * patch_size, atlas_columns * patch_size
    atlas_frames = _decode_h264(payload, frame_count, atlas_height, atlas_width)
    sparse_frames = restore_sparse_frames(
        atlas_frames, indices, image_height, image_width, patch_size
    )
    visible_mask = np.zeros(patch_count, dtype=bool)
    visible_mask[indices] = True
    return {
        "sequence_id": sequence_id,
        "gop_id": gop_id,
        "start_frame_id": start_frame_id,
        "timestamp_ns": timestamp_ns,
        "frame_count": frame_count,
        "selected_indices": indices,
        "visible_mask": visible_mask.reshape(image_height // patch_size, image_width // patch_size),
        "sparse_frames": sparse_frames,
        "payload_bytes": payload_size,
        "metadata_bytes": HEADER.size + mask_size,
        "total_bytes": len(wire),
    }
