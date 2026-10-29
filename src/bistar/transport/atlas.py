import math
import struct
import subprocess
import shlex
from pathlib import Path

import numpy as np


MAGIC = b"BATL"
HEADER = struct.Struct("!4sBIIIQHHHBBHII")
X264_USER_DATA_UUID = bytes.fromhex("dc45e9bde6d948b7962cd820d923eeef")


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
    mode = encoder.get("mode", "bitrate")
    if mode == "bitrate":
        bitrate = float(encoder["bitrate_kbps"])
        command += ["-b:v", "{}k".format(bitrate), "-maxrate", "{}k".format(bitrate),
                    "-bufsize", "{}k".format(max(1.0, bitrate * 2.0))]
    elif mode == "crf":
        command += ["-crf", str(encoder.get("crf", 23))]
    elif mode == "qp":
        command += ["-qp", str(encoder["qp"])]
    else:
        raise ValueError("Unsupported Atlas rate-control mode: {}".format(mode))
    if encoder.get("pixel_format"):
        command += ["-pix_fmt", str(encoder["pixel_format"])]
    command += ["-f", "h264", str(output_path)]
    payload = b"".join(np.ascontiguousarray(frame, dtype=np.uint8).tobytes() for frame in frames)
    result = subprocess.run(command, input=payload, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output_path.with_suffix(".encode.log").write_text(
        "$ " + shlex.join(command) + "\n" + result.stderr.decode("utf-8", errors="replace"),
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError("Atlas encode failed: {}".format(result.stderr.decode("utf-8", errors="replace")))
    return output_path.read_bytes()


def _annexb_nals(payload):
    """Return (start-code, NAL) pairs and reject non-Annex-B prefixes."""
    starts = []
    index = 0
    while index + 3 <= len(payload):
        if payload[index:index + 4] == b"\x00\x00\x00\x01":
            starts.append((index, 4)); index += 4
        elif payload[index:index + 3] == b"\x00\x00\x01":
            starts.append((index, 3)); index += 3
        else:
            index += 1
    if not starts or starts[0][0] != 0:
        raise ValueError("H.264 payload is not a canonical Annex-B stream")
    result = []
    for position, (offset, size) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else len(payload)
        nal = payload[offset + size:end]
        if not nal:
            raise ValueError("Empty Annex-B NAL unit")
        result.append((payload[offset:offset + size], nal))
    return result


def _ebsp_to_rbsp(data):
    result = bytearray()
    zeros = 0
    index = 0
    while index < len(data):
        value = data[index]
        if zeros >= 2 and value == 3:
            if index + 1 >= len(data) or data[index + 1] > 3:
                raise ValueError("Malformed emulation-prevention sequence")
            zeros = 0
            index += 1
            continue
        result.append(value)
        zeros = zeros + 1 if value == 0 else 0
        index += 1
    return bytes(result)


def _rbsp_to_ebsp(data):
    result = bytearray()
    zeros = 0
    for value in data:
        if zeros >= 2 and value <= 3:
            result.append(3)
            zeros = 0
        result.append(value)
        zeros = zeros + 1 if value == 0 else 0
    return bytes(result)


def _read_sei_number(rbsp, offset, label):
    total = 0
    while True:
        if offset >= len(rbsp):
            raise ValueError("Truncated SEI {}".format(label))
        value = rbsp[offset]
        offset += 1
        total += value
        if value != 0xFF:
            return total, offset


def _encode_sei_number(value):
    if value < 0:
        raise ValueError("SEI number must be nonnegative")
    return b"\xff" * (value // 255) + bytes([value % 255])


def parse_sei_messages(nal):
    """Parse an H.264 prefix SEI NAL into messages and validate trailing bits."""
    if not nal or nal[0] & 0x1F != 6:
        raise ValueError("NAL unit is not H.264 SEI")
    rbsp = _ebsp_to_rbsp(nal[1:])
    messages = []
    offset = 0
    while offset < len(rbsp):
        if rbsp[offset:] == b"\x80" or (
                rbsp[offset] == 0x80 and not any(rbsp[offset + 1:])):
            return messages
        payload_type, offset = _read_sei_number(rbsp, offset, "payload type")
        payload_size, offset = _read_sei_number(rbsp, offset, "payload size")
        end = offset + payload_size
        if end > len(rbsp):
            raise ValueError("Truncated SEI payload")
        messages.append({"payload_type": payload_type, "payload": rbsp[offset:end]})
        offset = end
    raise ValueError("SEI RBSP lacks trailing bits")


def _is_x264_identification(message):
    data = message["payload"]
    return (message["payload_type"] == 5 and len(data) >= 16 and
            data[:16] == X264_USER_DATA_UUID and data[16:].startswith(b"x264 - core"))


def filter_encoder_identification_sei(payload):
    """Remove only recognized x264 identification messages from Annex-B H.264."""
    output = bytearray()
    records = []
    for nal_index, (start_code, nal) in enumerate(_annexb_nals(payload)):
        nal_type = nal[0] & 0x1F
        if nal_type != 6:
            output.extend(start_code + nal)
            records.append({"nal_index": nal_index, "nal_type": nal_type,
                            "payload_type": None, "removed": False,
                            "original_bytes": len(start_code) + len(nal),
                            "filtered_bytes": len(start_code) + len(nal)})
            continue
        messages = parse_sei_messages(nal)
        kept = []
        for message_index, message in enumerate(messages):
            removed = _is_x264_identification(message)
            records.append({"nal_index": nal_index, "message_index": message_index,
                            "nal_type": nal_type, "payload_type": message["payload_type"],
                            "uuid": message["payload"][:16].hex()
                            if message["payload_type"] == 5 and len(message["payload"]) >= 16 else None,
                            "removed": removed, "payload_bytes": len(message["payload"])})
            if not removed:
                kept.append(message)
        if not kept:
            continue
        rbsp = bytearray()
        for message in kept:
            rbsp.extend(_encode_sei_number(message["payload_type"]))
            rbsp.extend(_encode_sei_number(len(message["payload"])))
            rbsp.extend(message["payload"])
        rbsp.append(0x80)
        output.extend(start_code + bytes([nal[0]]) + _rbsp_to_ebsp(bytes(rbsp)))
    filtered = bytes(output)
    return filtered, {"original_bytes": len(payload), "filtered_bytes": len(filtered),
                      "removed_bytes": len(payload) - len(filtered), "records": records}


def analyze_annexb_h264(payload):
    """Return a byte-exact NAL decomposition without changing the stream."""
    categories = {1: "image", 5: "image", 6: "sei", 7: "parameter_set", 8: "parameter_set"}
    rows = []
    for index, (start_code, nal) in enumerate(_annexb_nals(payload)):
        nal_type = nal[0] & 0x1F
        messages = parse_sei_messages(nal) if nal_type == 6 else []
        rows.append({
            "nal_index": index,
            "nal_type": nal_type,
            "category": categories.get(nal_type, "other"),
            "bytes": len(start_code) + len(nal),
            "start_code_bytes": len(start_code),
            "sei_payload_types": ",".join(str(item["payload_type"]) for item in messages),
            "sei_uuids": ",".join(
                item["payload"][:16].hex() for item in messages
                if item["payload_type"] == 5 and len(item["payload"]) >= 16
            ),
        })
    if sum(row["bytes"] for row in rows) != len(payload):
        raise RuntimeError("NAL analysis does not account for the complete payload")
    return rows


def replace_atlas_payload(wire, payload):
    """Replace BATL v1 codec payload while preserving and validating metadata."""
    if len(wire) < HEADER.size:
        raise ValueError("Truncated atlas packet")
    values = list(HEADER.unpack_from(wire))
    mask_size, old_payload_size = values[-2:]
    if len(wire) != HEADER.size + mask_size + old_payload_size:
        raise ValueError("Atlas packet length mismatch")
    values[-1] = len(payload)
    return HEADER.pack(*values) + wire[HEADER.size:HEADER.size + mask_size] + payload


def filter_atlas_wire_encoder_sei(wire):
    values = HEADER.unpack_from(wire)
    mask_size, payload_size = values[-2:]
    if len(wire) != HEADER.size + mask_size + payload_size:
        raise ValueError("Atlas packet length mismatch")
    payload = wire[HEADER.size + mask_size:]
    filtered, report = filter_encoder_identification_sei(payload)
    return replace_atlas_payload(wire, filtered), report


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
        "ffmpeg_command": bitstream_path.with_suffix(".encode.log").read_text(
            encoding="utf-8").splitlines()[0].removeprefix("$ "),
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
