import struct

from .packet import BiSTARPacket


MAGIC = b"BSTR"
HEADER = struct.Struct("!4sBIIQHHHHBI")
CODECS = {"png": 1, "jpeg": 2}
REVERSE_CODECS = {value: key for key, value in CODECS.items()}


def serialize_packet(packet):
    if packet.payload_codec not in CODECS: raise ValueError("Unsupported payload codec")
    indices = struct.pack("!{}H".format(len(packet.selected_indices)), *packet.selected_indices) if packet.selected_indices else b""
    header = HEADER.pack(MAGIC, packet.version, packet.sequence_id, packet.frame_id, packet.timestamp_ns,
                         packet.image_width, packet.image_height, packet.patch_size, len(packet.selected_indices),
                         CODECS[packet.payload_codec], len(packet.payload))
    return header + indices + packet.payload


def deserialize_packet(data):
    if len(data) < HEADER.size: raise ValueError("Truncated packet")
    magic, version, sequence_id, frame_id, timestamp, width, height, patch, count, codec, payload_size = HEADER.unpack_from(data)
    if magic != MAGIC: raise ValueError("Invalid packet magic")
    indices_size = count * 2; expected = HEADER.size + indices_size + payload_size
    if len(data) != expected: raise ValueError("Packet length mismatch")
    indices = list(struct.unpack_from("!{}H".format(count), data, HEADER.size)) if count else []
    return BiSTARPacket(version, sequence_id, frame_id, timestamp, width, height, patch, indices,
                        REVERSE_CODECS[codec], data[HEADER.size + indices_size:])


def byte_accounting(packet, serialized):
    return {"payload_bytes": len(packet.payload), "metadata_bytes": len(serialized) - len(packet.payload), "total_bytes": len(serialized)}
