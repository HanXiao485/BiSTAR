import numpy as np

from src.bistar.transport.packetizer import Packetizer
from src.bistar.transport.serializer import deserialize_packet


def test_packet_round_trip_and_real_size():
    frame = np.arange(224*224*3, dtype=np.uint8).reshape(224,224,3); packetizer = Packetizer(16, "png")
    packet, wire, sizes = packetizer.packetize(frame, [0, 15, 195], 2, 3, 4); decoded = deserialize_packet(wire)
    assert decoded.selected_indices == [0, 15, 195]
    assert sizes["payload_bytes"] + sizes["metadata_bytes"] == len(wire) == sizes["total_bytes"]
    sparse = packetizer.sparse_frame(decoded)
    assert np.array_equal(sparse[:16, :16], frame[:16, :16])
    assert not sparse[32:48, 32:48].any()
