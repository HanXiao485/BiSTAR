from src.bistar.transport.bitrate_accounting import bitrate_breakdown, bitrate_kbps


def test_known_bitrate(): assert abs(bitrate_kbps(10000, 1.0) - 80.0) < 1e-6


def test_payload_metadata_breakdown():
    values = bitrate_breakdown([1000] * 15, [100] * 15, 15)
    assert values["payload_bitrate_kbps"] == 120.0
    assert values["metadata_bitrate_kbps"] == 12.0
    assert values["total_bitrate_kbps"] == 132.0
