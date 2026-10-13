import numpy as np

from src.bistar.reconstruction.previous_frame import PreviousFrameReconstructor
from src.bistar.selector.random_selector import RandomSelector
from src.bistar.selector.full_selector import FullSelector
from src.bistar.transport.packetizer import Packetizer
from src.common.types import FrameRecord
from src.realtime.session import AtlasRealtimeSession, BiSTARRealtimeSession, CodecRealtimeSession


def test_causal_realtime_round_trip():
    session = BiSTARRealtimeSession(RandomSelector(40, seed=0), Packetizer(), PreviousFrameReconstructor())
    frame = FrameRecord("0", 0, 123, np.full((224, 224, 3), 100, np.uint8))
    session.push_frame(frame); packet = session.poll_packet(); assert packet is not None
    session.push_packet(packet); decoded = session.poll_decoded_frame()
    assert decoded.frame_id == 0 and decoded.total_bytes > decoded.payload_bytes > 0
    assert decoded.display_ready_timestamp_ns >= decoded.capture_timestamp_ns


def test_codec_realtime_round_trip():
    session = CodecRealtimeSession("h264")
    frame = FrameRecord("scene", 4, 1, np.full((32, 32, 3), 120, np.uint8))
    session.push_frame(frame); packet = session.poll_packet(); assert packet["payload"]
    session.push_packet(packet); decoded = session.poll_decoded_frame()
    assert decoded.rgb.shape == frame.rgb.shape and decoded.frame_id == 4 and decoded.total_bytes > 0


class _DummyModelAdapter:
    @staticmethod
    def reconstruct_clip(frames, visible_mask):
        return np.stack(frames)


def test_atlas_realtime_buffers_one_gop(tmp_path):
    config = {"codec": {
        "patch_size": 16, "gop_size": 2,
        "selector": {"type": "bistar_full", "num_visible_patches": 2},
        "encoder": {"codec_name": "libx264", "preset": "ultrafast", "tune": "zerolatency",
                    "mode": "bitrate", "bitrate_kbps": 100, "b_frames": 0},
    }}
    session = AtlasRealtimeSession(
        FullSelector(16), _DummyModelAdapter(), config, tmp_path, fps=15
    )
    session.push_frame(FrameRecord("0", 0, 100, np.full((32, 32, 3), 50, np.uint8)))
    assert session.poll_packet() is None
    session.push_frame(FrameRecord("0", 1, 200, np.full((32, 32, 3), 60, np.uint8)))
    packet = session.poll_packet()
    assert packet is not None
    session.push_packet(packet)
    decoded = [session.poll_decoded_frame(), session.poll_decoded_frame()]
    assert [item.frame_id for item in decoded] == [0, 1]
    assert all(item.total_bytes > 0 for item in decoded)
    assert all(item.method_metadata["codec"] == "h264_patch_atlas" for item in decoded)
