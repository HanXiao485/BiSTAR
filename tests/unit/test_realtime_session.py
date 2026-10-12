import numpy as np

from src.bistar.reconstruction.previous_frame import PreviousFrameReconstructor
from src.bistar.selector.random_selector import RandomSelector
from src.bistar.transport.packetizer import Packetizer
from src.common.types import FrameRecord
from src.realtime.session import BiSTARRealtimeSession, CodecRealtimeSession


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
