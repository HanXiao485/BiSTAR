import queue
import subprocess
import time
from pathlib import Path
from abc import ABC, abstractmethod

from src.common.types import DecodedFrame, FrameRecord


class RealtimeMethod(ABC):
    @abstractmethod
    def push_frame(self, frame: FrameRecord): raise NotImplementedError
    @abstractmethod
    def push_gaze(self, gaze): raise NotImplementedError
    @abstractmethod
    def poll_packet(self): raise NotImplementedError
    @abstractmethod
    def push_packet(self, packet): raise NotImplementedError
    @abstractmethod
    def poll_decoded_frame(self): raise NotImplementedError


class _QueueSession(RealtimeMethod):
    def __init__(self):
        self._packets = queue.Queue(); self._decoded = queue.Queue(); self._gaze = None

    def push_gaze(self, gaze): self._gaze = gaze
    def poll_packet(self):
        try: return self._packets.get_nowait()
        except queue.Empty: return None
    def poll_decoded_frame(self):
        try: return self._decoded.get_nowait()
        except queue.Empty: return None


class BiSTARRealtimeSession(_QueueSession):
    def __init__(self, selector, packetizer, reconstructor, sequence_number=0):
        super().__init__(); self.selector = selector; self.packetizer = packetizer; self.reconstructor = reconstructor
        self.sequence_number = sequence_number; self.previous_source = None; self.previous_decoded = None; self._masks = {}

    def push_frame(self, frame):
        start = time.time_ns(); selection = self.selector.select(frame.rgb, self.previous_source, self._gaze)
        packet, wire, accounting = self.packetizer.packetize(frame.rgb, selection["selected_indices"], self.sequence_number, frame.frame_id, frame.timestamp_ns)
        self._masks[frame.frame_id] = selection["visible_mask"]
        self._packets.put((wire, accounting, start, time.time_ns())); self.previous_source = frame.rgb

    def push_packet(self, packet_info):
        from src.bistar.transport.serializer import deserialize_packet
        wire, accounting, encode_start, encode_end = packet_info; receive = time.time_ns(); packet = deserialize_packet(wire)
        sparse = self.packetizer.sparse_frame(packet); decode_start = time.time_ns()
        rgb = self.reconstructor.reconstruct(sparse, self._masks.pop(packet.frame_id), self.previous_decoded); decode_end = time.time_ns()
        self.previous_decoded = rgb
        self._decoded.put(DecodedFrame(str(packet.sequence_id), packet.frame_id, rgb, packet.timestamp_ns, encode_start, encode_end,
                                       encode_end, receive, decode_start, decode_end, decode_end,
                                       accounting["payload_bytes"], accounting["metadata_bytes"], accounting["total_bytes"],
                                       {"evaluation_mode": "causal"}))


class CodecRealtimeSession(_QueueSession):
    """Low-latency single-frame codec session implementing the common API."""
    def __init__(self, codec="h264", fps=15):
        super().__init__()
        if codec not in ("h264", "h265"): raise ValueError("codec must be h264 or h265")
        self.codec = codec; self.fps = fps

    def push_frame(self, frame):
        height, width = frame.rgb.shape[:2]; start = time.time_ns()
        encoder = "libx264" if self.codec == "h264" else "libx265"; muxer = "h264" if self.codec == "h264" else "hevc"
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                   "-s", "{}x{}".format(width, height), "-r", str(self.fps), "-i", "pipe:0", "-frames:v", "1",
                   "-c:v", encoder, "-preset", "ultrafast", "-tune", "zerolatency", "-bf", "0", "-f", muxer, "pipe:1"]
        result = subprocess.run(command, input=frame.rgb.tobytes(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0: raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
        end = time.time_ns(); self._packets.put({"payload": result.stdout, "sequence_id": frame.sequence_id,
                                                  "frame_id": frame.frame_id, "capture_timestamp_ns": frame.timestamp_ns,
                                                  "encode_start_timestamp_ns": start, "encode_end_timestamp_ns": end,
                                                  "width": width, "height": height, "codec": self.codec})

    def push_packet(self, packet):
        receive = time.time_ns(); start = time.time_ns()
        demuxer = "h264" if packet["codec"] == "h264" else "hevc"
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", demuxer, "-i", "pipe:0", "-frames:v", "1",
                   "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
        result = subprocess.run(command, input=packet["payload"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0: raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
        import numpy as np
        rgb = np.frombuffer(result.stdout, np.uint8).reshape(packet["height"], packet["width"], 3).copy(); end = time.time_ns()
        self._decoded.put(DecodedFrame(packet["sequence_id"], packet["frame_id"], rgb, packet["capture_timestamp_ns"],
                                       packet["encode_start_timestamp_ns"], packet["encode_end_timestamp_ns"],
                                       packet["encode_end_timestamp_ns"], receive, start, end, end,
                                       len(packet["payload"]), 0, len(packet["payload"]),
                                       {"codec": packet["codec"], "evaluation_mode": "causal"}))


class AtlasRealtimeSession(_QueueSession):
    """GOP-buffered patch-atlas transport with explicit algorithmic waiting."""
    def __init__(self, selector, model_adapter, atlas_config, output_dir, sequence_number=0, fps=15):
        super().__init__()
        self.selector = selector
        self.model_adapter = model_adapter
        self.codec = atlas_config["codec"] if "codec" in atlas_config else atlas_config
        self.output_dir = Path(output_dir)
        self.sequence_number = sequence_number
        self.fps = float(fps)
        self.frames = []
        self.previous_source = None
        self.gop_id = 0
        self.budget_controller = None

    def push_frame(self, frame):
        self.frames.append(frame)
        if len(self.frames) >= int(self.codec["gop_size"]):
            self._flush()

    def flush(self):
        if self.frames:
            self._flush()

    def _flush(self):
        from src.bistar.transport.atlas import encode_atlas_gop
        started = time.time_ns()
        records = self.frames
        self.frames = []
        expected = int(self.codec["gop_size"])
        rgb = [record.rgb for record in records]
        padded = rgb + [rgb[-1]] * (expected - len(rgb))
        first = rgb[0]
        if self.budget_controller is None:
            from src.bistar.transport.patch_budget import PatchBudgetController
            patch_size = int(self.codec["patch_size"])
            patch_count = (first.shape[0] // patch_size) * (first.shape[1] // patch_size)
            target_kbps = float(self.codec["encoder"].get("bitrate_kbps", 100))
            self.budget_controller = PatchBudgetController(
                self.codec["selector"], target_kbps, patch_count, target_kbps=target_kbps
            )
        gaze = self._gaze or (first.shape[1] / 2.0, first.shape[0] / 2.0, 1.0)
        selection = self.selector.select(
            first, self.previous_source, gaze, self.budget_controller.current
        )
        encoded = encode_atlas_gop(
            padded, selection["selected_indices"], self.fps, int(self.codec["patch_size"]),
            self.output_dir, self.codec["encoder"], sequence_id=self.sequence_number,
            gop_id=self.gop_id, start_frame_id=records[0].frame_id,
            timestamp_ns=records[0].timestamp_ns,
        )
        ended = time.time_ns()
        target_kbps = float(self.codec["encoder"].get("bitrate_kbps", 100))
        self.budget_controller.update(encoded["total_bytes"], len(padded), self.fps, target_kbps)
        self._packets.put({
            "wire": encoded["wire"], "valid_frames": len(records),
            "visible_patch_count": len(selection["selected_indices"]),
            "patch_budget_strategy": self.budget_controller.strategy,
            "capture_timestamps_ns": [record.timestamp_ns for record in records],
            "encode_start_timestamp_ns": started, "encode_end_timestamp_ns": ended,
        })
        self.previous_source = rgb[-1]
        self.gop_id += 1

    def push_packet(self, packet):
        from src.bistar.transport.atlas import decode_atlas_gop
        received = time.time_ns()
        decoded = decode_atlas_gop(packet["wire"])
        decode_started = time.time_ns()
        reconstructed = self.model_adapter.reconstruct_clip(
            decoded["sparse_frames"], decoded["visible_mask"]
        )
        decode_ended = time.time_ns()
        valid_count = int(packet["valid_frames"])
        payload_base, payload_remainder = divmod(decoded["payload_bytes"], valid_count)
        metadata_base, metadata_remainder = divmod(decoded["metadata_bytes"], valid_count)
        for offset in range(valid_count):
            payload = payload_base + (1 if offset < payload_remainder else 0)
            metadata = metadata_base + (1 if offset < metadata_remainder else 0)
            self._decoded.put(DecodedFrame(
                str(decoded["sequence_id"]), decoded["start_frame_id"] + offset,
                reconstructed[offset], packet["capture_timestamps_ns"][offset],
                packet["encode_start_timestamp_ns"], packet["encode_end_timestamp_ns"],
                packet["encode_end_timestamp_ns"], received, decode_started, decode_ended,
                decode_ended, payload, metadata, payload + metadata,
                {"codec": "h264_patch_atlas", "gop_id": decoded["gop_id"],
                 "evaluation_mode": "chunked_causal_selection",
                 "visible_patch_count": packet["visible_patch_count"],
                 "patch_budget_strategy": packet["patch_budget_strategy"]},
            ))
