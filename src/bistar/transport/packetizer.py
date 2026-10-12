import cv2
import numpy as np

from .packet import BiSTARPacket
from .serializer import byte_accounting, serialize_packet


class Packetizer:
    def __init__(self, patch_size=16, mode="png", quality=90):
        if mode not in ("png", "jpeg"): raise ValueError("mode must be png or jpeg")
        self.patch_size, self.mode, self.quality = patch_size, mode, quality

    def packetize(self, frame, selected_indices, sequence_id, frame_id, timestamp_ns):
        h, w = frame.shape[:2]; cols = w // self.patch_size; patches = []
        for index in selected_indices:
            row, col = divmod(index, cols); patches.append(frame[row*self.patch_size:(row+1)*self.patch_size, col*self.patch_size:(col+1)*self.patch_size])
        strip = np.concatenate(patches, axis=1) if patches else np.zeros((self.patch_size, 1, 3), dtype=np.uint8)
        ext = ".png" if self.mode == "png" else ".jpg"; params = [] if self.mode == "png" else [cv2.IMWRITE_JPEG_QUALITY, self.quality]
        ok, encoded = cv2.imencode(ext, cv2.cvtColor(strip, cv2.COLOR_RGB2BGR), params)
        if not ok: raise RuntimeError("Patch encoding failed")
        packet = BiSTARPacket(1, sequence_id, frame_id, timestamp_ns, w, h, self.patch_size, list(selected_indices), self.mode, encoded.tobytes())
        data = serialize_packet(packet)
        return packet, data, byte_accounting(packet, data)

    @staticmethod
    def sparse_frame(packet):
        data = cv2.imdecode(np.frombuffer(packet.payload, np.uint8), cv2.IMREAD_COLOR)
        if data is None: raise ValueError("Invalid image payload")
        strip = cv2.cvtColor(data, cv2.COLOR_BGR2RGB); frame = np.zeros((packet.image_height, packet.image_width, 3), dtype=np.uint8)
        cols = packet.image_width // packet.patch_size
        for offset, index in enumerate(packet.selected_indices):
            row, col = divmod(index, cols); patch = strip[:, offset*packet.patch_size:(offset+1)*packet.patch_size]
            frame[row*packet.patch_size:(row+1)*packet.patch_size, col*packet.patch_size:(col+1)*packet.patch_size] = patch
        return frame
