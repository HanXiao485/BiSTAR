from pathlib import Path
import importlib
import sys

import numpy as np

from .mask_adapter import MaskAdapter
from src.common.hashing import sha256_file


class VideoMAEAdapter:
    def __init__(self, config):
        self.config = config["model"] if "model" in config else config; self.model = None
        arch = self.config["architecture"]; temporal = self.config["temporal"]
        self.mask_adapter = MaskAdapter(temporal["num_frames"], arch["tubelet_size"])

    def load_model(self):
        import timm
        source = Path(self.config["source"])
        if not source.is_dir():
            raise FileNotFoundError("VideoMAE source missing: {}".format(source))
        source_text = str(source.resolve())
        if source_text not in sys.path:
            sys.path.insert(0, source_text)
        # Import registers the official pretraining architectures with timm.
        importlib.import_module("modeling_pretrain")
        checkpoint = Path(self.config["checkpoint"]["path"])
        if not checkpoint.is_file(): raise FileNotFoundError("VideoMAE checkpoint missing: {}".format(checkpoint))
        expected_hash = self.config["checkpoint"].get("sha256")
        if expected_hash and sha256_file(checkpoint) != expected_hash:
            raise ValueError("VideoMAE checkpoint SHA-256 mismatch")
        decoder_depth = int(self.config.get("runtime", {}).get("decoder_depth", 4))
        self.model = timm.create_model(self.config["architecture"]["model_name"], pretrained=False, decoder_depth=decoder_depth)
        import torch
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=True); self.model.load_state_dict(state.get("model", state), strict=True); self.model.eval()
        device = self.config.get("runtime", {}).get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was configured but is not available")
        self.model.to(device)
        return self.model

    def preprocess_clip(self, frames):
        array = np.stack(frames).astype(np.float32) / 255.0
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        array = (array - mean) / std
        return np.transpose(array, (3, 0, 1, 2))[None]

    def build_mask(self, visible_patch_mask): return self.mask_adapter.build(visible_patch_mask)

    def reconstruct_clip(self, frames, visible_patch_mask):
        """Reconstruct RGB without using hidden ground-truth pixels."""
        if self.model is None: self.load_model()
        import torch
        frames_array = np.stack(frames).astype(np.uint8)
        visible_2d = np.asarray(visible_patch_mask, dtype=bool)
        patch_size = int(self.config["architecture"]["patch_size"])
        tubelet = int(self.config["architecture"]["tubelet_size"])
        pixel_visible = np.repeat(np.repeat(visible_2d, patch_size, 0), patch_size, 1)
        sparse = frames_array.copy(); sparse[:, ~pixel_visible] = 0
        clip = torch.from_numpy(self.preprocess_clip(sparse))
        mask_array = self.build_mask(visible_2d); mask = torch.from_numpy(mask_array)[None]
        device = next(self.model.parameters()).device; clip = clip.to(device); mask = mask.to(device)
        with torch.no_grad(): predictions = self.model(clip, mask).detach().float().cpu().numpy()[0]
        output = sparse.copy(); hidden_indices = np.flatnonzero(mask_array)
        visible_values = frames_array[:, pixel_visible].astype(np.float32) / 255.0
        mean = visible_values.mean(axis=(0, 1)) if visible_values.size else np.array([0.5, 0.5, 0.5])
        std = visible_values.std(axis=(0, 1)) if visible_values.size else np.array([0.25, 0.25, 0.25])
        std = np.maximum(std, 1e-3); h_tokens, w_tokens = visible_2d.shape
        for prediction, token_index in zip(predictions, hidden_indices):
            temporal, spatial = divmod(int(token_index), h_tokens * w_tokens); row, col = divmod(spatial, w_tokens)
            patch = prediction.reshape(tubelet, patch_size, patch_size, 3)
            patch = np.clip(patch * std + mean, 0.0, 1.0)
            t0, y0, x0 = temporal * tubelet, row * patch_size, col * patch_size
            output[t0:t0+tubelet, y0:y0+patch_size, x0:x0+patch_size] = np.rint(patch * 255).astype(np.uint8)
        return output

    def reconstruct_dynamic_clip(self, frames, visible_pixel_masks):
        """Reconstruct a clip from receiver-visible, frame-varying pixels only."""
        if self.model is None: self.load_model()
        import torch
        frames_array = np.stack(frames).astype(np.uint8)
        pixel_visible = np.asarray(visible_pixel_masks, dtype=bool)
        if pixel_visible.shape != frames_array.shape[:3]:
            raise ValueError("Dynamic pixel mask must match [T,H,W] frames")
        patch_size = int(self.config["architecture"]["patch_size"])
        height, width = pixel_visible.shape[1:]
        if height % patch_size or width % patch_size:
            raise ValueError("Frame dimensions must be divisible by VideoMAE patch size")
        patch_visible = pixel_visible.reshape(
            len(frames_array), height // patch_size, patch_size,
            width // patch_size, patch_size,
        ).all(axis=(2, 4))
        sparse = frames_array.copy(); sparse[~pixel_visible] = 0
        clip = torch.from_numpy(self.preprocess_clip(sparse))
        mask_array = self.mask_adapter.build_dynamic(patch_visible)
        mask = torch.from_numpy(mask_array)[None]
        device = next(self.model.parameters()).device
        with torch.no_grad():
            predictions = self.model(clip.to(device), mask.to(device)).detach().float().cpu().numpy()[0]
        output = sparse.copy()
        tubelet = int(self.config["architecture"]["tubelet_size"])
        hidden_indices = np.flatnonzero(mask_array)
        values = frames_array[pixel_visible].reshape(-1, 3).astype(np.float32) / 255.0
        mean = values.mean(axis=0) if values.size else np.array([0.5, 0.5, 0.5])
        std = np.maximum(values.std(axis=0), 1e-3) if values.size else np.array([0.25] * 3)
        h_tokens, w_tokens = height // patch_size, width // patch_size
        for prediction, token_index in zip(predictions, hidden_indices):
            temporal, spatial = divmod(int(token_index), h_tokens * w_tokens)
            row, col = divmod(spatial, w_tokens)
            patch = prediction.reshape(tubelet, patch_size, patch_size, 3)
            patch = np.clip(patch * std + mean, 0.0, 1.0)
            t0, y0, x0 = temporal * tubelet, row * patch_size, col * patch_size
            output[t0:t0 + tubelet, y0:y0 + patch_size, x0:x0 + patch_size] = np.rint(
                patch * 255).astype(np.uint8)
        output[pixel_visible] = frames_array[pixel_visible]
        return output

    def reconstruct_frame(self, history_frames, current_sparse_frame, current_visible_mask):
        frames = list(history_frames) + [current_sparse_frame]
        expected = int(self.config["temporal"]["num_frames"])
        if len(frames) != expected:
            raise ValueError("VideoMAE requires exactly {} decoder-available frames".format(expected))
        return self.reconstruct_clip(frames, current_visible_mask)[-1]
