from pathlib import Path
from typing import List
import subprocess

import cv2
import numpy as np


def read_rgb_frames(path: Path) -> List[np.ndarray]:
    capture = cv2.VideoCapture(str(path)); frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok: break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames: raise ValueError("No frames decoded from {}".format(path))
    return frames


def preprocess_video(input_path: Path, output_path: Path, preprocessing: dict) -> dict:
    """Materialize the common source once so every method receives identical frames."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width = int(preprocessing["target_width"]); height = int(preprocessing["target_height"])
    fps = float(preprocessing["target_fps"])
    interpolation = preprocessing.get("interpolation", "bilinear")
    flags = {"nearest": "neighbor", "bilinear": "bilinear", "bicubic": "bicubic", "lanczos": "lanczos"}
    if interpolation not in flags:
        raise ValueError("Unsupported interpolation: {}".format(interpolation))
    vf = "fps={fps},scale={w}:{h}:flags={flag},format=rgb24".format(fps=fps, w=width, h=height, flag=flags[interpolation])
    command = ["ffmpeg", "-y", "-i", str(input_path), "-an", "-vf", vf, "-c:v", "ffv1", str(output_path)]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output_path.parent / "preprocess.log").write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError("Source preprocessing failed; see {}".format(output_path.parent / "preprocess.log"))
    frames = read_rgb_frames(output_path)
    return {"path": str(output_path), "num_frames": len(frames), "fps": fps, "width": width, "height": height,
            "duration_sec": len(frames) / fps, "ffmpeg_command": " ".join(command)}
