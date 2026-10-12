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


def write_rgb_video(frames: List[np.ndarray], output_path: Path, fps: float, crf: int = 18) -> dict:
    """Encode RGB uint8 frames as an H.264 MP4 intended for visual inspection."""
    if len(frames) == 0:
        raise ValueError("Cannot write an empty video")
    if fps <= 0:
        raise ValueError("fps must be positive")
    first = np.asarray(frames[0])
    if first.ndim != 3 or first.shape[2] != 3:
        raise ValueError("Frames must have HxWx3 RGB shape")
    height, width = first.shape[:2]
    if width % 2 or height % 2:
        raise ValueError("H.264 yuv420p output requires even frame dimensions")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "{}x{}".format(width, height),
        "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
        "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        for frame in frames:
            array = np.asarray(frame)
            if array.shape != first.shape:
                raise ValueError("All frames must have the same shape")
            process.stdin.write(np.ascontiguousarray(array, dtype=np.uint8).tobytes())
        process.stdin.close()
        error = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait()
    except Exception:
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
        process.kill()
        process.wait()
        raise
    if return_code != 0:
        raise RuntimeError("FFmpeg video encoding failed: {}".format(error))
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError("FFmpeg produced no output video: {}".format(output_path))
    return {
        "path": str(output_path),
        "num_frames": len(frames),
        "fps": float(fps),
        "width": width,
        "height": height,
        "bytes": output_path.stat().st_size,
    }


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
