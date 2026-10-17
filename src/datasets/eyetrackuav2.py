"""EyeTrackUAV2 ingestion helpers.

This module converts the dataset at its boundary.  It deliberately has no
dependency on the codec or reconstruction implementations.
"""

import csv
import hashlib
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


RAW_COLUMNS = (
    "frame_id", "player_time_ms", "tracker_time_ms",
    "x_left", "y_left", "x_right", "y_right",
    "x_binocular", "y_binocular", "button_flag",
    "dominant_eye", "x_dominant", "y_dominant",
)
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
EYE_FILE_PATTERN = re.compile(
    r"^(?P<video>.+)_subj_(?P<observer>[^_]+)_(?P<condition>FV|TASK)_OrigRes_DomEye\.csv$",
    re.IGNORECASE,
)
VIDEO_DIRECTORY_SUFFIX = "_RAW_DATA_OrigRes_DomEye"


@dataclass(frozen=True)
class EyeTrackFile:
    path: Path
    video_id: str
    observer_id: str
    condition: str

    @property
    def sequence_id(self):
        return "{}__{}__S{}".format(self.video_id, self.condition, self.observer_id)


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def discover_eye_files(root, conditions=("TASK",)):
    root = Path(root)
    wanted = {str(value).upper() for value in conditions}
    records = []
    for path in sorted(root.rglob("*_OrigRes_DomEye.csv")):
        match = EYE_FILE_PATTERN.match(path.name)
        if not match:
            continue
        condition = match.group("condition").upper()
        if condition in wanted:
            records.append(EyeTrackFile(
                path=path,
                video_id=match.group("video"),
                observer_id=match.group("observer"),
                condition=condition,
            ))
    return records


def discover_video_ids(root):
    root = Path(root)
    names = {path.name for path in root.iterdir()
             if path.is_dir() and path.name.endswith(VIDEO_DIRECTORY_SUFFIX)}
    listing = root / ".listing"
    if listing.is_file():
        for line in listing.read_text(encoding="utf-8", errors="replace").splitlines():
            candidate = line.rstrip("\r").split()[-1] if line.split() else ""
            if candidate.endswith(VIDEO_DIRECTORY_SUFFIX):
                names.add(candidate)
    return sorted(name[:-len(VIDEO_DIRECTORY_SUFFIX)] for name in names)


def load_raw_gaze(path):
    data = np.genfromtxt(str(path), delimiter=",", dtype=np.float64)
    if data.ndim == 1:
        data = data[None, :]
    if data.ndim != 2 or data.shape[1] != len(RAW_COLUMNS):
        raise ValueError("Expected 13-column EyeTrackUAV2 CSV: {}".format(path))
    if data.shape[0] == 0:
        raise ValueError("Empty EyeTrackUAV2 CSV: {}".format(path))
    return data


def probe_video(path):
    command = [
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames,duration",
        "-of", "json", str(path),
    ]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError("ffprobe failed for {}: {}".format(path, result.stderr.strip()))
    stream = json.loads(result.stdout)["streams"][0]
    numerator, denominator = stream["avg_frame_rate"].split("/")
    fps = float(numerator) / float(denominator)
    frames = stream.get("nb_read_frames")
    if frames in (None, "N/A"):
        duration = float(stream["duration"])
        frames = int(round(duration * fps))
    return {
        "width": int(stream["width"]), "height": int(stream["height"]),
        "fps": fps, "num_frames": int(frames),
    }


def read_video_map(path, project_root):
    result = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"video_id", "video_path", "split"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Video map missing fields: {}".format(sorted(required - set(reader.fieldnames or []))))
        for row in reader:
            video_id = row["video_id"].strip()
            if not video_id or video_id in result:
                raise ValueError("Empty or duplicate video_id in video map: {}".format(video_id))
            if row["split"] not in ("val", "test"):
                raise ValueError("EyeTrackUAV2 video split must be val or test: {}".format(video_id))
            raw_path = row["video_path"].strip()
            video_path = Path(raw_path) if raw_path else None
            if video_path is not None and not video_path.is_absolute():
                video_path = Path(project_root) / video_path
            result[video_id] = {**row, "resolved_path": video_path.resolve() if video_path else None}
    group_splits = {}
    for video_id, row in result.items():
        group = row.get("source_group", "").strip() or video_id
        group_splits.setdefault(group, set()).add(row["split"])
    leaked = sorted(group for group, splits in group_splits.items() if len(splits) > 1)
    if leaked:
        raise ValueError("source_group spans val/test splits: {}".format(leaked))
    return result


def make_video_map_template(video_ids, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    video_ids = sorted(set(video_ids))
    with output.open("w", newline="", encoding="utf-8") as handle:
        fields = ["video_id", "video_path", "split", "source_group",
                  "source_start_frame", "source_end_frame_exclusive", "notes"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        groups = []
        for video_id in video_ids:
            group = re.sub(r"_crop\d+$", "", video_id) if video_id.startswith("VIRAT_") else video_id
            if group not in groups:
                groups.append(group)
        group_splits = {group: ("val" if index % 5 == 0 else "test")
                        for index, group in enumerate(groups)}
        for video_id in video_ids:
            group = re.sub(r"_crop\d+$", "", video_id) if video_id.startswith("VIRAT_") else video_id
            writer.writerow({
                "video_id": video_id, "video_path": "", "split": group_splits[group],
                "source_group": group, "source_start_frame": 0,
                "source_end_frame_exclusive": "", "notes": "missing_video",
            })
    return video_ids


def _natural_key(path):
    return [int(value) if value.isdigit() else value.lower()
            for value in re.split(r"(\d+)", path.name)]


def _frame_range(mapping, count):
    start = int(mapping.get("source_start_frame") or 0)
    raw_end = mapping.get("source_end_frame_exclusive")
    end = int(raw_end) if raw_end not in (None, "") else count
    if start < 0 or end <= start or end > count:
        raise ValueError("Invalid source frame range [{}, {}) for {} frames".format(start, end, count))
    return start, end


def prepare_stimulus(mapping, output_path, source_fps):
    """Return an exact video file, materializing image sequences or frame crops as FFV1."""
    source = mapping["resolved_path"]
    if source is None or not source.exists():
        raise ValueError("Stimulus source does not exist: {}".format(source or ""))
    output_path = Path(output_path)
    if source.is_file():
        info = probe_video(source)
        start, end = _frame_range(mapping, info["num_frames"])
        if start == 0 and end == info["num_frames"]:
            return source, info
        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(source),
            "-vf", "trim=start_frame={}:end_frame={},setpts=PTS-STARTPTS".format(start, end),
            "-an", "-c:v", "ffv1", str(output_path),
        ]
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise RuntimeError("Stimulus crop failed: {}".format(result.stderr.strip()))
        return output_path, probe_video(output_path)
    if not source.is_dir():
        raise ValueError("Stimulus source must be a video or image directory: {}".format(source))

    import cv2
    images = sorted((path for path in source.iterdir()
                     if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")), key=_natural_key)
    if not images:
        raise ValueError("No image frames in {}".format(source))
    start, end = _frame_range(mapping, len(images))
    if output_path.exists() and output_path.stat().st_size > 0:
        try:
            return output_path, probe_video(output_path)
        except Exception:
            output_path.unlink()
    if start == 0 and end == len(images) and len({path.suffix.lower() for path in images}) == 1:
        pattern = str(source / ("*" + images[0].suffix.lower()))
        command = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-framerate", str(source_fps), "-pattern_type", "glob", "-i", pattern,
            "-an", "-c:v", "ffv1", str(output_path),
        ]
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            return output_path, probe_video(output_path)
    images = images[start:end]
    first = cv2.imread(str(images[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise ValueError("Cannot decode stimulus frame: {}".format(images[0]))
    height, width = first.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
        "-pix_fmt", "bgr24", "-s", "{}x{}".format(width, height), "-r", str(source_fps),
        "-i", "pipe:0", "-an", "-c:v", "ffv1", str(output_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for image_path in images:
            frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if frame is None or frame.shape != first.shape:
                raise ValueError("Invalid or inconsistent stimulus frame: {}".format(image_path))
            process.stdin.write(np.ascontiguousarray(frame).tobytes())
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
        raise RuntimeError("Image-sequence materialization failed: {}".format(error.strip()))
    return output_path, probe_video(output_path)


def _valid_xy(x, y, width, height):
    return bool(np.isfinite(x) and np.isfinite(y) and 0 <= x < width and 0 <= y < height)


def _median_gaze(rows, x_column, y_column, width, height):
    points = [(row[x_column], row[y_column]) for row in rows
              if _valid_xy(row[x_column], row[y_column], width, height)]
    if not points:
        return None
    return tuple(np.median(np.asarray(points, dtype=np.float64), axis=0))


def convert_observer(data, source_width, source_height, source_fps, target_fps,
                     target_width, target_height, roi_radius_px, gaze_delay_ms,
                     max_hold_ms):
    """Return annotations in the preprocessed stretch coordinate system.

    ROI centres use all valid samples assigned to the source frame.  Selector
    gaze is causal: it uses only the last sample at or before the output-frame
    timestamp minus the configured feedback delay.
    """
    if not np.isfinite(data[:, 0]).all():
        raise ValueError("Non-finite EyeTrackUAV2 frame ID")
    source_frames = data[:, 0].astype(np.int64)
    finite_frames = source_frames
    if finite_frames.size == 0:
        raise ValueError("No finite frame IDs")
    first_frame = int(finite_frames.min())
    last_frame = int(finite_frames.max())
    frame_count = last_frame - first_frame + 1
    output_count = int(math.floor((frame_count - 1) * target_fps / source_fps)) + 1
    sx, sy = target_width / float(source_width), target_height / float(source_height)
    rx, ry = float(roi_radius_px) * sx, float(roi_radius_px) * sy

    valid_samples = data[
        np.asarray([_valid_xy(x, y, source_width, source_height)
                    for x, y in data[:, (11, 12)]], dtype=bool)
    ]
    valid_times = valid_samples[:, 1] if valid_samples.size else np.empty(0)
    rows = []
    last_eval = None
    for output_frame in range(output_count):
        elapsed_ms = output_frame * 1000.0 / target_fps
        source_offset = int(round(output_frame * source_fps / target_fps))
        source_frame = min(first_frame + source_offset, last_frame)
        current = data[source_frames == source_frame]
        evaluation = _median_gaze(current, 11, 12, source_width, source_height)
        roi_valid = evaluation is not None
        if evaluation is not None:
            last_eval = evaluation
        elif last_eval is not None:
            evaluation = last_eval
        else:
            evaluation = (source_width / 2.0, source_height / 2.0)

        deadline = elapsed_ms - float(gaze_delay_ms)
        eligible = np.nonzero(valid_times <= deadline)[0]
        control = None
        control_age = None
        if eligible.size:
            sample = valid_samples[int(eligible[-1])]
            control_age = elapsed_ms - float(sample[1])
            if control_age <= float(max_hold_ms):
                control = (float(sample[11]), float(sample[12]))
        gaze_valid = control is not None
        if control is None:
            control = (source_width / 2.0, source_height / 2.0)

        ex, ey = evaluation[0] * sx, evaluation[1] * sy
        gx, gy = control[0] * sx, control[1] * sy
        rows.append({
            "frame_id": output_frame,
            "source_frame_id": source_frame,
            "source_time_ms": elapsed_ms,
            "x0": max(0.0, ex - rx) / target_width,
            "y0": max(0.0, ey - ry) / target_height,
            "x1": min(float(target_width), ex + rx) / target_width,
            "y1": min(float(target_height), ey + ry) / target_height,
            "gaze_x": min(max(gx / target_width, 0.0), np.nextafter(1.0, 0.0)),
            "gaze_y": min(max(gy / target_height, 0.0), np.nextafter(1.0, 0.0)),
            "roi_valid": int(roi_valid), "gaze_valid": int(gaze_valid),
            "gaze_age_ms": "" if control_age is None else float(control_age),
        })
    return rows
