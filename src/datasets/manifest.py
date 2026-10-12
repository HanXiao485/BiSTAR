import csv
import subprocess
from pathlib import Path
from typing import Dict, List


FIELDS = ["sequence_id", "video_path", "fps", "width", "height", "num_frames", "split"]


def probe_video(path: Path) -> Dict[str, str]:
    import json
    command = ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
               "stream=width,height,avg_frame_rate,nb_read_frames", "-of", "json", str(path)]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0: raise RuntimeError(result.stderr)
    stream = json.loads(result.stdout)["streams"][0]
    n, d = stream["avg_frame_rate"].split("/")
    return {"fps": str(float(n) / float(d)), "width": stream["width"], "height": stream["height"], "num_frames": stream["nb_read_frames"]}


def build_manifest(input_dir: Path, output: Path, split: str = "test") -> List[Dict[str, str]]:
    extensions = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
    videos = sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in extensions)
    rows = []
    for path in videos:
        row = {"sequence_id": path.stem, "video_path": str(path), "split": split}
        row.update(probe_video(path)); rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
    return rows


def read_manifest(path: Path, split: str = "test") -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row["split"] == split]
