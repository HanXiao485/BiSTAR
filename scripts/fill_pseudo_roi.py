#!/usr/bin/env python3
import argparse
import csv
import json
import math
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets.task_roi import ROI_FIELDS, read_task_annotations


def fill_pseudo_roi(path, mode="center", box_size=0.5, overwrite=False):
    """Populate an existing normalized-coordinate template with labelled proxy ROIs."""
    path = Path(path)
    if mode not in ("center", "moving") or not 0 < box_size <= 1:
        raise ValueError("mode must be center/moving and box_size must be in (0,1]")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    if not set(ROI_FIELDS).issubset(fields or []) or not rows:
        raise ValueError("Input must be a nonempty task ROI template CSV")
    annotated = [row for row in rows if any(row.get(key, "") for key in ("x0", "y0", "x1", "y1"))]
    if annotated and not overwrite:
        raise ValueError("Input contains annotations; use --overwrite explicitly to replace them")
    sequence_frames = {}
    for row in rows:
        frame_id = int(row["frame_id"])
        if frame_id < 0:
            raise ValueError("frame_id cannot be negative")
        sequence_frames.setdefault(row["sequence_id"], []).append(frame_id)
    for row in rows:
        ids = sequence_frames[row["sequence_id"]]
        phase = (int(row["frame_id"]) - min(ids)) / max(1, max(ids) - min(ids))
        amplitude = min(0.15, (1.0 - box_size) / 2.0)
        cx = 0.5 + (amplitude * math.sin(2 * math.pi * phase) if mode == "moving" else 0)
        cy = 0.5 + (amplitude * math.sin(4 * math.pi * phase) if mode == "moving" else 0)
        half = box_size / 2
        values = {"x0": cx - half, "y0": cy - half, "x1": cx + half, "y1": cy + half,
                  "gaze_x": cx, "gaze_y": cy}
        row.update({key: "{:.6f}".format(min(1.0, max(0.0, value))) for key, value in values.items()})
        row["provenance"] = "example_{}_proxy".format(mode)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup = path.with_name(path.stem + ".backup_" + timestamp + path.suffix)
    temporary = path.with_name(path.name + ".tmp")
    shutil.copy2(path, backup)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    read_task_annotations(temporary, "normalized")
    temporary.replace(path)
    metadata = {"output": str(path), "backup": str(backup), "mode": mode,
                "box_size": box_size, "coordinate_space": "normalized", "rows": len(rows),
                "sequences": {key: len(set(values)) for key, values in sequence_frames.items()},
                "provenance": "example_{}_proxy".format(mode), "proxy_ROI_only": True,
                "gaze_source": "synthetic_box_center_not_measured_eye_tracking"}
    path.with_suffix(".pseudo.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fill a ROI template with explicitly labelled pseudo annotations")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--mode", choices=("center", "moving"), default="center")
    parser.add_argument("--box-size", type=float, default=0.5, help="Normalized width and height")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing annotations after backup")
    args = parser.parse_args()
    print(json.dumps(fill_pseudo_roi(args.input, args.mode, args.box_size, args.overwrite), indent=2))
