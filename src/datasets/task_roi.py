import csv
from pathlib import Path

import numpy as np


ROI_FIELDS = ["sequence_id", "frame_id", "roi_id", "x0", "y0", "x1", "y1",
              "gaze_x", "gaze_y", "provenance"]


def read_task_annotations(path, coordinate_space="normalized"):
    if coordinate_space not in ("normalized", "preprocessed_pixels"):
        raise ValueError("ROI coordinate_space must be normalized or preprocessed_pixels")
    annotations = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"sequence_id", "frame_id", "x0", "y0", "x1", "y1", "provenance"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("ROI CSV missing fields: {}".format(sorted(required - set(reader.fieldnames or []))))
        for row in reader:
            key = (row["sequence_id"], int(row["frame_id"]))
            if not row["provenance"]:
                raise ValueError("Every ROI row requires provenance: {}".format(key))
            try:
                box = [float(row[name]) for name in ("x0", "y0", "x1", "y1")]
            except ValueError as error:
                raise ValueError("Unannotated ROI at {}".format(key)) from error
            if not np.isfinite(box).all() or not (0 <= box[0] < box[2] and 0 <= box[1] < box[3]):
                raise ValueError("Invalid ROI box at {}".format(key))
            if coordinate_space == "normalized" and max(box) > 1:
                raise ValueError("Normalized ROI coordinates must be in [0, 1]")
            gaze = None
            if row.get("gaze_x") or row.get("gaze_y"):
                gaze = (float(row["gaze_x"]), float(row["gaze_y"]))
                if not np.isfinite(gaze).all() or min(gaze) < 0:
                    raise ValueError("Invalid gaze point at {}".format(key))
                if coordinate_space == "normalized" and max(gaze) > 1:
                    raise ValueError("Normalized gaze coordinates must be in [0, 1]")
            annotations.setdefault(key, []).append({"box": box, "gaze": gaze,
                                                   "provenance": row["provenance"]})
    if not annotations:
        raise ValueError("ROI CSV is empty")
    return annotations


def frame_annotations(annotations, sequence_id, frame_ids, height, width, coordinate_space):
    masks, gaze_points, provenance = [], [], set()
    for frame_id in frame_ids:
        key = (sequence_id, int(frame_id))
        if key not in annotations:
            raise ValueError("Missing task ROI for {} frame {}".format(*key))
        mask = np.zeros((height, width), dtype=bool)
        gaze = None
        for annotation in annotations[key]:
            box = np.asarray(annotation["box"], dtype=float)
            if coordinate_space == "normalized":
                box *= np.asarray([width, height, width, height])
            if box[2] > width or box[3] > height:
                raise ValueError("ROI exceeds preprocessed frame at {}".format(key))
            x0, y0 = np.floor(box[:2]).astype(int)
            x1, y1 = np.ceil(box[2:]).astype(int)
            mask[y0:y1, x0:x1] = True
            if annotation["gaze"] is not None:
                point = np.asarray(annotation["gaze"], dtype=float)
                if coordinate_space == "normalized":
                    point *= np.asarray([width, height])
                if point[0] > width or point[1] > height:
                    raise ValueError("Gaze exceeds preprocessed frame at {}".format(key))
                candidate = (float(point[0]), float(point[1]), 1.0)
                if gaze is not None and gaze != candidate:
                    raise ValueError("Conflicting gaze annotations at {}".format(key))
                gaze = candidate
            provenance.add(annotation["provenance"])
        if not mask.any():
            raise ValueError("Empty task ROI at {}".format(key))
        masks.append(mask)
        gaze_points.append(gaze)
    return masks, gaze_points, sorted(provenance)
