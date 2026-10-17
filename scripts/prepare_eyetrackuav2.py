#!/usr/bin/env python3
"""Convert EyeTrackUAV2 gaze files into BiSTAR manifest and ROI interfaces."""

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.eyetrackuav2 import (  # noqa: E402
    convert_observer, discover_eye_files, discover_video_ids, load_raw_gaze, make_video_map_template,
    prepare_stimulus, read_video_map, sha256_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EYE_ROOT = PROJECT_ROOT / "datasets/raw/EyeTrackUAV2/RAW_DATA/RAW_DATA_OrigRes_DomEye"
DEFAULT_VIDEO_MAP = PROJECT_ROOT / "datasets/metadata/eyetrackuav2/video_map.csv"
DEFAULT_MANIFEST = PROJECT_ROOT / "datasets/manifests/eyetrackuav2.csv"
DEFAULT_ANNOTATIONS = PROJECT_ROOT / "datasets/annotations/eyetrackuav2_task.csv"
DEFAULT_INVENTORY = PROJECT_ROOT / "datasets/metadata/eyetrackuav2/acquisition_inventory.csv"


def write_csv(rows, path, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eye-root", type=Path, default=DEFAULT_EYE_ROOT)
    parser.add_argument("--video-map", type=Path, default=DEFAULT_VIDEO_MAP)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--stimulus-output", type=Path,
                        default=PROJECT_ROOT / "datasets/processed/EyeTrackUAV2/stimuli")
    parser.add_argument("--conditions", nargs="+", default=["TASK"], choices=["TASK", "FV"])
    parser.add_argument("--source-fps", type=float, default=30.0)
    parser.add_argument("--target-fps", type=float, default=15.0)
    parser.add_argument("--target-width", type=int, default=224)
    parser.add_argument("--target-height", type=int, default=224)
    parser.add_argument("--roi-radius-px", type=float, default=64.0,
                        help="ROI half-size in original stimulus pixels")
    parser.add_argument("--gaze-delay-ms", type=float, default=0.0)
    parser.add_argument("--max-hold-ms", type=float, default=100.0)
    parser.add_argument("--observers-per-video", type=int, default=0,
                        help="0 keeps every observer; positive values create a deterministic pilot subset")
    parser.add_argument("--strict", action="store_true", help="Fail if any stimulus video is missing or mismatched")
    parser.add_argument("--template-only", action="store_true")
    args = parser.parse_args()

    if args.source_fps <= 0 or args.target_fps <= 0 or args.target_fps > args.source_fps:
        raise ValueError("Require 0 < target_fps <= source_fps")
    if args.target_width <= 0 or args.target_height <= 0 or args.roi_radius_px <= 0:
        raise ValueError("Frame dimensions and ROI radius must be positive")

    eye_files = discover_eye_files(args.eye_root, args.conditions)
    video_ids = discover_video_ids(args.eye_root) if args.eye_root.is_dir() else []
    if not eye_files and not (args.template_only and video_ids):
        raise ValueError("No EyeTrackUAV2 OrigRes_DomEye CSV files found below {}".format(args.eye_root))
    if not args.video_map.exists():
        template_ids = video_ids or sorted({record.video_id for record in eye_files})
        make_video_map_template(template_ids, args.video_map)
        print("Created video map template for {} videos: {}".format(len(template_ids), args.video_map))
        if args.template_only:
            return 0
        raise ValueError("Fill the generated video map with exact stimulus paths, then rerun")
    if args.template_only:
        print("Video map already exists; preserved without changes: {}".format(args.video_map))
        return 0

    video_map = read_video_map(args.video_map, PROJECT_ROOT)
    selected = []
    counts = Counter()
    for record in eye_files:
        if args.observers_per_video and counts[record.video_id] >= args.observers_per_video:
            continue
        selected.append(record)
        counts[record.video_id] += 1

    probes = {}
    inventory = []
    for video_id in sorted({record.video_id for record in selected}):
        mapping = video_map.get(video_id)
        path = mapping["resolved_path"] if mapping else None
        status = "ready"
        error = ""
        probe = None
        if mapping is None:
            status, error = "missing_mapping", "video_id absent from video_map.csv"
        elif path is None or not path.exists():
            status, error = "missing_video", "" if path is None else str(path)
        else:
            try:
                stimulus_path, probe = prepare_stimulus(
                    mapping, args.stimulus_output / (video_id + ".mkv"), args.source_fps)
                mapping["stimulus_path"] = stimulus_path
                probes[video_id] = probe
            except Exception as exception:
                status, error = "invalid_video", str(exception)
        inventory.append({
            "video_id": video_id,
            "video_path": "" if path is None else str(path),
            "stimulus_path": "" if mapping is None else str(mapping.get("stimulus_path", "")),
            "split": "" if mapping is None else mapping["split"],
            "source_group": "" if mapping is None else mapping.get("source_group", video_id),
            "status": status, "error": error,
            "width": "" if probe is None else probe["width"],
            "height": "" if probe is None else probe["height"],
            "fps": "" if probe is None else probe["fps"],
            "num_frames": "" if probe is None else probe["num_frames"],
        })
    write_csv(inventory, args.inventory, [
        "video_id", "video_path", "stimulus_path", "split", "source_group", "status", "error",
        "width", "height", "fps", "num_frames",
    ])
    missing = [row for row in inventory if row["status"] != "ready"]
    if args.strict and missing:
        raise ValueError("{} of {} stimulus videos are unavailable; see {}".format(
            len(missing), len(inventory), args.inventory))

    annotation_rows, manifest_rows, observer_rows = [], [], []
    for index, record in enumerate(selected, 1):
        mapping = video_map.get(record.video_id)
        if mapping is None or record.video_id not in probes:
            continue
        probe = probes[record.video_id]
        data = load_raw_gaze(record.path)
        converted = convert_observer(
            data, probe["width"], probe["height"], args.source_fps, args.target_fps,
            args.target_width, args.target_height, args.roi_radius_px,
            args.gaze_delay_ms, args.max_hold_ms,
        )
        expected_frames = len(converted)
        expected_duration = expected_frames / args.target_fps
        actual_duration = probe["num_frames"] / probe["fps"]
        duration_error = abs(actual_duration - expected_duration)
        aligned = duration_error <= max(2.0 / args.source_fps, 0.05)
        if args.strict and not aligned:
            raise ValueError("Stimulus duration mismatch for {}: video {:.3f}s, gaze {:.3f}s".format(
                record.video_id, actual_duration, expected_duration))
        manifest_rows.append({
            "sequence_id": record.sequence_id,
            "video_path": str(mapping["stimulus_path"]),
            "fps": probe["fps"], "width": probe["width"], "height": probe["height"],
            "num_frames": probe["num_frames"], "split": mapping["split"],
        })
        valid_roi = valid_gaze = 0
        for row in converted:
            valid_roi += row["roi_valid"]
            valid_gaze += row["gaze_valid"]
            annotation_rows.append({
                "sequence_id": record.sequence_id, "frame_id": row["frame_id"],
                "roi_id": "gaze_roi", "x0": row["x0"], "y0": row["y0"],
                "x1": row["x1"], "y1": row["y1"],
                "gaze_x": row["gaze_x"], "gaze_y": row["gaze_y"],
                "provenance": "eyetrackuav2_{}_dominant".format(record.condition.lower()),
                "roi_valid": row["roi_valid"], "gaze_valid": row["gaze_valid"],
                "source_frame_id": row["source_frame_id"],
                "source_time_ms": row["source_time_ms"], "gaze_age_ms": row["gaze_age_ms"],
                "video_id": record.video_id, "observer_id": record.observer_id,
                "condition": record.condition,
            })
        observer_rows.append({
            "sequence_id": record.sequence_id, "video_id": record.video_id,
            "observer_id": record.observer_id, "condition": record.condition,
            "split": mapping["split"], "source_group": mapping.get("source_group", record.video_id),
            "eye_file": str(record.path), "eye_file_sha256": sha256_file(record.path),
            "output_frames": expected_frames, "roi_valid_frames": valid_roi,
            "gaze_valid_frames": valid_gaze, "roi_valid_fraction": valid_roi / expected_frames,
            "gaze_valid_fraction": valid_gaze / expected_frames,
            "video_duration_sec": actual_duration, "gaze_duration_sec": expected_duration,
            "duration_error_sec": duration_error, "duration_aligned": int(aligned),
        })
        print("[{}/{}] {}: {} frames".format(index, len(selected), record.sequence_id, expected_frames), flush=True)

    if not manifest_rows:
        raise ValueError("No experiment-ready pairs. Fill valid paths in {}".format(args.video_map))
    write_csv(manifest_rows, args.manifest,
              ["sequence_id", "video_path", "fps", "width", "height", "num_frames", "split"])
    write_csv(annotation_rows, args.annotations, [
        "sequence_id", "frame_id", "roi_id", "x0", "y0", "x1", "y1", "gaze_x", "gaze_y",
        "provenance", "roi_valid", "gaze_valid", "source_frame_id", "source_time_ms", "gaze_age_ms",
        "video_id", "observer_id", "condition",
    ])
    observer_path = args.inventory.with_name("observer_inventory.csv")
    write_csv(observer_rows, observer_path, [
        "sequence_id", "video_id", "observer_id", "condition", "split", "source_group",
        "eye_file", "eye_file_sha256", "output_frames", "roi_valid_frames", "gaze_valid_frames",
        "roi_valid_fraction", "gaze_valid_fraction", "video_duration_sec", "gaze_duration_sec",
        "duration_error_sec", "duration_aligned",
    ])
    metadata = {
        "dataset": "EyeTrackUAV2", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "conditions": args.conditions, "source_fps": args.source_fps, "target_fps": args.target_fps,
        "target_width": args.target_width, "target_height": args.target_height,
        "roi_radius_px_at_source": args.roi_radius_px, "gaze_delay_ms": args.gaze_delay_ms,
        "max_hold_ms": args.max_hold_ms, "observer_count": len(observer_rows),
        "video_count": len({row["video_id"] for row in observer_rows}),
        "annotation_rows": len(annotation_rows), "missing_video_count": len(missing),
        "invalid_roi_policy": "hold_last_for_geometry_and_exclude_via_roi_valid",
        "invalid_selector_policy": "hold_up_to_max_hold_then_center_fallback",
    }
    metadata_path = args.inventory.with_name("preparation_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
