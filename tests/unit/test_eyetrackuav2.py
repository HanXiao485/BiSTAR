import csv

import cv2
import numpy as np

from src.datasets.eyetrackuav2 import (
    convert_observer, discover_eye_files, discover_video_ids, load_raw_gaze,
    make_video_map_template, prepare_stimulus, read_video_map,
)
from src.datasets.task_roi import frame_annotations_with_validity, read_task_annotations
from src.metrics.task_fidelity import evaluate_task_sequence


def test_discovery_parsing_and_causal_conversion(tmp_path):
    folder = tmp_path / "DTB70_Basketball_RAW_DATA_OrigRes_DomEye_TASK"
    folder.mkdir()
    path = folder / "DTB70_Basketball_subj_10_TASK_OrigRes_DomEye.csv"
    rows = [
        [1, 1, 1001, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0, 2, np.nan, np.nan],
        [1, 20, 1020, 10, 20, 10, 20, 10, 20, 0, 2, 10, 20],
        [2, 34, 1034, 30, 40, 30, 40, 30, 40, 0, 2, 30, 40],
        [3, 67, 1067, 50, 60, 50, 60, 50, 60, 0, 2, 50, 60],
    ]
    np.savetxt(path, np.asarray(rows), delimiter=",", fmt="%.4f")
    found = discover_eye_files(tmp_path, ["TASK"])
    assert len(found) == 1
    assert found[0].sequence_id == "DTB70_Basketball__TASK__S10"
    data = load_raw_gaze(path)
    converted = convert_observer(data, 100, 80, 30, 15, 200, 160, 10, 0, 100)
    assert len(converted) == 2
    assert converted[0]["roi_valid"] == 1
    assert converted[0]["gaze_valid"] == 0
    assert converted[0]["gaze_x"] == 0.5
    assert converted[1]["gaze_valid"] == 1
    assert converted[1]["source_frame_id"] == 3


def test_video_id_discovery_does_not_require_completed_csv_download(tmp_path):
    (tmp_path / "UAV123_car1_RAW_DATA_OrigRes_DomEye").mkdir()
    (tmp_path / "unrelated").mkdir()
    assert discover_video_ids(tmp_path) == ["UAV123_car1"]


def test_image_sequence_is_materialized_with_requested_crop(tmp_path):
    source = tmp_path / "frames"
    source.mkdir()
    for index in range(5):
        image = np.full((16, 24, 3), index * 20, dtype=np.uint8)
        assert cv2.imwrite(str(source / "img{:03d}.jpg".format(index)), image)
    mapping = {"resolved_path": source, "source_start_frame": "1",
               "source_end_frame_exclusive": "4"}
    output, info = prepare_stimulus(mapping, tmp_path / "stimulus.mkv", 30)
    assert output.is_file()
    assert info["num_frames"] == 3
    assert info["width"] == 24
    assert info["height"] == 16


def test_video_map_groups_virat_crops_and_rejects_split_leakage(tmp_path):
    path = tmp_path / "video_map.csv"
    make_video_map_template([
        "VIRAT_scene_crop1", "VIRAT_scene_crop2", "DTB70_Basketball"], path)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    virat = [row for row in rows if row["video_id"].startswith("VIRAT")]
    assert {row["source_group"] for row in virat} == {"VIRAT_scene"}
    assert len({row["split"] for row in virat}) == 1
    rows[1]["split"] = "test" if rows[2]["split"] == "val" else "val"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    try:
        read_video_map(path, tmp_path)
    except ValueError as error:
        assert "source_group spans" in str(error)
    else:
        raise AssertionError("Expected source-group leakage rejection")


def test_validity_fields_exclude_invalid_roi_metrics(tmp_path):
    path = tmp_path / "roi.csv"
    fields = ["sequence_id", "frame_id", "roi_id", "x0", "y0", "x1", "y1",
              "gaze_x", "gaze_y", "provenance", "roi_valid", "gaze_valid"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(dict(sequence_id="s", frame_id=0, roi_id="g", x0=.25, y0=.25,
                             x1=.75, y1=.75, gaze_x=.5, gaze_y=.5, provenance="real",
                             roi_valid=1, gaze_valid=1))
        writer.writerow(dict(sequence_id="s", frame_id=1, roi_id="g", x0=.25, y0=.25,
                             x1=.75, y1=.75, gaze_x=.5, gaze_y=.5, provenance="real",
                             roi_valid=0, gaze_valid=0))
    annotations = read_task_annotations(path)
    masks, gaze, _, roi_validity, gaze_validity = frame_annotations_with_validity(
        annotations, "s", [0, 1], 16, 16, "normalized")
    assert roi_validity == [True, False]
    assert gaze_validity == [True, False]
    reference = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(2)]
    reconstructed = [reference[0].copy(), np.full((16, 16, 3), 255, dtype=np.uint8)]
    summary, frames, _ = evaluate_task_sequence(
        reference, reconstructed, masks, {"sequence_id": "s"}, valid_frames=roi_validity)
    assert summary["roi_psnr"] > 100
    assert summary["roi_valid_frames"] == 1
    assert summary["roi_valid_fraction"] == 0.5
    assert frames[1]["roi_psnr"] is None
