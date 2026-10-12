import csv

from src.datasets.manifest import read_manifest


def test_manifest_split_is_source_of_truth(tmp_path):
    path = tmp_path / "manifest.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sequence_id","video_path","fps","width","height","num_frames","split"])
        writer.writeheader(); writer.writerow({"sequence_id":"a","video_path":"a.mp4","fps":15,"width":224,"height":224,"num_frames":1,"split":"test"})
        writer.writerow({"sequence_id":"b","video_path":"b.mp4","fps":15,"width":224,"height":224,"num_frames":1,"split":"train"})
    assert [row["sequence_id"] for row in read_manifest(path, "test")] == ["a"]
