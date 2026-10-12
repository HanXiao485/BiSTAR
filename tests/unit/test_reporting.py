import csv

from src.experiments.runner import write_csv


def test_csv_accepts_method_specific_columns(tmp_path):
    path = tmp_path / "mixed.csv"
    write_csv([{"method": "codec", "bytes": 1}, {"method": "bistar", "metadata_bytes": 2}], path)
    rows = list(csv.DictReader(path.open()))
    assert rows[0]["bytes"] == "1" and rows[1]["metadata_bytes"] == "2"
