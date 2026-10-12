import csv
import json
from pathlib import Path

import numpy as np

from src.common.config import dump_yaml
from src.common.logging import write_run_metadata


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def prepare_run(config, output_dir):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    dump_yaml(config, output_dir / "config_resolved.yaml"); write_run_metadata(PROJECT_ROOT, output_dir)
    return output_dir


def write_csv(rows, path):
    if not rows: path.write_text("", encoding="utf-8"); return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames: fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames); writer.writeheader(); writer.writerows(rows)


def json_safe(value):
    if isinstance(value, dict): return {key: json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)): return [json_safe(val) for val in value]
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    return value


def write_summary(summary, path): path.write_text(json.dumps(json_safe(summary), indent=2, allow_nan=False), encoding="utf-8")
