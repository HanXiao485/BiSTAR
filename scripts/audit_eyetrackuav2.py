#!/usr/bin/env python3
"""Audit a wget mirror of EyeTrackUAV2 against saved FTP directory listings."""

import argparse
import csv
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


LISTING_LINE = re.compile(r"^-\S+\s+\d+\s+\S+\s+\S+\s+(\d+)\s+\S+\s+\d+\s+\S+\s+(.+)$")


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def expected_files(root, condition):
    expected = []
    suffix = "_{}".format(condition.upper())
    for listing in sorted(root.rglob(".listing")):
        if listing.parent.name.endswith(("_FV", "_TASK")) and not listing.parent.name.endswith(suffix):
            continue
        for raw_line in listing.read_text(encoding="utf-8", errors="replace").splitlines():
            match = LISTING_LINE.match(raw_line.rstrip("\r"))
            if not match:
                continue
            name = match.group(2).rstrip("\r")
            if name in (".", "..", ".listing"):
                continue
            expected.append((listing.parent / name, int(match.group(1))))
    unique = {}
    for path, size in expected:
        unique[path] = size
    return sorted(unique.items())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("datasets/raw/EyeTrackUAV2/RAW_DATA/RAW_DATA_OrigRes_DomEye"))
    parser.add_argument("--condition", choices=["TASK", "FV", "ALL"], default="TASK")
    parser.add_argument("--checksums", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=Path("datasets/metadata/eyetrackuav2/raw_data_audit.json"))
    args = parser.parse_args()
    condition = "" if args.condition == "ALL" else args.condition
    items = expected_files(args.root, condition)
    rows, missing, mismatched = [], 0, 0
    for path, expected_size in items:
        exists = path.is_file()
        actual_size = path.stat().st_size if exists else None
        status = "complete" if actual_size == expected_size else ("missing" if not exists else "size_mismatch")
        missing += status == "missing"
        mismatched += status == "size_mismatch"
        rows.append({
            "path": str(path), "expected_size": expected_size, "actual_size": actual_size,
            "status": status,
            "sha256": sha256_file(path) if args.checksums and status == "complete" else "",
        })
    complete = len(items) > 0 and not missing and not mismatched
    summary = {
        "dataset": "EyeTrackUAV2", "condition": args.condition,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "expected_files": len(items), "complete_files": sum(row["status"] == "complete" for row in rows),
        "missing_files": missing, "size_mismatches": mismatched,
        "expected_bytes": sum(row["expected_size"] for row in rows),
        "complete_bytes": sum(row["actual_size"] or 0 for row in rows if row["status"] == "complete"),
        "checksums_computed": bool(args.checksums), "complete": complete,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "path", "expected_size", "actual_size", "status", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
