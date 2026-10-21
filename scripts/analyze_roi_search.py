#!/usr/bin/env python3
"""Regenerate formal tables and plots from a completed ROI search run."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "outputs/.matplotlib_cache"))
from src.reports.formal_search import analyze_search_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.bootstrap_draws < 1:
        parser.error("--bootstrap-draws must be positive")
    print(json.dumps(analyze_search_output(args.output_dir, args.bootstrap_draws, args.seed), indent=2))
