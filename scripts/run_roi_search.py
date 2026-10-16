#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "outputs/.matplotlib_cache"))
from src.experiments.roi_search_runner import run_roi_search


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search configurations on annotated task ROIs and evaluate held-out winners")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", help="Recorded per-run model device override; does not edit model YAML")
    parser.add_argument("--plan", action="store_true", help="Validate and count candidates without running codecs")
    parser.add_argument("--threads", type=int, default=4, help="PyTorch CPU threads")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    print(json.dumps(run_roi_search(args.config, args.device, args.plan), indent=2))
