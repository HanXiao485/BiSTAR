#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.config import load_yaml
from src.experiments.matched_bitrate_runner import run_matched
from src.experiments.rd_runner import run_rd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)
    if "rd" in config:
        run_rd(args.config)
    elif "matched_bitrate" in config:
        run_matched(args.config)
    else:
        raise ValueError("Config must contain rd or matched_bitrate")


if __name__ == "__main__":
    main()
