#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experiments.rd_runner import run_rd
if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--config", type=Path, required=True); print(run_rd(p.parse_args().config))
