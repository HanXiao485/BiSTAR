#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets.manifest import build_manifest


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--input", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--split", default="test")
    args = parser.parse_args(); rows = build_manifest(args.input, args.output, args.split); print("Wrote {} sequences to {}".format(len(rows), args.output))


if __name__ == "__main__": main()
