#!/usr/bin/env python3
import argparse
import copy
import csv
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import dump_yaml, load_yaml
from src.experiments.ablation_runner import run_ablation
from src.experiments.bistar_runner import encode_sparse_clip, evaluate_videomae_clip
from src.experiments.rd_runner import run_rd
from src.datasets.frame_reader import read_rgb_frames, write_rgb_video
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def generate_video(path, frames):
    path.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc2=size=224x224:rate=15", "-frames:v", str(frames), "-c:v", "ffv1", str(path)]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    manifest = ROOT / "datasets/manifests/bistar_eval.csv"; manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sequence_id", "video_path", "fps", "width", "height", "num_frames", "split"]); writer.writeheader()
        writer.writerow({"sequence_id": "synthetic", "video_path": str(path.relative_to(ROOT)), "fps": 15, "width": 224, "height": 224, "num_frames": frames, "split": "test"})


def main(config_path):
    config = load_yaml(config_path)
    if config["smoke"].get("generate_synthetic", True): generate_video(ROOT / "datasets/raw/synthetic.mkv", config["smoke"]["num_frames"])
    rd_summary = run_rd(config_path); result = {"rd": rd_summary}
    if config["smoke"].get("run_bistar", True):
        ablation = load_yaml(ROOT / "configs/experiments/ablation.yaml"); ablation["experiment"]["output_dir"] = "outputs/smoke_test/bistar"
        ablation["experiment"]["save_videomae_videos"] = bool(config["experiment"].get("save_videomae_videos", False))
        ablation["ablation"]["visible_patch_counts"] = [40]; ablation["ablation"]["variants"] = ["bistar_full"]
        generated = ROOT / "outputs/smoke_test/smoke_ablation_resolved.yaml"; dump_yaml(ablation, generated)
        result["bistar"] = {"points": len(run_ablation(generated))}
        frames = read_rgb_frames(ROOT / "datasets/raw/synthetic.mkv")[:16]
        encoded = encode_sparse_clip(frames, budget=40, fps=15)
        evaluated = evaluate_videomae_clip(frames, encoded)
        video = None
        if config["experiment"].get("save_videomae_videos", False):
            video = write_rgb_video(evaluated["reconstructed"], ROOT / "outputs/smoke_test/bistar_videomae/reconstructed.mp4", 15)
        vm_summary = {"frames": len(frames), "total_bytes": evaluated["total_bytes"],
                      "payload_bytes": evaluated["payload_bytes"], "metadata_bytes": evaluated["metadata_bytes"],
                      "ssim": float(np.mean([x["ssim"] for x in evaluated["frame_metrics"]])),
                      "lpips": float(np.mean([x["lpips"] for x in evaluated["frame_metrics"]])),
                      "inference_ms_per_frame": evaluated["inference_ms_per_frame"],
                      "gpu_peak_memory_bytes": evaluated["gpu_peak_memory_bytes"],
                      "reconstruction_video": video["path"] if video else None,
                      "reconstruction_video_bytes": video["bytes"] if video else None}
        vm_dir = ROOT / "outputs/smoke_test/bistar_videomae"; vm_dir.mkdir(parents=True, exist_ok=True)
        (vm_dir / "summary.json").write_text(json.dumps(vm_summary, indent=2), encoding="utf-8")
        result["bistar_videomae"] = vm_summary
    print(result)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True); main(parser.parse_args().config)
