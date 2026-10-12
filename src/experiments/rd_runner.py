import copy
from pathlib import Path

import numpy as np

from src.codecs import H264Adapter, H265Adapter
from src.common.config import load_codec_config, load_yaml, resolve_path, validate_experiment_config
from src.datasets.frame_reader import preprocess_video, read_rgb_frames
from src.datasets.manifest import read_manifest
from src.metrics.image_quality import create_lpips_model, frame_metrics
from src.experiments.bistar_runner import evaluate_videomae_clip, select_budget_for_rate
from src.bistar.reconstruction.videomae_adapter import VideoMAEAdapter
from .runner import PROJECT_ROOT, prepare_run, write_csv, write_summary


ADAPTERS = {"h264": H264Adapter, "h265": H265Adapter}


def _aggregate(rows, keys):
    return {key: float(np.mean([row[key] for row in rows if np.isfinite(row[key])])) if any(np.isfinite(row[key]) for row in rows) else None for key in keys}


def run_rd(config_path):
    config = validate_experiment_config(load_yaml(config_path), "rd"); output = prepare_run(config, resolve_path(config["experiment"]["output_dir"], PROJECT_ROOT))
    dataset_cfg = load_yaml(resolve_path(config["dataset"]["config"], PROJECT_ROOT))["dataset"]
    rows = read_manifest(resolve_path(dataset_cfg["manifest"], PROJECT_ROOT), config["dataset"].get("split", "test"))
    if not rows: raise ValueError("Dataset manifest contains no selected sequences")
    frame_rows, sequence_rows = [], []
    lpips_model = create_lpips_model("cpu") if "lpips" in config.get("quality_metrics", ["lpips"]) else None
    rates = config["rd"]["target_bitrates_kbps"]
    for row in rows:
        raw_source = resolve_path(row["video_path"], PROJECT_ROOT)
        common = preprocess_video(raw_source, output / "sources" / (row["sequence_id"] + ".mkv"), dataset_cfg["preprocessing"])
        source = Path(common["path"]); originals = read_rgb_frames(source)
        for method, enabled in config["methods"].items():
            if not enabled.get("enabled", False) or method not in ADAPTERS: continue
            base = load_codec_config(resolve_path(enabled["config"], PROJECT_ROOT)); prep = dataset_cfg["preprocessing"]
            for target in rates:
                codec_cfg = copy.deepcopy(base); codec_cfg["preprocessing"] = prep
                run_dir = output / "artifacts" / row["sequence_id"] / method / str(target)
                adapter = ADAPTERS[method](); requested = float(target); tolerance = config["rd"]["bitrate_match_tolerance_percent"]
                for iteration in range(1, int(config["rd"].get("max_match_iterations", 8)) + 1):
                    codec_cfg["codec"]["encoder"]["bitrate_kbps"] = round(requested, 3)
                    encoded = adapter.encode_sequence(source, run_dir / "match_{}".format(iteration), codec_cfg)
                    if abs(encoded["actual_bitrate_kbps"] - target) / target * 100 <= tolerance: break
                    requested *= max(0.5, min(2.0, target / encoded["actual_bitrate_kbps"]))
                decoded = adapter.decode_sequence(Path(encoded["bitstream_path"]), run_dir / "decoded", codec_cfg)
                recon = read_rgb_frames(Path(decoded["decoded_path"]))
                if len(recon) != encoded["num_frames"]: raise RuntimeError("Decoded frame alignment failure")
                if len(originals) != len(recon): raise RuntimeError("Source/decoded frame count differs")
                current = []
                for frame_id, (reference, reconstructed) in enumerate(zip(originals, recon)):
                    metrics = frame_metrics(reference, reconstructed, lpips_model); item = {"sequence_id": row["sequence_id"], "method": method, "target_kbps": target, "frame_id": frame_id, **metrics}
                    frame_rows.append(item); current.append(item)
                aggregate = _aggregate(current, ["psnr", "ssim", "ms_ssim", "lpips"])
                sequence_rows.append({"sequence_id": row["sequence_id"], "method": method, "target_kbps": target,
                                      "actual_bitrate_kbps": encoded["actual_bitrate_kbps"], "total_bytes": encoded["total_bytes"],
                                      "num_frames": len(recon), "encode_time_sec": encoded["encode_time_sec"], "decode_time_sec": decoded["decode_time_sec"], **aggregate})
                sequence_rows[-1].update({"requested_kbps": requested, "match_iterations": iteration,
                                          "status": "matched" if abs(encoded["actual_bitrate_kbps"]-target)/target*100 <= tolerance else "bitrate_unreachable"})
        bistar_cfg = config["methods"].get("bistar", {})
        if bistar_cfg.get("enabled", False):
            model_adapter = VideoMAEAdapter(load_yaml(PROJECT_ROOT / "configs/models/videomae.yaml")); model_adapter.load_model()
            fps = float(dataset_cfg["preprocessing"]["target_fps"]); clip_size = 16
            for target in rates:
                current, total_bytes, payload_bytes, metadata_bytes, inference = [], 0, 0, 0, 0.0
                for start in range(0, len(originals), clip_size):
                    valid = originals[start:start+clip_size]; valid_count = len(valid)
                    clip = valid + [valid[-1]] * (clip_size - valid_count)
                    matched = select_budget_for_rate(clip, target, fps)
                    evaluated = evaluate_videomae_clip(clip, matched["encoded"], adapter=model_adapter, lpips_model=lpips_model)
                    inference += evaluated["inference_time_sec"] * valid_count / clip_size
                    total_bytes += int(evaluated["total_bytes"] * valid_count / clip_size)
                    payload_bytes += int(evaluated["payload_bytes"] * valid_count / clip_size)
                    metadata_bytes += int(evaluated["metadata_bytes"] * valid_count / clip_size)
                    for offset in range(valid_count):
                        metrics = evaluated["frame_metrics"][offset]
                        item = {"sequence_id": row["sequence_id"], "method": "bistar", "target_kbps": target,
                                "frame_id": start + offset, **metrics}
                        frame_rows.append(item); current.append(item)
                duration = len(originals) / fps; actual = total_bytes * 8 / duration / 1000
                aggregate = _aggregate(current, ["psnr", "ssim", "ms_ssim", "lpips"])
                sequence_rows.append({"sequence_id": row["sequence_id"], "method": "bistar", "target_kbps": target,
                                      "actual_bitrate_kbps": actual, "total_bytes": total_bytes, "payload_bytes": payload_bytes,
                                      "metadata_bytes": metadata_bytes, "num_frames": len(originals), "encode_time_sec": 0.0,
                                      "decode_time_sec": inference, "status": "matched" if abs(actual-target)/target <= config["rd"]["bitrate_match_tolerance_percent"]/100 else "bitrate_unreachable", **aggregate})
    write_csv(frame_rows, output / "frame_results.csv"); write_csv(sequence_rows, output / "sequence_results.csv")
    write_csv(sequence_rows, output / "metrics.csv"); write_csv(sequence_rows, output / "rd_points.csv")
    from src.reports.plots import plot_rd
    from src.reports.tables import write_rd_table, write_runtime_table
    plot_rd(sequence_rows, output / "plots")
    write_rd_table(sequence_rows, output / "tables/rate_distortion.csv")
    write_runtime_table(sequence_rows, output / "tables/runtime.csv")
    summary = {"status": "complete", "sequences": len(rows), "points": len(sequence_rows), "frames": len(frame_rows)}
    write_summary(summary, output / "summary.json"); return summary
