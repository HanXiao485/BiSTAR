import copy
from pathlib import Path

from src.codecs import H264Adapter, H265Adapter
from src.common.config import load_codec_config, load_yaml, resolve_path, validate_experiment_config
from src.datasets.manifest import read_manifest
from src.datasets.frame_reader import preprocess_video, read_rgb_frames, write_rgb_video
from src.experiments.bistar_runner import evaluate_videomae_clip, resolve_patch_budget_options, select_budget_for_rate
from src.bistar.reconstruction.videomae_adapter import VideoMAEAdapter
from .runner import PROJECT_ROOT, prepare_run, write_csv, write_summary


def run_matched(config_path):
    config = validate_experiment_config(load_yaml(config_path), "matched_bitrate"); output = prepare_run(config, resolve_path(config["experiment"]["output_dir"], PROJECT_ROOT))
    save_videomae_videos = bool(config["experiment"].get("save_videomae_videos", False))
    patch_budget_mode, fixed_patch_count = resolve_patch_budget_options(config)
    ds = load_yaml(resolve_path(config["dataset"]["config"], PROJECT_ROOT))["dataset"]
    sequences = read_manifest(resolve_path(ds["manifest"], PROJECT_ROOT), config["dataset"].get("split", "test"))
    options = config["matched_bitrate"]; results = []
    for sequence in sequences:
        raw_source = resolve_path(sequence["video_path"], PROJECT_ROOT)
        common = preprocess_video(raw_source, output / "sources" / (sequence["sequence_id"] + ".mkv"), ds["preprocessing"])
        source = Path(common["path"])
        for method, cls in (("h264", H264Adapter), ("h265", H265Adapter)):
            cfg = load_codec_config(PROJECT_ROOT / "configs/codecs/{}.yaml".format(method)); cfg["preprocessing"] = ds["preprocessing"]
            for target in options["target_kbps"]:
                requested = float(target); actual = 0.0
                for iteration in range(1, options["max_iterations"] + 1):
                    trial = copy.deepcopy(cfg); trial["codec"]["encoder"]["bitrate_kbps"] = round(requested, 3)
                    encoded = cls().encode_sequence(source, output / "artifacts" / sequence["sequence_id"] / method / str(target) / str(iteration), trial)
                    actual = encoded["actual_bitrate_kbps"]; error = abs(actual - target) / target * 100
                    if error <= options["tolerance_percent"]: break
                    requested *= max(0.5, min(2.0, target / actual))
                decoded = cls().decode_sequence(
                    Path(encoded["bitstream_path"]),
                    output / "artifacts" / sequence["sequence_id"] / method / str(target) / "decoded",
                    trial,
                )
                reconstructed = read_rgb_frames(Path(decoded["decoded_path"]))
                video = write_rgb_video(
                    reconstructed,
                    output / "reconstructions" / sequence["sequence_id"] /
                    "{}_{}kbps.mp4".format(method, target),
                    float(ds["preprocessing"]["target_fps"]),
                )
                results.append({"sequence_id": sequence["sequence_id"], "method": method, "target_kbps": target,
                                "requested_kbps": requested, "actual_kbps": actual, "error_percent": (actual-target)/target*100,
                                "iterations": iteration, "status": "matched" if abs(actual-target)/target*100 <= options["tolerance_percent"] else "bitrate_unreachable",
                                "reconstruction_video": video["path"], "reconstruction_video_bytes": video["bytes"]})
        frames = read_rgb_frames(source)
        clip_size = 16
        if frames and "bistar_videomae" in config.get("methods", []):
            model_adapter = VideoMAEAdapter(load_yaml(PROJECT_ROOT / "configs/models/videomae.yaml"))
            model_adapter.load_model()
            for target in options["target_kbps"]:
                reconstructed_frames, budgets = [], []
                total_bytes = 0
                inference_time = 0.0
                fps = float(ds["preprocessing"]["target_fps"])
                for start in range(0, len(frames), clip_size):
                    valid = frames[start:start + clip_size]
                    valid_count = len(valid)
                    clip = valid + [valid[-1]] * (clip_size - valid_count)
                    matched = select_budget_for_rate(
                        clip, target, fps,
                        fixed_visible_patch_count=fixed_patch_count,
                        tolerance_percent=options["tolerance_percent"],
                    )
                    evaluated = evaluate_videomae_clip(
                        clip, matched["encoded"], adapter=model_adapter, compute_metrics=False
                    )
                    reconstructed_frames.extend(evaluated["reconstructed"][:valid_count])
                    total_bytes += int(evaluated["total_bytes"] * valid_count / clip_size)
                    inference_time += evaluated["inference_time_sec"] * valid_count / clip_size
                    budgets.append(matched["budget"])
                actual = total_bytes * 8.0 / (len(frames) / fps) / 1000.0
                video = None
                if save_videomae_videos:
                    video = write_rgb_video(
                        reconstructed_frames,
                        output / "reconstructions" / sequence["sequence_id"] / "bistar_videomae_{}kbps.mp4".format(target),
                        fps,
                    )
                results.append({"sequence_id": sequence["sequence_id"], "method": "bistar_videomae", "target_kbps": target,
                                "requested_kbps": None, "actual_kbps": actual,
                                "error_percent": (actual-target)/target*100, "iterations": len(budgets),
                                "status": "matched" if abs(actual-target)/target*100 <= options["tolerance_percent"] else "bitrate_unreachable",
                                "patch_budget_mode": patch_budget_mode,
                                "patch_budget_strategy": "fixed" if patch_budget_mode else "bitrate_match",
                                "visible_patch_count": float(sum(budgets)) / len(budgets),
                                "visible_patch_counts": ",".join(str(value) for value in budgets),
                                "inference_ms_per_frame": inference_time * 1000.0 / len(frames),
                                "reconstruction_video": video["path"] if video else None,
                                "reconstruction_video_bytes": video["bytes"] if video else None})
    write_csv(results, output / "metrics.csv"); write_summary({"status": "complete", "results": results}, output / "summary.json"); return results
