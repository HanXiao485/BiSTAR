import copy
from pathlib import Path

from src.codecs import H264Adapter, H265Adapter
from src.common.config import load_codec_config, load_yaml, resolve_path, validate_experiment_config
from src.datasets.manifest import read_manifest
from src.datasets.frame_reader import preprocess_video, read_rgb_frames
from src.experiments.bistar_runner import evaluate_videomae_clip, select_budget_for_rate
from .runner import PROJECT_ROOT, prepare_run, write_csv, write_summary


def run_matched(config_path):
    config = validate_experiment_config(load_yaml(config_path), "matched_bitrate"); output = prepare_run(config, resolve_path(config["experiment"]["output_dir"], PROJECT_ROOT))
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
                results.append({"sequence_id": sequence["sequence_id"], "method": method, "target_kbps": target,
                                "requested_kbps": requested, "actual_kbps": actual, "error_percent": (actual-target)/target*100,
                                "iterations": iteration, "status": "matched" if abs(actual-target)/target*100 <= options["tolerance_percent"] else "bitrate_unreachable"})
        frames = read_rgb_frames(source)
        clip_size = 16
        if len(frames) >= clip_size and "bistar_videomae" in config.get("methods", []):
            clip = frames[:clip_size]
            for target in options["target_kbps"]:
                matched = select_budget_for_rate(clip, target, float(ds["preprocessing"]["target_fps"]))
                evaluated = evaluate_videomae_clip(clip, matched["encoded"])
                results.append({"sequence_id": sequence["sequence_id"], "method": "bistar_videomae", "target_kbps": target,
                                "requested_kbps": None, "actual_kbps": matched["actual_kbps"],
                                "error_percent": (matched["actual_kbps"]-target)/target*100, "iterations": matched["budget"],
                                "status": matched["status"], "visible_patch_count": matched["budget"],
                                "inference_ms_per_frame": evaluated["inference_ms_per_frame"]})
    write_csv(results, output / "metrics.csv"); write_summary({"status": "complete", "results": results}, output / "summary.json"); return results
