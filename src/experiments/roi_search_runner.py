import copy
import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from src.bistar.reconstruction.videomae_adapter import VideoMAEAdapter
from src.common.config import dump_yaml, load_yaml, resolve_path, validate_experiment_config
from src.common.hashing import sha256_file
from src.datasets.frame_reader import preprocess_video, read_rgb_frames, write_rgb_video
from src.datasets.manifest import read_manifest
from src.datasets.task_roi import frame_annotations, read_task_annotations
from src.experiments.config_search import OBJECTIVES, choose_winners, compare_validation, expand_candidates
from src.experiments.runner import PROJECT_ROOT, prepare_run, write_csv, write_summary
from src.experiments.search_codecs import evaluate_candidate
from src.metrics.task_fidelity import evaluate_task_sequence


def load_search_config(path):
    config = validate_experiment_config(load_yaml(path), "roi_search")
    for section in ("roi", "search", "methods", "model"):
        if section not in config:
            raise ValueError("ROI search requires {}".format(section))
    options = config["search"]
    for key in ("target_bitrates_kbps", "tolerance_percent", "pair_tolerance_percent",
                "max_match_iterations", "objective", "partition"):
        if key not in options:
            raise ValueError("search requires {}".format(key))
    if options["objective"] not in OBJECTIVES:
        raise ValueError("Unsupported search objective")
    if not options["target_bitrates_kbps"] or any(float(rate) <= 0 for rate in options["target_bitrates_kbps"]):
        raise ValueError("Target bitrates must be positive")
    if int(options["max_match_iterations"]) < 1:
        raise ValueError("max_match_iterations must be positive")
    if not (0 < float(options["tolerance_percent"]) <= 100 and
            0 < float(options["pair_tolerance_percent"]) <= 100):
        raise ValueError("Rate tolerances must be in (0, 100]")
    return config


def search_plan(config):
    candidates = expand_candidates(config["methods"], lambda path: load_yaml(resolve_path(path, PROJECT_ROOT)))
    for candidate in candidates:
        codec = candidate["config"]["codec"]
        encoder = codec.get("encoder", {})
        if encoder.get("b_frames", 0) != 0:
            raise ValueError("Formal low-delay comparison requires b_frames=0")
        if candidate["method"] in ("bistar", "bistar_atlas"):
            if int(codec["patch_size"]) != 16:
                raise ValueError("Search uses a common 16x16 patch grid")
            mode = codec["selector"].get("patch_budget_mode", True)
            if not isinstance(mode, bool):
                raise ValueError("selector.patch_budget_mode must be a boolean")
            from src.experiments.atlas_runner import SELECTORS
            if codec["selector"]["type"] not in SELECTORS:
                raise ValueError("Unknown sparse selector")
            gop = int(codec.get("gop_size", 16))
            if gop < 1 or (codec["reconstruction"]["type"] == "videomae" and gop > 16):
                raise ValueError("Sparse GOP size must be positive and at most 16 for VideoMAE")
            if codec["reconstruction"]["type"] not in ("videomae", "previous_frame", "interpolation"):
                raise ValueError("Unknown sparse reconstruction")
    return candidates


def _sequence_partitions(config, dataset):
    options = config["search"]["partition"]
    manifest = resolve_path(dataset["manifest"], PROJECT_ROOT)
    mode = options["mode"]
    if mode == "manifest_splits":
        calibration = read_manifest(manifest, options["calibration_split"])
        validation = read_manifest(manifest, options["validation_split"])
        if not calibration or not validation:
            raise ValueError("Formal search needs nonempty calibration and validation manifest splits")
        if {row["sequence_id"] for row in calibration} & {row["sequence_id"] for row in validation}:
            raise ValueError("Calibration and validation sequence IDs must be disjoint")
        return [("calibration", row) for row in calibration] + [("validation", row) for row in validation]
    if mode == "temporal_holdout":
        fraction = float(options["calibration_fraction"])
        if not 0 < fraction < 1 or int(options.get("gap_frames", 16)) < 1:
            raise ValueError("Temporal holdout needs fraction in (0,1) and positive gap_frames")
        rows = read_manifest(manifest, config["dataset"].get("split", "test"))
        if not rows:
            raise ValueError("No manifest rows for temporal holdout")
        return [(phase, row) for phase in ("calibration", "validation") for row in rows]
    raise ValueError("partition.mode must be manifest_splits or temporal_holdout")


def _prepare_sources(config, dataset, output, annotations):
    sources, materialized = {}, {}
    records = []
    for phase, row in _sequence_partitions(config, dataset):
        sequence_id = row["sequence_id"]
        raw = resolve_path(row["video_path"], PROJECT_ROOT)
        if sequence_id not in materialized:
            source_path = output / "sources" / sequence_id / "common.mkv"
            common = preprocess_video(raw, source_path, dataset["preprocessing"])
            frames = read_rgb_frames(source_path)
            materialized[sequence_id] = (frames, common, sha256_file(raw))
        frames, common, raw_hash = materialized[sequence_id]
        partition = config["search"]["partition"]
        if partition["mode"] == "temporal_holdout":
            boundary = int(len(frames) * float(partition["calibration_fraction"]))
            start, end = (0, boundary) if phase == "calibration" else (boundary + int(partition.get("gap_frames", 16)), len(frames))
        else:
            start, end = 0, len(frames)
        if end <= start:
            raise ValueError("Empty {} partition for {}".format(phase, sequence_id))
        selected = frames[start:end]
        frame_ids = list(range(start, end))
        roi = config["roi"]
        masks, gaze, provenance = frame_annotations(
            annotations, sequence_id, frame_ids, selected[0].shape[0], selected[0].shape[1],
            roi.get("coordinate_space", "normalized"),
        )
        if roi.get("use_gaze_for_selection", False) and any(point is None for point in gaze):
            raise ValueError("use_gaze_for_selection requires gaze annotations on every evaluated frame")
        source = output / "sources" / sequence_id / (phase + ".mkv")
        # Lossless slice materialization avoids counting other partition frames in rate.
        command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", common["path"],
                   "-vf", "trim=start_frame={}:end_frame={},setpts=PTS-STARTPTS".format(start, end),
                   "-an", "-c:v", "ffv1", str(source)]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if len(read_rgb_frames(source)) != len(selected):
            raise RuntimeError("Partition source frame alignment failed")
        sources[(phase, sequence_id)] = {"frames": selected, "source": source, "masks": masks,
                                         "gaze": gaze, "frame_ids": frame_ids}
        records.append({"phase": phase, "sequence_id": sequence_id, "source_path": str(raw),
                        "source_sha256": raw_hash, "preprocessed_sha256": sha256_file(source),
                        "start_frame": start, "end_frame_exclusive": end, "num_frames": len(selected),
                        "provenance": provenance, "preprocessing": common})
    if config["search"]["partition"]["mode"] == "manifest_splits":
        hashes = {phase: {row["source_sha256"] for row in records if row["phase"] == phase}
                  for phase in ("calibration", "validation")}
        if hashes["calibration"] & hashes["validation"]:
            raise ValueError("Identical source videos appear in calibration and validation splits")
    write_summary(records, output / "source_metadata.json")
    return sources


def run_roi_search(config_path, device=None, plan_only=False, model_adapter=None):
    config = load_search_config(config_path)
    if device is not None:
        config["model"]["device_override"] = str(device)
    candidates = search_plan(config)
    if plan_only:
        return {"candidates": {method: sum(item["method"] == method for item in candidates)
                               for method in config["methods"]},
                "target_bitrates_kbps": config["search"]["target_bitrates_kbps"],
                "partition": config["search"]["partition"],
                "annotations": str(resolve_path(config["roi"]["annotations"], PROJECT_ROOT))}
    annotation_path = resolve_path(config["roi"]["annotations"], PROJECT_ROOT)
    if not annotation_path.is_file():
        raise ValueError("Missing task ROI annotations: {}. Run scripts/prepare_task_roi.py first.".format(annotation_path))
    annotations = read_task_annotations(annotation_path, config["roi"].get("coordinate_space", "normalized"))
    provenance = sorted({annotation["provenance"] for values in annotations.values() for annotation in values})
    proxy = any(value.startswith("example_") for value in provenance)
    if proxy and not config["roi"].get("allow_proxy_annotations", False):
        raise ValueError("Example proxy ROIs are not accepted for formal search; annotate task ROIs or explicitly enable roi.allow_proxy_annotations for a pipeline check")
    output = resolve_path(config["experiment"]["output_dir"], PROJECT_ROOT)
    dataset = load_yaml(resolve_path(config["dataset"]["config"], PROJECT_ROOT))["dataset"]
    manifest_path = resolve_path(dataset["manifest"], PROJECT_ROOT)
    model_config = load_yaml(resolve_path(config["model"]["config"], PROJECT_ROOT))
    if config["model"].get("device_override"):
        model_config["model"]["runtime"]["device"] = config["model"]["device_override"]
    import torch
    seed = int(config["experiment"]["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    execution = {"torch_num_threads": torch.get_num_threads(), "seed": seed,
                 "cuda_available": torch.cuda.is_available(),
                 "model_device": model_config["model"]["runtime"]["device"]}
    snapshot = {"experiment": config, "dataset": dataset, "model": model_config,
                "execution": execution,
                "candidates": candidates, "annotations_sha256": sha256_file(annotation_path),
                "manifest_sha256": sha256_file(manifest_path),
                "source_hashes": {str(resolve_path(row["video_path"], PROJECT_ROOT)):
                                  sha256_file(resolve_path(row["video_path"], PROJECT_ROOT))
                                  for _, row in _sequence_partitions(config, dataset)},
                "code_hashes": {str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
                                for path in sorted((PROJECT_ROOT / "src").rglob("*.py"))}}
    checkpoint = resolve_path(model_config["model"]["checkpoint"]["path"], PROJECT_ROOT)
    snapshot["checkpoint_sha256"] = sha256_file(checkpoint) if checkpoint.is_file() else None
    signature = hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()
    signature_file = output / "run_signature.txt"
    if signature_file.exists() and signature_file.read_text().strip() != signature:
        raise ValueError("Output contains a different search run; choose a new experiment.output_dir")
    output = prepare_run(config, output)
    signature_file.write_text(signature + "\n", encoding="utf-8")
    dump_yaml(snapshot, output / "config_snapshot.yaml")
    shutil.copyfile(annotation_path, output / "annotations.csv")
    shutil.copyfile(manifest_path, output / "manifest.csv")
    code_snapshot = output / "code_snapshot"
    for path in (PROJECT_ROOT / "src").rglob("*.py"):
        target = code_snapshot / path.relative_to(PROJECT_ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    diff = subprocess.run(["git", "diff", "HEAD"], cwd=PROJECT_ROOT, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output / "working_tree.patch").write_text(diff.stdout, encoding="utf-8")
    catalog = [{"candidate_id": candidate["candidate_id"], "method": candidate["method"],
                "overrides": json.dumps(candidate["overrides"], sort_keys=True)} for candidate in candidates]
    write_csv(catalog, output / "candidate_catalog.csv")
    for candidate in candidates:
        dump_yaml(candidate["config"], output / (candidate["candidate_id"] + ".yaml"))
    sources = _prepare_sources(config, dataset, output, annotations)
    from src.reports.search_plots import plot_roi_preview
    plot_roi_preview(sources, output / "plots" / "annotations")
    models = [model_adapter]

    def get_model():
        if models[0] is None:
            models[0] = VideoMAEAdapter(model_config)
            models[0].load_model()
        return models[0]

    # Fail once before the sweep if the requested inference device is unavailable.
    if any(candidate["config"]["codec"].get("reconstruction", {}).get("type") == "videomae"
           for candidate in candidates):
        get_model()
    all_rows, validation_rows, validation_patches = [], [], []
    failed = []

    def trial(phase, sequence_id, candidate, target):
        trial_dir = output / "trials" / phase / sequence_id / candidate["candidate_id"] / str(target)
        trial_dir.mkdir(parents=True, exist_ok=True)
        result_file = trial_dir / "result.json"
        if config["search"].get("resume", True) and result_file.exists():
            cached = json.loads(result_file.read_text())
            if cached.get("status") != "failed":
                return cached
        source = sources[(phase, sequence_id)]
        metadata = {"phase": phase, "sequence_id": sequence_id, "method": candidate["method"],
                    "candidate_id": candidate["candidate_id"], "target_kbps": target}
        try:
            evaluated = evaluate_candidate(
                source["frames"], source["source"], float(dataset["preprocessing"]["target_fps"]),
                dataset["preprocessing"], candidate, trial_dir, target, config["search"], get_model,
                source["gaze"], use_gaze=config["roi"].get("use_gaze_for_selection", False),
            )
            row = {**metadata, **{key: value for key, value in evaluated.items()
                                  if key not in ("reconstructed", "visible_masks", "gop_results")}}
            row["rate_error_percent"] = (row["actual_bitrate_kbps"] - target) / target * 100
            if evaluated["status"] == "matched":
                metrics = config.get("metrics", {})
                quality, frame_rows, patch_rows = evaluate_task_sequence(
                    source["frames"], evaluated["reconstructed"], source["masks"], metadata,
                    evaluated["visible_masks"], psnr_threshold=metrics.get("patch_psnr_threshold", 30.0),
                    ssim_threshold=metrics.get("patch_ssim_threshold", 0.9),
                )
                row.update(quality)
                for frame, frame_id in zip(frame_rows, source["frame_ids"]):
                    frame["frame_id"] = frame_id
                write_csv(frame_rows, trial_dir / "frame_results.csv")
                write_csv(patch_rows, trial_dir / "patch_results.csv")
                if config["experiment"].get("save_videomae_videos", True):
                    video = write_rgb_video(evaluated["reconstructed"], trial_dir / "reconstructed.mp4",
                                            float(dataset["preprocessing"]["target_fps"]))
                    row["reconstruction_video"] = video["path"]
                    row["reconstruction_video_bytes"] = video["bytes"]
            write_summary(row, result_file)
            return row
        except Exception as error:
            row = {**metadata, "status": "failed", "error": str(error)}
            write_summary(row, result_file)
            return row

    rates = config["search"]["target_bitrates_kbps"]
    calibration_ids = [sequence for phase, sequence in sources if phase == "calibration"]
    total_trials = len(calibration_ids) * len(candidates) * len(rates)
    count = 0
    for candidate in candidates:
        for target in rates:
            for sequence_id in calibration_ids:
                row = trial("calibration", sequence_id, candidate, target)
                all_rows.append(row)
                if row["status"] == "failed":
                    failed.append(row)
                count += 1
                print("[{}/{}] {} {} kbps {}: {}".format(count, total_trials, candidate["candidate_id"],
                                                       target, sequence_id, row["status"]), flush=True)
                write_csv(all_rows, output / "calibration_results.csv")
    winners = choose_winners(all_rows, config["search"]["objective"])
    write_csv(winners, output / "best_configs.csv")
    catalog_by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    for winner in winners:
        candidate = catalog_by_id[winner["candidate_id"]]
        frozen = output / "best_configs" / str(winner["target_kbps"])
        frozen.mkdir(parents=True, exist_ok=True)
        dump_yaml(candidate["config"], frozen / (winner["method"] + ".yaml"))
        for phase, sequence_id in sources:
            if phase != "validation":
                continue
            row = trial("validation", sequence_id, candidate, winner["target_kbps"])
            validation_rows.append(row)
            if row["status"] == "failed":
                failed.append(row)
            patch_path = output / "trials" / phase / sequence_id / candidate["candidate_id"] / str(winner["target_kbps"]) / "patch_results.csv"
            if row["status"] == "matched" and patch_path.exists():
                with patch_path.open() as handle:
                    validation_patches.extend(csv.DictReader(handle))
            print("[validation] {} {} kbps {}: {}".format(candidate["candidate_id"], winner["target_kbps"],
                                                       sequence_id, row["status"]), flush=True)
    comparisons = compare_validation(validation_rows, config["search"]["objective"],
                                     config["search"]["pair_tolerance_percent"])
    write_csv(validation_rows, output / "validation_results.csv")
    write_csv(comparisons, output / "comparisons.csv")
    write_csv(validation_patches, output / "validation_patch_results.csv")
    from src.reports.search_plots import plot_search_results
    plot_search_results(all_rows, validation_rows, comparisons, validation_patches, output / "plots", candidates)
    groups = {(row["sequence_id"], row["target_kbps"], row["method"]) for row in comparisons}
    proven = sum(all(row["outperforms"] and row["comparable"] for row in comparisons
                     if (row["sequence_id"], row["target_kbps"], row["method"]) == group)
                 and len([row for row in comparisons if (row["sequence_id"], row["target_kbps"], row["method"]) == group]) == 2
                 for group in groups)
    expected_winners = len({candidate["method"] for candidate in candidates}) * len(rates)
    summary = {"status": "complete" if not failed else "complete_with_failures", "run_signature": signature,
               "calibration_trials": len(all_rows), "matched_calibration_trials": sum(row["status"] == "matched" for row in all_rows),
               "selected_configs": len(winners), "expected_selected_configs": expected_winners,
               "unmatched_method_rate_groups": expected_winners - len(winners),
               "validation_trials": len(validation_rows), "failed_trials": len(failed),
               "objective": config["search"]["objective"], "partition": config["search"]["partition"],
               "outperforms_both_baselines_groups": int(proven),
               "roi_annotation_source": str(annotation_path),
               "annotation_provenance": provenance, "proxy_ROI_only": proxy,
               "claim_scope": "held_out_visual_ROI_metrics_only_not_teleoperation_task_success",
               "bitrate_scope": "codec_and_application_wire_bytes_excludes_network_headers",
               "fidelity_definition": "ROI PSNR/SSIM/L1 and thresholded critical-patch retention; not mutual information"}
    write_summary(summary, output / "summary.json")
    lines = ["# ROI configuration search", "", "Objective: `{}`".format(summary["objective"]),
             "", "Calibration selects one configuration per method and target. Held-out results do not select configurations.",
             "", "Critical-patch retention is ROI-area-weighted thresholded PSNR/SSIM, not a mutual-information estimate.",
             "", "Annotation provenance: {}. Proxy ROIs: {}.".format(", ".join(provenance), proxy),
             "", "Both-baseline wins in comparable held-out sequence/rate/method groups: {}.".format(proven),
             "", "Pairwise actual-rate tolerance: {}%. Unmatched pairs have no superiority claim.".format(
                 config["search"]["pair_tolerance_percent"]),
             "", "| Method | Baseline | Target kbps | Sequence | Delta | Comparable | Outperforms |",
             "| --- | --- | ---: | --- | ---: | --- | --- |"]
    for row in comparisons:
        lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            row["method"], row["baseline"], row["target_kbps"], row["sequence_id"],
            "{:.6f}".format(row["delta"]) if row["delta"] is not None else "n/a",
            row["comparable"], row["outperforms"]))
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
