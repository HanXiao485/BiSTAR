import copy
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from src.bistar.reconstruction.videomae_adapter import VideoMAEAdapter
from src.common.config import dump_yaml, load_yaml, resolve_path, validate_experiment_config
from src.common.hashing import sha256_file
from src.common.progress import ProgressBar
from src.datasets.frame_reader import preprocess_video, read_rgb_frames, write_rgb_video
from src.datasets.manifest import read_manifest
from src.datasets.task_roi import frame_annotations_with_validity, read_task_annotations
from src.experiments.config_search import OBJECTIVES, choose_winners, compare_validation, expand_candidates
from src.experiments.runner import PROJECT_ROOT, json_safe, prepare_run, write_csv, write_summary
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
    experiment = config["experiment"]
    scene_limit = experiment.get("max_comparison_video_scenes", 3)
    if scene_limit is not None and int(scene_limit) < 0:
        raise ValueError("max_comparison_video_scenes must be nonnegative or null")
    requested_scenes = experiment.get("comparison_video_sequence_ids", [])
    if requested_scenes is not None and not isinstance(requested_scenes, list):
        raise ValueError("comparison_video_sequence_ids must be a list or null")
    if experiment.get("artifact_retention", "all") not in ("all", "plot_data_only"):
        raise ValueError("artifact_retention must be all or plot_data_only")
    return config


def _select_comparison_sequences(config, sources, eligible=None):
    """Choose held-out scenes independently of their measured quality."""
    available = sorted(sequence for phase, sequence in sources if phase == "validation")
    if eligible is not None:
        available = [sequence for sequence in available if sequence in set(eligible)]
    experiment = config["experiment"]
    requested = experiment.get("comparison_video_sequence_ids") or []
    missing = sorted(set(requested) - set(available))
    if missing:
        raise ValueError("Comparison video IDs are unavailable or not fully rate-matched: {}".format(
            ", ".join(missing)))
    limit = experiment.get("max_comparison_video_scenes", 3)
    limit = len(available) if limit is None else int(limit)
    if requested:
        return list(dict.fromkeys(requested))[:limit]
    seed = int(experiment.get("seed", 0))
    ranked = sorted(available, key=lambda sequence: hashlib.sha256(
        "{}:{}".format(seed, sequence).encode("utf-8")).hexdigest())
    return ranked[:limit]


def _comparison_video_path(output, sequence_id, target, method):
    safe_sequence = sequence_id.replace("/", "_").replace("\\", "_")
    safe_method = method.replace("/", "_").replace("\\", "_")
    return Path(output) / "comparison_videos" / safe_sequence / "{}kbps".format(target) / (safe_method + ".mp4")


def _montage_filter(input_count):
    """Return an FFmpeg filter for a compact row-major grid with an optional blank cell."""
    if int(input_count) < 1:
        raise ValueError("Comparison montage requires at least one method")
    columns = int(math.ceil(math.sqrt(input_count)))
    rows = int(math.ceil(input_count / columns))
    filters = []
    row_labels = []
    for row in range(rows):
        indices = list(range(row * columns, min((row + 1) * columns, input_count)))
        label = "row{}".format(row)
        inputs = "".join("[{}:v]".format(index) for index in indices)
        if len(indices) == 1:
            filters.append("{}null[{}raw]".format(inputs, label))
        else:
            filters.append("{}hstack=inputs={}[{}raw]".format(inputs, len(indices), label))
        if len(indices) < columns:
            filters.append("[{}raw]pad=iw*{}/{}:ih:0:0:color=black[{}]".format(
                label, columns, len(indices), label))
        else:
            filters.append("[{}raw]null[{}]".format(label, label))
        row_labels.append("[{}]".format(label))
    if len(row_labels) == 1:
        filters.append("{}null[v]".format(row_labels[0]))
    else:
        filters.append("{}vstack=inputs={}[v]".format("".join(row_labels), len(row_labels)))
    return ";".join(filters), rows, columns


def _comparison_layout(methods):
    _, rows, columns = _montage_filter(len(methods))
    return "grid={}x{}; row-major={}".format(columns, rows, ",".join(methods))


def _write_comparison_montage(video_paths, output_path, methods=None):
    """Create a compact comparison grid for any positive number of methods."""
    methods = list(methods or video_paths)
    if not methods or any(method not in video_paths or not Path(video_paths[method]).is_file()
                          for method in methods):
        return None
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    for method in methods:
        command += ["-i", str(video_paths[method])]
    filter_graph, _, _ = _montage_filter(len(methods))
    command += [
        "-filter_complex", filter_graph,
        "-map", "[v]", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-shortest", str(output_path),
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError("Comparison montage failed: {}".format(
            result.stderr.decode("utf-8", errors="replace")))
    return output_path


def _write_comparison_manifest(output, selected_sequences, rates, validation_rows, methods):
    methods = list(methods)
    layout = _comparison_layout(methods)
    records = []
    for sequence_id in selected_sequences:
        for target in rates:
            videos = {}
            for method in methods:
                row = next((item for item in validation_rows
                            if item["sequence_id"] == sequence_id and item["target_kbps"] == target
                            and item["method"] == method), None)
                video_path = _comparison_video_path(output, sequence_id, target, method)
                exists = video_path.is_file()
                if exists:
                    videos[method] = video_path
                records.append({"sequence_id": sequence_id, "target_kbps": target, "method": method,
                                "candidate_id": row.get("candidate_id") if row else None,
                                "status": row.get("status") if row else "missing_result",
                                "roi_ssim": row.get("roi_ssim") if row else None,
                                "actual_bitrate_kbps": row.get("actual_bitrate_kbps") if row else None,
                                "video_path": str(video_path) if exists else None,
                                "montage_layout": layout})
            montage = _write_comparison_montage(
                videos, _comparison_video_path(output, sequence_id, target, "comparison_grid"), methods)
            if montage is not None:
                records.append({"sequence_id": sequence_id, "target_kbps": target,
                                "method": "comparison_grid", "candidate_id": None,
                                "status": "created", "roi_ssim": None, "actual_bitrate_kbps": None,
                                "video_path": str(montage), "montage_layout": layout})
    write_csv(records, Path(output) / "comparison_video_manifest.csv")
    return records


def _prune_intermediate_artifacts(output, candidates):
    """Remove resumable working data only after a fully successful run."""
    output = Path(output)
    removed = []
    for directory in (output / "trials", output / "sources"):
        if directory.exists():
            shutil.rmtree(directory)
            removed.append(str(directory.relative_to(output)))
    for candidate in candidates:
        path = output / (candidate["candidate_id"] + ".yaml")
        if path.exists():
            path.unlink()
            removed.append(str(path.relative_to(output)))
    metadata = {
        "policy": "plot_data_only",
        "cleanup_complete": True,
        "removed": removed,
        "retained": ["plots", "comparison_videos", "best_configs", "summary/report files",
                     "plot-source CSV files", "configuration and provenance metadata"],
    }
    write_summary(metadata, output / "artifact_manifest.json")
    return metadata


def _read_trial_cache(path, expected):
    """Treat interrupted or mismatched trial records as cache misses."""
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(cached, dict) or cached.get("status") not in ("matched", "bitrate_unreachable"):
        return None
    if any(cached.get(key) != value for key, value in expected.items()):
        return None
    if cached.get("actual_bitrate_kbps") is None:
        return None
    if cached["status"] == "matched" and cached.get("roi_ssim") is None:
        return None
    return cached


def _write_trial_result(row, path):
    """Replace a finished JSON record atomically so interruption cannot truncate it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=".result-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(json_safe(row), handle, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _compatible_resume_signature(output, snapshot, saved_signature):
    """Allow a runner-only recovery fix while preserving the original run record."""
    snapshot_file = output / "config_snapshot.yaml"
    original_runner = output / "code_snapshot" / "src" / "experiments" / "roi_search_runner.py"
    if not snapshot_file.is_file() or not original_runner.is_file():
        return False
    previous = load_yaml(snapshot_file)
    if hashlib.sha256(json.dumps(previous, sort_keys=True).encode()).hexdigest() != saved_signature:
        return False
    old_hashes = previous.pop("code_hashes", {})
    new_copy = copy.deepcopy(snapshot)
    new_hashes = new_copy.pop("code_hashes", {})
    runner_key = "src/experiments/roi_search_runner.py"
    if previous != new_copy or old_hashes.get(runner_key) != sha256_file(original_runner):
        return False
    if old_hashes.get(runner_key) == new_hashes.get(runner_key):
        return False
    old_hashes.pop(runner_key)
    new_hashes.pop(runner_key)
    return old_hashes == new_hashes


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
    sources, materialized, slices = {}, {}, {}
    records = []
    for phase, row in _sequence_partitions(config, dataset):
        sequence_id = row["sequence_id"]
        raw = resolve_path(row["video_path"], PROJECT_ROOT)
        materialized_key = str(raw.resolve())
        if materialized_key not in materialized:
            shared_id = hashlib.sha256(materialized_key.encode()).hexdigest()[:16]
            source_path = output / "sources" / "_shared" / shared_id / "common.mkv"
            common = preprocess_video(raw, source_path, dataset["preprocessing"])
            frames = read_rgb_frames(source_path)
            materialized[materialized_key] = (frames, common, sha256_file(raw), shared_id)
        frames, common, raw_hash, shared_id = materialized[materialized_key]
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
        masks, gaze, provenance, roi_validity, gaze_validity = frame_annotations_with_validity(
            annotations, sequence_id, frame_ids, selected[0].shape[0], selected[0].shape[1],
            roi.get("coordinate_space", "normalized"),
        )
        if roi.get("use_gaze_for_selection", False) and any(point is None for point in gaze):
            raise ValueError("use_gaze_for_selection requires gaze annotations on every evaluated frame")
        slice_key = (materialized_key, start, end)
        if slice_key not in slices:
            source = output / "sources" / "_shared" / shared_id / "{}_{}.mkv".format(start, end)
            # Lossless slice materialization avoids counting other partition frames in rate.
            command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", common["path"],
                       "-vf", "trim=start_frame={}:end_frame={},setpts=PTS-STARTPTS".format(start, end),
                       "-an", "-c:v", "ffv1", str(source)]
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if len(read_rgb_frames(source)) != len(selected):
                raise RuntimeError("Partition source frame alignment failed")
            slices[slice_key] = source
        source = slices[slice_key]
        sources[(phase, sequence_id)] = {"frames": selected, "source": source, "masks": masks,
                                         "gaze": gaze, "frame_ids": frame_ids,
                                         "roi_validity": roi_validity, "gaze_validity": gaze_validity}
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
                "selection_objective": config["search"]["objective"],
                "max_comparison_video_scenes": config["experiment"].get("max_comparison_video_scenes", 3),
                "artifact_retention": config["experiment"].get("artifact_retention", "all"),
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
    runner_only_resume = False
    if signature_file.exists():
        saved_signature = signature_file.read_text(encoding="utf-8").strip()
        completed = output / "summary.json"
        artifact_manifest = output / "artifact_manifest.json"
        if saved_signature == signature and completed.is_file() and artifact_manifest.is_file():
            previous_summary = json.loads(completed.read_text(encoding="utf-8"))
            retained = json.loads(artifact_manifest.read_text(encoding="utf-8"))
            if previous_summary.get("status") == "complete" and retained.get("cleanup_complete"):
                return previous_summary
        if saved_signature != signature:
            runner_only_resume = (config["search"].get("resume", True) and
                                  _compatible_resume_signature(output, snapshot, saved_signature))
            if not runner_only_resume:
                raise ValueError("Output contains a different search run; choose a new experiment.output_dir")
            write_summary({"original_run_signature": saved_signature,
                           "resume_code_signature": signature,
                           "changed_code_file": "src/experiments/roi_search_runner.py",
                           "reason": "recover interrupted trial cache and use atomic trial writes"},
                          output / "resume_code_metadata.json")
            signature = saved_signature
    output = prepare_run(config, output)
    if not runner_only_resume:
        signature_file.write_text(signature + "\n", encoding="utf-8")
        dump_yaml(snapshot, output / "config_snapshot.yaml")
    shutil.copyfile(annotation_path, output / "annotations.csv")
    shutil.copyfile(manifest_path, output / "manifest.csv")
    code_snapshot = output / ("resume_code_snapshot" if runner_only_resume else "code_snapshot")
    for path in (PROJECT_ROOT / "src").rglob("*.py"):
        target = code_snapshot / path.relative_to(PROJECT_ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    diff = subprocess.run(["git", "diff", "HEAD"], cwd=PROJECT_ROOT, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output / ("resume_working_tree.patch" if runner_only_resume else "working_tree.patch")).write_text(
        diff.stdout, encoding="utf-8")
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

    def trial(phase, sequence_id, candidate, target, comparison_video=None):
        trial_dir = output / "trials" / phase / sequence_id / candidate["candidate_id"] / str(target)
        trial_dir.mkdir(parents=True, exist_ok=True)
        result_file = trial_dir / "result.json"
        metadata = {"phase": phase, "sequence_id": sequence_id, "method": candidate["method"],
                    "candidate_id": candidate["candidate_id"], "target_kbps": target}
        if config["search"].get("resume", True) and result_file.exists():
            cached = _read_trial_cache(result_file, metadata)
            if cached is not None and (comparison_video is None or Path(comparison_video).is_file()):
                return cached
            reason = "missing comparison video" if cached is not None else "invalid trial cache"
            print("[resume] Recomputing {}: {}".format(reason, result_file), flush=True)
        source = sources[(phase, sequence_id)]
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
                    valid_frames=source["roi_validity"],
                )
                row.update(quality)
                for frame, frame_id in zip(frame_rows, source["frame_ids"]):
                    frame["frame_id"] = frame_id
                write_csv(frame_rows, trial_dir / "frame_results.csv")
                write_csv(patch_rows, trial_dir / "patch_results.csv")
                if comparison_video is not None:
                    video = write_rgb_video(evaluated["reconstructed"], comparison_video,
                                            float(dataset["preprocessing"]["target_fps"]))
                    row["reconstruction_video"] = video["path"]
                    row["reconstruction_video_bytes"] = video["bytes"]
            _write_trial_result(row, result_file)
            return row
        except Exception as error:
            row = {**metadata, "status": "failed", "error": str(error)}
            _write_trial_result(row, result_file)
            return row

    rates = config["search"]["target_bitrates_kbps"]
    calibration_ids = [sequence for phase, sequence in sources if phase == "calibration"]
    total_trials = len(calibration_ids) * len(candidates) * len(rates)
    count = 0
    progress = ProgressBar(total_trials, "ROI calibration")
    for candidate in candidates:
        for target in rates:
            for sequence_id in calibration_ids:
                row = trial("calibration", sequence_id, candidate, target)
                all_rows.append(row)
                if row["status"] == "failed":
                    failed.append(row)
                count += 1
                progress.update(count, "{} {} {}".format(candidate["method"], target, row["status"]))
                write_csv(all_rows, output / "calibration_results.csv")
    progress.close()
    winners = choose_winners(all_rows, config["search"]["objective"])
    write_csv(winners, output / "best_configs.csv")
    catalog_by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    expected_winners = len({candidate["method"] for candidate in candidates}) * len(rates)
    winner_method_set = {winner["method"] for winner in winners}
    comparison_methods = [method for method in config["methods"] if method in winner_method_set]
    comparison_methods += sorted(winner_method_set - set(comparison_methods))
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
    comparison_sequences = []
    if config["experiment"].get("save_videomae_videos", True):
        expected_pairs = {(winner["method"], winner["target_kbps"]) for winner in winners}
        eligible = []
        for phase, sequence_id in sources:
            if phase != "validation":
                continue
            matched = {(row["method"], row["target_kbps"]) for row in validation_rows
                       if row["sequence_id"] == sequence_id and row["status"] == "matched"}
            if len(expected_pairs) == expected_winners and matched == expected_pairs:
                eligible.append(sequence_id)
        comparison_sequences = _select_comparison_sequences(config, sources, eligible)
        for winner in winners:
            candidate = catalog_by_id[winner["candidate_id"]]
            for sequence_id in comparison_sequences:
                video_path = _comparison_video_path(
                    output, sequence_id, winner["target_kbps"], candidate["method"])
                video_row = trial("validation", sequence_id, candidate, winner["target_kbps"], video_path)
                if video_row["status"] == "failed":
                    failed.append(video_row)
                for index, previous in enumerate(validation_rows):
                    if (previous["sequence_id"] == sequence_id and
                            previous["method"] == candidate["method"] and
                            previous["target_kbps"] == winner["target_kbps"]):
                        validation_rows[index] = video_row
                        break
        write_summary({"selection": "fully_rate_matched_then_seeded_hash_independent_of_quality",
                       "seed": seed,
                       "max_scenes": config["experiment"].get("max_comparison_video_scenes", 3),
                       "eligible_scene_count": len(eligible), "sequence_ids": comparison_sequences,
                       "methods": comparison_methods,
                       "target_bitrates_kbps": rates},
                      output / "comparison_video_selection.json")
    comparisons = compare_validation(validation_rows, config["search"]["objective"],
                                     config["search"]["pair_tolerance_percent"])
    write_csv(validation_rows, output / "validation_results.csv")
    write_csv(comparisons, output / "comparisons.csv")
    write_csv(validation_patches, output / "validation_patch_results.csv")
    comparison_video_records = _write_comparison_manifest(
        output, comparison_sequences, rates, validation_rows, comparison_methods
    ) if comparison_sequences else []
    from src.reports.search_plots import plot_search_results
    plot_search_results(all_rows, validation_rows, comparisons, validation_patches, output / "plots", candidates)
    groups = {(row["sequence_id"], row["target_kbps"], row["method"]) for row in comparisons}
    proven = sum(all(row["outperforms"] and row["comparable"] for row in comparisons
                     if (row["sequence_id"], row["target_kbps"], row["method"]) == group)
                 and len([row for row in comparisons if (row["sequence_id"], row["target_kbps"], row["method"]) == group]) == 2
                 for group in groups)
    run_complete = not failed and len(winners) == expected_winners
    summary = {"status": "complete" if run_complete else "complete_with_failures", "run_signature": signature,
               "calibration_trials": len(all_rows), "matched_calibration_trials": sum(row["status"] == "matched" for row in all_rows),
               "selected_configs": len(winners), "expected_selected_configs": expected_winners,
               "unmatched_method_rate_groups": expected_winners - len(winners),
               "validation_trials": len(validation_rows), "failed_trials": len(failed),
               "objective": config["search"]["objective"], "partition": config["search"]["partition"],
               "configuration_selection_scope": "calibration_ROI_only",
               "comparison_video_scenes": comparison_sequences,
               "comparison_video_methods": comparison_methods,
               "comparison_video_files": sum(bool(row.get("video_path")) for row in comparison_video_records),
               "artifact_retention": config["experiment"].get("artifact_retention", "all"),
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
    from src.reports.formal_search import analyze_search_output
    analyze_search_output(output)
    if summary["status"] == "complete" and config["experiment"].get("artifact_retention") == "plot_data_only":
        _prune_intermediate_artifacts(output, candidates)
    return summary
