import time

import numpy as np

from src.bistar.reconstruction.interpolation import InterpolationReconstructor
from src.bistar.reconstruction.previous_frame import PreviousFrameReconstructor
from src.bistar.selector import EdgeSelector, FullSelector, GazeEdgeSelector, GazeSelector, GazeTemporalSelector, RandomSelector, StructuralSelector, TemporalSSIMSelector
from src.bistar.transport.packetizer import Packetizer
from src.bistar.transport.serializer import deserialize_packet
from src.common.config import load_bistar_config, load_yaml, resolve_path, validate_experiment_config
from src.datasets.frame_reader import read_rgb_frames, write_rgb_video
from src.datasets.manifest import read_manifest
from src.experiments.bistar_runner import encode_sparse_clip, evaluate_videomae_clip
from src.experiments.atlas_runner import evaluate_atlas_sequence, load_atlas_config
from src.experiments.tracked_window_runner import (
    evaluate_tracked_window_sequence,
    load_tracked_window_config,
    resolve_experiment_focus,
)
from src.bistar.reconstruction.videomae_adapter import VideoMAEAdapter
from src.common.progress import ProgressBar
from src.metrics.image_quality import create_lpips_model, frame_metrics, roi_metrics
from src.metrics.temporal_quality import residual_difference
from .runner import PROJECT_ROOT, prepare_run, write_csv, write_summary
from .video_budget import VideoSaveLimiter


SELECTORS = {"random": RandomSelector, "gaze_only": GazeSelector, "temporal_only": TemporalSSIMSelector,
             "edge_only": EdgeSelector, "structural_only": StructuralSelector, "gaze_temporal": GazeTemporalSelector,
             "gaze_edge": GazeEdgeSelector, "bistar_full": FullSelector}


def run_ablation(config_path):
    cfg = validate_experiment_config(load_yaml(config_path), "ablation"); output = prepare_run(cfg, resolve_path(cfg["experiment"]["output_dir"], PROJECT_ROOT))
    ds = load_yaml(resolve_path(cfg["dataset"]["config"], PROJECT_ROOT))["dataset"]
    sequences = read_manifest(resolve_path(ds["manifest"], PROJECT_ROOT), cfg["dataset"].get("split", "test")); results = []; frame_results = []
    codec = load_bistar_config(resolve_path(cfg["bistar"]["config"], PROJECT_ROOT))["codec"]
    patch_size = int(codec["patch_size"])
    payload_encoding = codec["payload_encoding"]
    reconstruction_type = codec["reconstruction"]["type"]
    lpips_model = create_lpips_model("cpu")
    save_videomae = bool(cfg["experiment"].get("save_videomae_videos", False))
    video_limiter = VideoSaveLimiter(cfg["experiment"].get("max_videomae_videos"))
    transports = cfg["ablation"].get("transports", ["jpeg_patch"])
    atlas_config = load_atlas_config(resolve_path(cfg["bistar_atlas"]["config"], PROJECT_ROOT)) if "h264_atlas" in transports else None
    tracked_config = load_tracked_window_config(resolve_path(
        cfg["bistar_tracked_window"]["config"], PROJECT_ROOT)) if "tracked_window" in transports else None
    tracked_sizes = cfg["ablation"].get("window_sizes", [64, 96, 128]) if tracked_config else []
    progress = ProgressBar(
        len(sequences) * len(cfg["ablation"]["variants"]) * len(cfg["ablation"]["visible_patch_counts"])
        * (1 + int(atlas_config is not None)) + len(sequences) * len(tracked_sizes), "Ablation")
    model_adapter = None
    if save_videomae or atlas_config is not None or tracked_config is not None:
        model_adapter = VideoMAEAdapter(load_yaml(PROJECT_ROOT / "configs/models/videomae.yaml"))
        model_adapter.load_model()
    for seq_num, sequence in enumerate(sequences):
        frames = read_rgb_frames(resolve_path(sequence["video_path"], PROJECT_ROOT))
        for variant in cfg["ablation"]["variants"]:
            for budget in cfg["ablation"]["visible_patch_counts"]:
                selector = SELECTORS[variant](patch_size=patch_size); packetizer = Packetizer(**payload_encoding, patch_size=patch_size)
                reconstructor = PreviousFrameReconstructor(patch_size) if reconstruction_type == "previous_frame" else InterpolationReconstructor(patch_size)
                previous_source = previous_reconstructed = None; total_bytes = payload_bytes = metadata_bytes = 0; metrics = []; roi_values = []; latencies = []; temporal_values = []
                for frame_id, frame in enumerate(frames):
                    selected = selector.select(frame, previous_source, (frame.shape[1]/2, frame.shape[0]/2, 1), budget)
                    packet, wire, sizes = packetizer.packetize(frame, selected["selected_indices"], seq_num, frame_id, time.time_ns())
                    received = deserialize_packet(wire); sparse = packetizer.sparse_frame(received); started = time.perf_counter()
                    reconstructed = reconstructor.reconstruct(sparse, selected["visible_mask"], previous_reconstructed); latencies.append((time.perf_counter()-started)*1000)
                    values = frame_metrics(frame, reconstructed, lpips_model); metrics.append(values)
                    roi = np.zeros(frame.shape[:2], dtype=bool); h, w = frame.shape[:2]; roi[h//4:3*h//4, w//4:3*w//4] = True
                    roi_value = roi_metrics(frame, reconstructed, roi, lpips_model); roi_values.append(roi_value)
                    temporal = None if previous_source is None else residual_difference(previous_source, frame, previous_reconstructed, reconstructed)
                    if temporal is not None: temporal_values.append(temporal)
                    frame_results.append({"sequence_id": sequence["sequence_id"], "transport": "jpeg_patch",
                                          "selector": variant, "visible_patch_count": budget,
                                          "frame_id": frame_id, "payload_bytes": sizes["payload_bytes"], "metadata_bytes": sizes["metadata_bytes"],
                                          "total_bytes": sizes["total_bytes"], "reconstruction_ms": latencies[-1], "temporal_error": temporal, **values, **roi_value})
                    previous_source, previous_reconstructed = frame, reconstructed
                    total_bytes += sizes["total_bytes"]; payload_bytes += sizes["payload_bytes"]; metadata_bytes += sizes["metadata_bytes"]
                duration = len(frames) / float(ds["preprocessing"]["target_fps"])
                videomae_video = None
                if save_videomae and video_limiter.acquire():
                    reconstructed_frames = []
                    video_selector = SELECTORS[variant](patch_size=patch_size)
                    clip_size = 16
                    for start in range(0, len(frames), clip_size):
                        valid = frames[start:start + clip_size]
                        valid_count = len(valid)
                        clip = valid + [valid[-1]] * (clip_size - valid_count)
                        encoded = encode_sparse_clip(
                            clip,
                            budget,
                            sequence_number=seq_num,
                            fps=float(ds["preprocessing"]["target_fps"]),
                            patch_size=patch_size,
                            payload_encoding=payload_encoding,
                            selector=video_selector,
                        )
                        evaluated = evaluate_videomae_clip(
                            clip, encoded, adapter=model_adapter, compute_metrics=False
                        )
                        reconstructed_frames.extend(evaluated["reconstructed"][:valid_count])
                    videomae_video = write_rgb_video(
                        reconstructed_frames,
                        output / "videomae_reconstructions" / sequence["sequence_id"] /
                        "{}_{}patches.mp4".format(variant, budget),
                        float(ds["preprocessing"]["target_fps"]),
                    )
                results.append({"sequence_id": sequence["sequence_id"], "transport": "jpeg_patch",
                                "selector": variant, "visible_patch_count": budget,
                                "actual_bitrate_kbps": total_bytes*8/duration/1000, "payload_bitrate_kbps": payload_bytes*8/duration/1000,
                                "metadata_bitrate_kbps": metadata_bytes*8/duration/1000, "ssim": float(np.mean([x["ssim"] for x in metrics])),
                                "lpips": float(np.mean([x["lpips"] for x in metrics])),
                                "roi_ssim": float(np.mean([x["roi_ssim"] for x in roi_values])), "roi_lpips": float(np.mean([x["roi_lpips"] for x in roi_values])),
                                "temporal_error": float(np.mean(temporal_values)) if temporal_values else 0.0,
                                "reconstruction_ms": float(np.mean(latencies)), "fps": 1000.0 / float(np.mean(latencies)),
                                "payload_encoding_mode": payload_encoding["mode"],
                                "payload_encoding_quality": payload_encoding["quality"],
                                "videomae_reconstruction_video": videomae_video["path"] if videomae_video else None,
                                "videomae_reconstruction_video_bytes": videomae_video["bytes"] if videomae_video else None})
                progress.update(message="{} {} {}".format(sequence["sequence_id"], variant, budget))
                if atlas_config is not None:
                    fps = float(ds["preprocessing"]["target_fps"])
                    atlas = evaluate_atlas_sequence(
                        frames, fps, atlas_config,
                        output / "atlas_artifacts" / sequence["sequence_id"] /
                        "{}_{}patches".format(variant, budget),
                        selector_type=variant, visible_patch_count=budget,
                        model_adapter=model_adapter, lpips_model=lpips_model, compute_metrics=True,
                        sequence_number=seq_num,
                    )
                    atlas_roi = []
                    atlas_temporal = []
                    previous_source = previous_reconstructed = None
                    for frame_id, (frame, reconstructed, values, latency) in enumerate(zip(
                            frames, atlas["reconstructed"], atlas["frame_metrics"], atlas["frame_latencies_ms"])):
                        roi = np.zeros(frame.shape[:2], dtype=bool)
                        h, w = frame.shape[:2]; roi[h//4:3*h//4, w//4:3*w//4] = True
                        roi_value = roi_metrics(frame, reconstructed, roi, lpips_model)
                        temporal = None if previous_source is None else residual_difference(
                            previous_source, frame, previous_reconstructed, reconstructed
                        )
                        if temporal is not None:
                            atlas_temporal.append(temporal)
                        atlas_roi.append(roi_value)
                        frame_results.append({
                            "sequence_id": sequence["sequence_id"], "transport": "h264_atlas",
                            "selector": variant, "visible_patch_count": budget, "frame_id": frame_id,
                            "reconstruction_ms": latency, "temporal_error": temporal,
                            **values, **roi_value,
                        })
                        previous_source, previous_reconstructed = frame, reconstructed
                    atlas_video = None
                    if save_videomae and video_limiter.acquire():
                        atlas_video = write_rgb_video(
                            atlas["reconstructed"],
                            output / "atlas_reconstructions" / sequence["sequence_id"] /
                            "{}_{}patches.mp4".format(variant, budget),
                            fps,
                        )
                    results.append({
                        "sequence_id": sequence["sequence_id"], "transport": "h264_atlas",
                        "selector": variant, "visible_patch_count": budget,
                        "actual_bitrate_kbps": atlas["actual_bitrate_kbps"],
                        "payload_bitrate_kbps": atlas["payload_bytes"]*8/duration/1000,
                        "metadata_bitrate_kbps": atlas["metadata_bytes"]*8/duration/1000,
                        "ssim": float(np.mean([x["ssim"] for x in atlas["frame_metrics"]])),
                        "lpips": float(np.mean([x["lpips"] for x in atlas["frame_metrics"]])),
                        "roi_ssim": float(np.mean([x["roi_ssim"] for x in atlas_roi])),
                        "roi_lpips": float(np.mean([x["roi_lpips"] for x in atlas_roi])),
                        "temporal_error": float(np.mean(atlas_temporal)) if atlas_temporal else 0.0,
                        "reconstruction_ms": atlas["mean_latency_ms"],
                        "p95_latency_ms": atlas["p95_latency_ms"],
                        "fps": 1000.0 / atlas["mean_latency_ms"],
                        "evaluation_mode": atlas["evaluation_mode"],
                        "videomae_reconstruction_video": atlas_video["path"] if atlas_video else None,
                        "videomae_reconstruction_video_bytes": atlas_video["bytes"] if atlas_video else None,
                    })
                    progress.update(message="{} atlas {} {}".format(sequence["sequence_id"], variant, budget))
        if tracked_config is not None:
            fps = float(ds["preprocessing"]["target_fps"])
            focus_points, focus_provenance, annotated_masks = resolve_experiment_focus(
                cfg, sequence["sequence_id"], frames, PROJECT_ROOT)
            for window_size in tracked_sizes:
                trial_config = {"codec": dict(tracked_config["codec"])}
                trial_config["codec"]["window"] = dict(tracked_config["codec"]["window"])
                trial_config["codec"]["window"].update(
                    {"width": int(window_size), "height": int(window_size)})
                evaluated = evaluate_tracked_window_sequence(
                    frames, focus_points, fps, trial_config,
                    output / "tracked_window_artifacts" / sequence["sequence_id"] /
                    "{}x{}".format(window_size, window_size), model_adapter=model_adapter,
                    lpips_model=lpips_model, compute_metrics=True, sequence_number=seq_num,
                )
                roi_values = []
                temporal_values = []
                previous_source = previous_reconstructed = None
                for frame_id, (frame, reconstructed, values) in enumerate(zip(
                        frames, evaluated["reconstructed"], evaluated["frame_metrics"])):
                    if annotated_masks is not None:
                        roi = annotated_masks[frame_id]
                    else:
                        roi = np.zeros(frame.shape[:2], dtype=bool)
                        h, w = frame.shape[:2]; roi[h//4:3*h//4, w//4:3*w//4] = True
                    roi_value = roi_metrics(frame, reconstructed, roi, lpips_model)
                    temporal = None if previous_source is None else residual_difference(
                        previous_source, frame, previous_reconstructed, reconstructed)
                    if temporal is not None:
                        temporal_values.append(temporal)
                    roi_values.append(roi_value)
                    frame_results.append({
                        "sequence_id": sequence["sequence_id"], "transport": "tracked_window",
                        "selector": "visual_focus", "window_size": window_size,
                        "frame_id": frame_id, "temporal_error": temporal,
                        **values, **roi_value,
                    })
                    previous_source, previous_reconstructed = frame, reconstructed
                video = None
                if save_videomae and video_limiter.acquire():
                    video = write_rgb_video(
                        evaluated["reconstructed"], output / "tracked_window_reconstructions" /
                        sequence["sequence_id"] / "{}x{}.mp4".format(window_size, window_size), fps)
                duration = len(frames) / fps
                results.append({
                    "sequence_id": sequence["sequence_id"], "transport": "tracked_window",
                    "selector": "visual_focus", "window_size": window_size,
                    "encoded_area_fraction": window_size * window_size /
                    (frames[0].shape[0] * frames[0].shape[1]),
                    "actual_bitrate_kbps": evaluated["actual_bitrate_kbps"],
                    "payload_bitrate_kbps": evaluated["payload_bytes"] * 8 / duration / 1000,
                    "metadata_bitrate_kbps": evaluated["metadata_bytes"] * 8 / duration / 1000,
                    "ssim": float(np.mean([x["ssim"] for x in evaluated["frame_metrics"]])),
                    "lpips": float(np.mean([x["lpips"] for x in evaluated["frame_metrics"]])),
                    "roi_ssim": float(np.mean([x["roi_ssim"] for x in roi_values])),
                    "roi_lpips": float(np.mean([x["roi_lpips"] for x in roi_values])),
                    "temporal_error": float(np.mean(temporal_values)) if temporal_values else 0.0,
                    "reconstruction_ms": evaluated["mean_latency_ms"],
                    "p95_latency_ms": evaluated["p95_latency_ms"],
                    "fps": 1000.0 / evaluated["mean_latency_ms"],
                    "focus_provenance": ",".join(focus_provenance),
                    "evaluation_mode": evaluated["evaluation_mode"],
                    "videomae_reconstruction_video": video["path"] if video else None,
                    "videomae_reconstruction_video_bytes": video["bytes"] if video else None,
                })
                progress.update(message="{} tracked_window {}".format(
                    sequence["sequence_id"], window_size))
    progress.close()
    write_csv(frame_results, output / "frame_results.csv"); write_csv(results, output / "sequence_results.csv")
    from src.reports.tables import write_ablation_table
    write_ablation_table(results, output / "tables/ablation.csv")
    write_csv(results, output / "metrics.csv"); write_csv(results, output / "ablation.csv"); write_summary({"status": "complete", "results": results}, output / "summary.json"); return results
