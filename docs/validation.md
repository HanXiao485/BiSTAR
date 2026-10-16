# Validation record

Validated on 2026-10-12 with the Conda prefix `/home/xiaohan/anaconda3/envs/bistar`.

- Python 3.10.15
- PyTorch 2.4.0+cu121; torchvision 0.19.0+cu121; CUDA unavailable on this host, CPU inference used
- VideoMAE upstream commit `14ef8d856287c94ef1f985fe30f958eb4ec2c55d`
- VideoMAE checkpoint SHA-256 `decba3e0264ae77030eadf097403e13051da35e19297d59bb0abf721cf4fa0c2`
- FFmpeg 4.4.2 with libx264 and libx265
- GStreamer 1.20.3 with x264enc, x265enc, H.264/H.265 parsers and RTP payloaders

Acceptance commands:

```bash
PYTHONNOUSERSITE=1 /home/xiaohan/anaconda3/envs/bistar/bin/python scripts/check_environment.py
PYTHONNOUSERSITE=1 /home/xiaohan/anaconda3/envs/bistar/bin/python -m pytest tests/unit
PYTHONNOUSERSITE=1 /home/xiaohan/anaconda3/envs/bistar/bin/python -m pytest tests/integration
PYTHONNOUSERSITE=1 /home/xiaohan/anaconda3/envs/bistar/bin/python -m pytest tests/regression
PYTHONNOUSERSITE=1 /home/xiaohan/anaconda3/envs/bistar/bin/python scripts/run_smoke_test.py --config configs/experiments/smoke_test.yaml
PYTHONNOUSERSITE=1 /home/xiaohan/anaconda3/envs/bistar/bin/python scripts/run_rd.py --config configs/experiments/rd_sweep.yaml
PYTHONNOUSERSITE=1 /home/xiaohan/anaconda3/envs/bistar/bin/python scripts/run_matched_bitrate.py --config configs/experiments/matched_bitrate.yaml
PYTHONNOUSERSITE=1 /home/xiaohan/anaconda3/envs/bistar/bin/python scripts/run_ablation.py --config configs/experiments/ablation.yaml
PYTHONNOUSERSITE=1 /home/xiaohan/anaconda3/envs/bistar/bin/python scripts/run_realtime.py --config configs/experiments/realtime.yaml
```

The synthetic acceptance dataset is deterministic, 224×224, 15 FPS. Formal experiments should replace its manifest rows with the paper dataset while keeping the same preprocessing and runner paths.

## GOP patch atlas validation

Validated on 2026-10-13 on branch `dev` with FFmpeg 4.4.2.

- `python -m pytest -q`: 42 passed, 1 skipped. The skipped checkpoint forward test requires the configured `cuda:0`, which was unavailable to the validation process.
- All seven RD, matched-bitrate, ablation, and real-time YAML files passed schema validation.
- A 16-frame segment from `datasets/raw/test.mp4` completed atlas selection, H.264 encode/decode, packet accounting, and sparse-frame restoration. It produced one GOP with 23,825 total wire bytes, including 68 metadata bytes, at 178.688 kbps for a 100 kbps encoder request.
- `scripts/run_realtime.py --config configs/experiments/realtime.yaml` completed its dependency and protocol-readiness check.

The full VideoMAE experiment runners require access to the GPU selected by `configs/models/videomae.yaml`. The transport test above deliberately stopped before neural reconstruction; the checkpoint integration test runs automatically when CUDA is visible.

The adaptive GOP controller was also checked on 48 frames from `test.mp4`. At target rates of 50, 100, and 300 kbps, the first three GOP budgets were `[81, 64, 49]`, `[144, 121, 100]`, and `[196, 169, 144]`, respectively. This verifies target-rate initialization, causal feedback from prior wire bytes, configured candidate quantization, and the 196-patch upper bound.

## Annotated ROI configuration search

Validated on 2026-10-15 on branch `dev`.

- Full regression suite: 48 passed, 1 skipped (the existing configured-CUDA checkpoint test cannot access CUDA in this process).
- Four-method end-to-end integration covers independent ROI loading, temporal holdout, candidate selection, wire accounting, plots, metadata, and resume.
- Generated a standalone annotation editor and blank 175-frame CSV template for the existing `test.mp4` manifest. Editor JavaScript passed `node --check`.
- Ran an actual VideoMAE checkpoint on CPU with four PyTorch threads for a real-video pipeline demonstration. This per-run device override did not edit the shared CUDA model configuration.
- `outputs/roi_search_demo/strict_run` evaluated eight configurations at 100 kbps, with seven matching the 1% target tolerance. Four configurations selected on 105 calibration frames were frozen and evaluated on 54 held-out frames, separated by a 16-frame gap.
- Held-out ROI SSIM: H.264 0.874184, H.265 0.866112, BiSTAR 0.273328, BiSTAR Atlas 0.929634. Actual rates were 99.7822, 100.6422, 100.4178, and 100.2067 kbps, respectively. Atlas's selected configuration used gaze-only selection, 64 fixed patches, x264 fast, yuv420p, and VideoMAE reconstruction.
- All demo ROIs are explicitly labelled `example_center_proxy`, and the summary records `proxy_ROI_only: true`. These figures validate the experiment implementation and do not establish task-ROI or teleoperation performance.

Formal search defaults to independent `val`/`test` videos and requires completed task annotations. Single-video temporal holdout is available only as an exploratory protocol. See `docs/roi_config_search.md`.
