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
