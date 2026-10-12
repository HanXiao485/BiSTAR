# BiSTAR Benchmark

Reproducible offline benchmarking for H.264, H.265, and sparse-patch BiSTAR reconstruction.

## Setup

```bash
./scripts/bootstrap.sh
conda activate /home/xiaohan/BiSTAR-Benchmark/.conda/bistar-full
python scripts/check_environment.py
python scripts/run_smoke_test.py --config configs/experiments/smoke_test.yaml
```

The smoke test creates a deterministic synthetic clip and evaluates all three methods. Full datasets are declared by CSV manifests; experiments never discover inputs recursively.

The checked configuration uses the official VideoMAE ViT-B SSV2 pretraining checkpoint for 16 frames and tubelet size 2. Its expected SHA-256 is stored in `configs/models/videomae.yaml`. Hidden pixels are zeroed before inference, and predicted patches are denormalized only with statistics available from transmitted pixels.

The validated local environment is `.conda/bistar-full` (Python 3.10.15, PyTorch 2.4.0, torchvision 0.19.0). The current host has no visible NVIDIA device, so model acceptance runs on CPU; setting the model runtime device to `cuda:0` makes the environment check require CUDA.

## Commands

```bash
python scripts/build_manifest.py --input datasets/raw --output datasets/manifests/bistar_eval.csv
python scripts/run_rd.py --config configs/experiments/rd_sweep.yaml
python scripts/run_matched_bitrate.py --config configs/experiments/matched_bitrate.yaml
python scripts/run_ablation.py --config configs/experiments/ablation.yaml
pytest
```

Every run saves its resolved configuration, environment and Git state, command logs, frame/sequence measurements, summary, and plots under its output directory.
