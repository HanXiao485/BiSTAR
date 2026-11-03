---
name: bistar-block-qp-benchmark
description: Implement, extend, or validate BiSTAR block-level H.264 QP benchmarks, including ROI/background quality control, QP-map providers, encoder backends, runner integration, bitrate accounting, and compatibility checks. Use for real per-frame macroblock QP control in a full-frame stream; do not use for atlas/patch cropping or for claiming an RL-RC-DoT reproduction without its learning system.
---

# BiSTAR Block-QP Benchmark

Implement the requested feature and run proportionate validation; do not stop at a design document. Preserve unrelated and uncommitted user work, existing output, baseline behavior, and exit-code semantics. Do not push, start large formal experiments, or train PPO unless the user explicitly asks.

## Workflow

### 1. Audit the checkout

Read applicable `AGENTS.md`, the README, packaging metadata, codec adapters, configuration/types, RD and matched-bitrate runners, codec search and ROI-search paths, ROI datasets, spatial/task metrics, and existing H.264 configuration. Record the current HEAD and working-tree state plus Python, FFmpeg, and x264 versions/build options.

Use `rg` to find method registries, allowlists, dispatch sites, cache keys, result columns, plotting/reporting consumers, and tests. If an expected path moved, locate the current equivalent instead of creating a parallel framework. Produce a minimal integration checklist before editing.

### 2. Prove a backend is feasible

Choose a backend only after inspecting the installed headers/source and running a minimal feasibility probe. Prefer a small, optional libx264 bridge when its public API supports the required per-picture macroblock offsets. Consider FFmpeg ROI side data only when the current wrapper's dynamic behavior and offset conversion are verified.

Read [references/backend-truth.md](references/backend-truth.md) before selecting or implementing a backend. A global QP/CRF, qpfile, unverified `addroi`, separate patch streams, cropping, atlas packing, or blurred background is not block-level QP control.

### 3. Implement the stable interface

Adapt names to the repository, keeping the implementation local and avoiding a general plugin framework. Implement uniform, fixed-ROI, and external-map providers plus a replaceable provider contract. Validate all maps and isolate provider state per sequence and bitrate-search trial.

Read [references/interface-and-config.md](references/interface-and-config.md) for the contract, ROI composition, configuration semantics, tracing, and required failures.

### 4. Integrate evaluation

Register the method through every relevant real dispatch path: codec construction, RD, matched bitrate, search, ROI selection/evaluation, caching, metrics, results, and reporting. Reuse existing data alignment, search, output, and configuration mechanisms. Keep the ordinary FFmpeg H.264 baseline intact and make native dependencies lazy/optional.

Add a repository-root-aware `scripts/run_block_qp_benchmark.sh` that accepts `--config`, uses the active Python environment, and preserves the invoked command's exit status. Keep dependency builds explicit and separate; never replace system FFmpeg/x264.

### 5. Validate

Run focused unit/integration tests and a small smoke benchmark. The central proof is an applied-QP trace or decoded-bitstream analysis showing that swapped deltas change the intended macroblocks; size or PSNR differences alone are insufficient. Use same-backend zero-map controls to separate backend effects.

Read [references/evaluation-and-acceptance.md](references/evaluation-and-acceptance.md) and execute every applicable check. Mark unavailable evidence as unverified rather than claiming success. Do not infer quality or BD-rate gains before a held-out experiment.

### 6. Deliver

Report implementation paths, commands run, observed results, environment/backend versions, supported and unsupported capabilities, unrun checks, and blockers. Include adapters/backend, providers/configs, runner/report integration, tests, one-command smoke entry point, user-facing run instructions, and small reproducible outputs where the repository convention permits them.

Call the method `h264_block_qp` or another factual name. Use `oracle` in method/result names whenever evaluation-only ground truth reaches the encoding policy. Never label the work RL-RC-DoT unless the paper's state, reward, losses, and training procedure are actually implemented and validated.
