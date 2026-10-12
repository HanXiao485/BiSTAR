# BiSTAR Benchmark Infrastructure

**Document purpose:** Engineering implementation specification for Codex  
**Target project:** `BiSTAR-Benchmark`  
**Primary objective:** Build a reproducible experimental infrastructure for fair comparison among:

- **BiSTAR + VideoMAE reconstruction**
- **H.264 baseline via FFmpeg + libx264**
- **H.265 baseline via FFmpeg + libx265**

The infrastructure must support:

1. offline rate–distortion benchmarking;
2. BiSTAR component ablation;
3. VideoMAE mask/reconstruction adaptation;
4. bitrate accounting including metadata;
5. end-to-end latency measurement;
6. future real-time extension with GStreamer and Linux traffic control;
7. fully reproducible experiment configuration and result logging.

---

# 1. Engineering principles

The following rules are mandatory.

## 1.1 Reproducibility first

Every experiment must be reconstructable from:

- Git commit hash;
- Conda environment;
- FFmpeg build configuration;
- VideoMAE commit hash;
- checkpoint hash;
- dataset manifest;
- YAML experiment config;
- random seed.

No result may depend on manually modified parameters that are not stored in a configuration file.

---

## 1.2 Baseline fairness

H.264, H.265, and BiSTAR must:

- receive the **same source frames**;
- use the **same resolution**;
- use the **same frame rate**;
- use the **same frame interval**;
- be evaluated on exactly the **same test sequences**;
- report **actual encoded/transmitted bytes**, not theoretical bitrate;
- use the same metric implementation;
- use identical frame alignment.

For matched-bitrate experiments, actual measured bitrate difference between two compared systems should normally be within the configured tolerance.

Default:

```yaml
bitrate_match_tolerance_percent: 5.0
```

---

## 1.3 No hidden future information

For online or pseudo-online experiments, reconstruction of frame `t` must not use any frame later than the allowed inference window.

If VideoMAE needs future frames for a clip:

- the waiting time must be included in latency; or
- a causal/sliding-window mode must be implemented.

Offline quality experiments may use fixed clips, but these must be clearly tagged:

```yaml
evaluation_mode: offline_noncausal
```

Results from non-causal evaluation must not be reported as real-time performance.

---

## 1.4 Common output interface

Every codec/reconstruction method must eventually output:

```text
DecodedFrame
├── sequence_id
├── frame_id
├── capture_timestamp_ns
├── encode_start_timestamp_ns
├── encode_end_timestamp_ns
├── send_timestamp_ns
├── receive_timestamp_ns
├── decode_start_timestamp_ns
├── decode_end_timestamp_ns
├── display_ready_timestamp_ns
├── rgb_frame
├── payload_bytes
├── metadata_bytes
├── total_bytes
└── method_metadata
```

This common interface is the basis for all quality, bitrate, and latency comparisons.

---

# 2. Repository layout

Create the project with the following structure.

```text
BiSTAR-Benchmark/
├── README.md
├── environment.yml
├── pyproject.toml
├── .gitignore
├── .gitmodules
│
├── configs/
│   ├── datasets/
│   │   ├── example_dataset.yaml
│   │   └── bistar_eval.yaml
│   ├── codecs/
│   │   ├── h264.yaml
│   │   ├── h265.yaml
│   │   └── bistar.yaml
│   ├── models/
│   │   └── videomae.yaml
│   ├── metrics/
│   │   └── default.yaml
│   └── experiments/
│       ├── smoke_test.yaml
│       ├── rd_sweep.yaml
│       ├── matched_bitrate.yaml
│       ├── ablation.yaml
│       └── realtime.yaml
│
├── third_party/
│   └── VideoMAE/
│
├── src/
│   ├── common/
│   │   ├── types.py
│   │   ├── config.py
│   │   ├── logging.py
│   │   ├── timing.py
│   │   └── hashing.py
│   │
│   ├── datasets/
│   │   ├── base.py
│   │   ├── video_dataset.py
│   │   ├── frame_reader.py
│   │   └── manifest.py
│   │
│   ├── codecs/
│   │   ├── base.py
│   │   ├── ffmpeg_common.py
│   │   ├── h264.py
│   │   └── h265.py
│   │
│   ├── bistar/
│   │   ├── selector/
│   │   │   ├── base.py
│   │   │   ├── random_selector.py
│   │   │   ├── gaze_selector.py
│   │   │   ├── temporal_ssim_selector.py
│   │   │   ├── edge_selector.py
│   │   │   ├── structural_selector.py
│   │   │   └── full_selector.py
│   │   ├── transport/
│   │   │   ├── packet.py
│   │   │   ├── packetizer.py
│   │   │   ├── serializer.py
│   │   │   └── bitrate_accounting.py
│   │   └── reconstruction/
│   │       ├── base.py
│   │       ├── videomae_adapter.py
│   │       ├── mask_adapter.py
│   │       ├── previous_frame.py
│   │       └── interpolation.py
│   │
│   ├── metrics/
│   │   ├── image_quality.py
│   │   ├── temporal_quality.py
│   │   ├── bitrate.py
│   │   ├── latency.py
│   │   └── task_metrics.py
│   │
│   ├── experiments/
│   │   ├── runner.py
│   │   ├── rd_runner.py
│   │   ├── matched_bitrate_runner.py
│   │   ├── ablation_runner.py
│   │   └── realtime_runner.py
│   │
│   └── reports/
│       ├── aggregate.py
│       ├── tables.py
│       └── plots.py
│
├── realtime/
│   ├── gstreamer/
│   │   ├── sender_h264.sh
│   │   ├── sender_h265.sh
│   │   ├── receiver_h264.sh
│   │   └── receiver_h265.sh
│   └── network/
│       ├── setup_netem.sh
│       └── reset_netem.sh
│
├── scripts/
│   ├── bootstrap.sh
│   ├── check_environment.py
│   ├── build_manifest.py
│   ├── run_smoke_test.py
│   ├── run_rd.py
│   ├── run_matched_bitrate.py
│   ├── run_ablation.py
│   └── run_realtime.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── regression/
│
├── datasets/
├── checkpoints/
├── outputs/
└── docs/
```

---

# 3. Environment setup

## 3.1 Recommended OS

Primary target:

```text
Ubuntu 22.04 LTS
```

Also record:

```bash
uname -a
lsb_release -a
nvidia-smi
```

---

## 3.2 FFmpeg requirements

The environment must provide:

```bash
ffmpeg
ffprobe
libx264
libx265
```

Validation:

```bash
ffmpeg -hide_banner -encoders | grep -E 'libx264|libx265'
ffmpeg -buildconf
```

The build configuration must be saved to:

```text
outputs/environment/ffmpeg_buildconf.txt
```

---

## 3.3 VideoMAE dependency

Add VideoMAE as a Git submodule if possible:

```bash
git submodule add https://github.com/MCG-NJU/VideoMAE.git third_party/VideoMAE
git submodule update --init --recursive
```

Record its commit:

```bash
git -C third_party/VideoMAE rev-parse HEAD
```

Save to:

```text
outputs/environment/videomae_commit.txt
```

---

## 3.4 Conda environment

Create:

```yaml
name: bistar
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - numpy
  - scipy
  - pandas
  - pyyaml
  - matplotlib
  - opencv
  - ffmpeg
  - pip:
      - torch
      - torchvision
      - timm
      - einops
      - scikit-image
      - lpips
      - tqdm
      - rich
      - pydantic
      - pytest
      - tensorboard
```

Codex must pin exact working versions after the first successful installation.

The final environment file must not contain floating version dependencies such as:

```text
torch
numpy
timm
```

without versions.

---

# 4. Core data types

Create:

```text
src/common/types.py
```

Required dataclasses:

```python
@dataclass
class FrameRecord:
    sequence_id: str
    frame_id: int
    timestamp_ns: int
    rgb: np.ndarray

@dataclass
class EncodedPacket:
    sequence_id: str
    frame_id: int
    payload: bytes
    payload_bytes: int
    metadata_bytes: int
    total_bytes: int
    metadata: dict

@dataclass
class DecodedFrame:
    sequence_id: str
    frame_id: int
    rgb: np.ndarray

    capture_timestamp_ns: int | None = None
    encode_start_timestamp_ns: int | None = None
    encode_end_timestamp_ns: int | None = None
    send_timestamp_ns: int | None = None
    receive_timestamp_ns: int | None = None
    decode_start_timestamp_ns: int | None = None
    decode_end_timestamp_ns: int | None = None
    display_ready_timestamp_ns: int | None = None

    payload_bytes: int = 0
    metadata_bytes: int = 0
    total_bytes: int = 0

    method_metadata: dict = field(default_factory=dict)
```

All methods must use these common records.

---

# 5. Dataset interface

## 5.1 Dataset manifest

Every dataset must be converted to a manifest.

Example:

```csv
sequence_id,video_path,fps,width,height,num_frames,split
scene01,datasets/raw/scene01.mp4,15,224,224,900,test
scene02,datasets/raw/scene02.mp4,15,224,224,1200,test
scene03,datasets/raw/scene03.mp4,15,224,224,1050,val
```

No experiment may discover videos using unrestricted recursive directory scanning at runtime.

The manifest is the source of truth.

---

## 5.2 Dataset YAML schema

```yaml
dataset:
  name: bistar_eval
  manifest: datasets/manifests/bistar_eval.csv

  preprocessing:
    target_width: 224
    target_height: 224
    target_fps: 15
    interpolation: bilinear
    color_space: rgb24

  split:
    train: train
    val: val
    test: test
```

---

# 6. Common codec abstraction

Create:

```text
src/codecs/base.py
```

Interface:

```python
class CodecAdapter(ABC):

    @abstractmethod
    def encode_sequence(
        self,
        input_video: Path,
        output_dir: Path,
        config: dict,
    ) -> dict:
        pass

    @abstractmethod
    def decode_sequence(
        self,
        bitstream_path: Path,
        output_dir: Path,
        config: dict,
    ) -> dict:
        pass

    @abstractmethod
    def probe(self, output_path: Path) -> dict:
        pass
```

Expected `encode_sequence()` return structure:

```python
{
    "bitstream_path": "...",
    "num_frames": 100,
    "duration_sec": 6.6667,
    "payload_bytes": 83000,
    "metadata_bytes": 0,
    "total_bytes": 83000,
    "actual_bitrate_kbps": 99.6,
    "encode_time_sec": 1.23,
    "ffmpeg_command": "..."
}
```

FFmpeg baselines must record the exact command string.

---

# 7. H.264 baseline

## 7.1 Default config

`configs/codecs/h264.yaml`

```yaml
codec:
  name: h264
  implementation: ffmpeg_libx264

  pixel_format: yuv420p

  encoder:
    codec_name: libx264
    preset: medium

    mode: bitrate

    bitrate_kbps: 100
    maxrate_kbps: null
    bufsize_kbps: null

    crf: null

    gop_size: 30
    b_frames: 0

    tune: null

  output:
    container: raw
    extension: h264
```

---

## 7.2 Canonical bitrate-mode command

Codex should construct, not hard-code, the final command.

Reference form:

```bash
ffmpeg -y \
  -i INPUT \
  -an \
  -vf "fps=15,scale=224:224,format=yuv420p" \
  -c:v libx264 \
  -preset medium \
  -b:v 100k \
  -g 30 \
  -bf 0 \
  -f h264 \
  OUTPUT.h264
```

---

## 7.3 CRF sweep

Required CRF values:

```yaml
crf_values:
  - 18
  - 21
  - 24
  - 27
  - 30
  - 33
  - 36
  - 39
```

Optional extended sweep:

```yaml
crf_values:
  - 15
  - 18
  - 21
  - 24
  - 27
  - 30
  - 33
  - 36
  - 39
  - 42
```

---

# 8. H.265 baseline

## 8.1 Default config

`configs/codecs/h265.yaml`

```yaml
codec:
  name: h265
  implementation: ffmpeg_libx265

  pixel_format: yuv420p

  encoder:
    codec_name: libx265
    preset: medium

    mode: bitrate

    bitrate_kbps: 100
    maxrate_kbps: null
    bufsize_kbps: null

    crf: null

    gop_size: 30
    b_frames: 0

  output:
    container: raw
    extension: hevc
```

---

## 8.2 Canonical command

```bash
ffmpeg -y \
  -i INPUT \
  -an \
  -vf "fps=15,scale=224:224,format=yuv420p" \
  -c:v libx265 \
  -preset medium \
  -b:v 100k \
  -g 30 \
  -bf 0 \
  -f hevc \
  OUTPUT.hevc
```

---

# 9. FFmpeg parameter sweep

A dedicated sweep engine is required.

Create:

```text
src/experiments/rd_runner.py
```

## 9.1 Required sweep dimensions

Phase 1 must support:

```yaml
sweep:
  codecs:
    - h264
    - h265

  bitrate_kbps:
    - 50
    - 75
    - 100
    - 150
    - 200
    - 300

  presets:
    - medium

  gop_size:
    - 30

  b_frames:
    - 0
```

Optional expanded engineering sweep:

```yaml
presets:
  - ultrafast
  - veryfast
  - medium
  - slow
```

Do not mix results from different presets into one primary RD curve unless the comparison explicitly aims to study encoder presets.

---

## 9.2 Actual bitrate calculation

Do not rely only on FFmpeg's reported bitrate.

Use:

```math
R = \frac{8B}{T}
```

where:

- `B`: output raw bitstream byte size;
- `T`: duration corresponding to decoded frames.

Implementation:

```python
bitrate_kbps = total_bytes * 8.0 / duration_sec / 1000.0
```

---

## 9.3 Matched bitrate search

Implement automatic bitrate calibration.

Input:

```python
target_kbps = 100
tolerance = 0.05
```

Required search strategy:

1. encode with initial requested bitrate;
2. measure actual bitrate;
3. if outside tolerance, update requested bitrate;
4. iterate;
5. stop at max iteration limit.

Default:

```yaml
matched_bitrate:
  tolerance_percent: 5.0
  max_iterations: 8
```

Preferred update:

```python
new_request = old_request * target_kbps / actual_kbps
```

Clamp the update to avoid unstable jumps.

Required output:

```json
{
  "target_kbps": 100,
  "requested_kbps": 96,
  "actual_kbps": 101.7,
  "error_percent": 1.7,
  "iterations": 3
}
```

---

# 10. VideoMAE integration

## 10.1 Third-party source

Use:

```text
third_party/VideoMAE/
```

Do not modify upstream files unless absolutely required.

Prefer adapter/wrapper implementation inside:

```text
src/bistar/reconstruction/
```

If an upstream patch is required:

1. store it as a `.patch`;
2. document the modified file;
3. explain why wrapping was insufficient.

---

## 10.2 VideoMAE adapter

Create:

```text
src/bistar/reconstruction/videomae_adapter.py
```

Required interface:

```python
class VideoMAEAdapter:

    def __init__(self, config: dict):
        ...

    def load_model(self):
        ...

    def preprocess_clip(self, frames):
        ...

    def build_mask(self, visible_patch_mask):
        ...

    def reconstruct_clip(
        self,
        frames,
        visible_patch_mask,
    ):
        ...

    def reconstruct_frame(
        self,
        history_frames,
        current_sparse_frame,
        current_visible_mask,
    ):
        ...
```

---

## 10.3 Model config schema

`configs/models/videomae.yaml`

```yaml
model:
  name: videomae
  source: third_party/VideoMAE

  architecture:
    model_name: pretrain_videomae_base_patch16_224
    image_size: 224
    patch_size: 16
    tubelet_size: 2

  temporal:
    num_frames: 16
    sampling_rate: 1

  checkpoint:
    path: checkpoints/videomae_base.pth
    sha256: null

  precision:
    dtype: fp16
    amp: true

  reconstruction:
    mode: masked_reconstruction
    output_frame_policy: center
```

All fields must be derived from the actually used checkpoint/model definition.

---

# 11. BiSTAR mask adaptation

The original VideoMAE uses spatiotemporal masking. BiSTAR selects visible image regions.

A dedicated mask adapter is mandatory.

Create:

```text
src/bistar/reconstruction/mask_adapter.py
```

---

## 11.1 Input representation

BiSTAR selector output:

```python
visible_mask_2d: np.ndarray
shape = [H_patch, W_patch]
dtype = bool
```

For 224×224 and patch size 16:

```text
H_patch = 14
W_patch = 14
```

This gives:

```text
196 spatial patches per frame
```

---

## 11.2 Temporal expansion

VideoMAE uses tubelets.

The adapter must generate a mask compatible with:

```text
T_token × H_patch × W_patch
```

where:

```python
T_token = num_frames // tubelet_size
```

Example:

```text
num_frames = 16
tubelet_size = 2

T_token = 8
```

Final Boolean mask size:

```text
8 × 14 × 14 = 1568 tokens
```

The exact flattening order must match the upstream VideoMAE implementation.

This must be verified by a unit test.

---

## 11.3 Mask semantics

Upstream VideoMAE uses:

```python
~mask
```

for visible tokens.

Therefore:

```text
mask == True  -> hidden token
mask == False -> visible token
```

This convention must be explicitly documented.

The BiSTAR selector should internally use:

```text
visible == True
```

The mask adapter must perform the inversion.

---

# 12. BiSTAR selector interfaces

Create:

```text
src/bistar/selector/base.py
```

Interface:

```python
class PatchSelector(ABC):

    @abstractmethod
    def select(
        self,
        current_frame,
        previous_frame,
        gaze_point=None,
        budget=None,
    ) -> dict:
        pass
```

Output:

```python
{
    "visible_mask": np.ndarray,
    "selected_indices": list[int],
    "scores": np.ndarray | None,
    "metadata": {}
}
```

---

# 13. Required selector implementations

## 13.1 Random selector

Purpose:

- random-mask ablation;
- matched patch-count control.

Required parameters:

```yaml
selector:
  type: random
  num_visible_patches: 40
  seed: 0
```

---

## 13.2 Gaze-only

Required parameters:

```yaml
selector:
  type: gaze
  radius_pixels: 48
  fallback: center
```

Input:

```text
gaze_x
gaze_y
gaze_confidence
```

---

## 13.3 Temporal-SSIM-only

Required parameters:

```yaml
selector:
  type: temporal_ssim
  threshold: 0.15
  top_k: null
```

If `top_k` is provided, use fixed-budget mode.

---

## 13.4 Edge-only

Required parameters:

```yaml
selector:
  type: edge
  operator: canny
  threshold_low: 50
  threshold_high: 150
  score_aggregation: mean
```

---

## 13.5 Structural-only

Defined as:

```text
Temporal SSIM OR Edge
```

or score-level fusion if configured.

---

## 13.6 BiSTAR full

Initial implementation:

```text
Gaze union Structural
```

Optional Phase 2 extension:

```text
Budget-constrained weighted selection
```

Do not implement learned weights in Phase 1.

---

# 14. BiSTAR transport layer

The experiment must measure **real serialized data size**.

Do not estimate:

```text
number_of_patches × patch_pixels
```

as the final communication cost.

---

## 14.1 Packet schema

Create:

```text
src/bistar/transport/packet.py
```

Suggested logical packet:

```python
@dataclass
class BiSTARPacket:
    version: int
    sequence_id: int
    frame_id: int
    timestamp_ns: int

    image_width: int
    image_height: int

    patch_size: int

    selected_indices: list[int]

    payload_codec: str
    payload: bytes
```

---

## 14.2 Metadata accounting

Metadata must include at least:

```text
packet header
frame ID
timestamp
patch index information
payload codec identifier
payload size
```

Report separately:

```text
payload_bytes
metadata_bytes
total_bytes
```

Paper plots may primarily use:

```text
total_bytes
```

---

## 14.3 Patch payload compression

Phase 1 supported modes:

```yaml
payload_encoding:
  mode: png
```

and:

```yaml
payload_encoding:
  mode: jpeg
  quality: 90
```

Later optional:

```text
JPEG XL
AVIF
custom packed raw tensor
```

Do not introduce additional codecs before the baseline system works.

---

# 15. Reconstruction baselines

To determine whether VideoMAE itself contributes value, implement:

```text
previous_frame
spatial_interpolation
videomae
```

Required reconstruction modes:

```yaml
reconstruction:
  type: previous_frame
```

```yaml
reconstruction:
  type: interpolation
```

```yaml
reconstruction:
  type: videomae
```

---

# 16. Unified evaluation metrics

## 16.1 Image quality

Required:

```text
PSNR
SSIM
MS-SSIM
LPIPS
```

Metrics must be computed:

```text
per frame
per sequence
global aggregate
```

---

## 16.2 ROI metrics

Support:

```text
ROI-PSNR
ROI-SSIM
ROI-LPIPS
```

ROI must be independently defined.

Do not use the BiSTAR selected patch set itself as the only ROI ground truth.

---

## 16.3 Temporal quality

Implement at least one temporal stability measurement.

Suggested Phase 1 metric:

```text
frame-to-frame reconstruction residual difference
```

Optional Phase 2:

```text
optical-flow-warped temporal error
```

---

## 16.4 Communication metrics

Required:

```text
mean bitrate
peak 1-second bitrate
P95 1-second bitrate
payload bitrate
metadata bitrate
total bitrate
```

---

## 16.5 Runtime metrics

Required:

```text
encode time per frame
decode/reconstruction time per frame
mean latency
P50 latency
P95 latency
P99 latency
FPS
GPU memory usage
```

---

# 17. Experiment config schema

All experiment configs use a shared schema.

Example:

```yaml
experiment:
  name: rd_sweep_v1
  seed: 0
  output_dir: outputs/rd_sweep_v1

dataset:
  config: configs/datasets/bistar_eval.yaml
  split: test

methods:
  - name: h264
    config: configs/codecs/h264.yaml

  - name: h265
    config: configs/codecs/h265.yaml

  - name: bistar_videomae
    config: configs/codecs/bistar.yaml

metrics:
  config: configs/metrics/default.yaml

runtime:
  device: cuda:0
  num_workers: 4

logging:
  save_frames: false
  save_bitstreams: true
  save_per_frame_metrics: true
```

---

# 18. RD experiment

Create:

```text
configs/experiments/rd_sweep.yaml
```

Example:

```yaml
experiment:
  name: rd_sweep

dataset:
  split: test

rd:
  target_bitrates_kbps:
    - 50
    - 75
    - 100
    - 150
    - 200

  bitrate_match_tolerance_percent: 5.0

methods:
  h264:
    enabled: true

  h265:
    enabled: true

  bistar:
    enabled: true

quality_metrics:
  - psnr
  - ssim
  - ms_ssim
  - lpips
```

Outputs:

```text
outputs/rd_sweep/
├── config_resolved.yaml
├── environment.json
├── sequence_results.csv
├── frame_results.csv
├── rd_points.csv
├── summary.json
├── plots/
│   ├── bitrate_vs_psnr.png
│   ├── bitrate_vs_ssim.png
│   └── bitrate_vs_lpips.png
└── logs/
```

---

# 19. Ablation experiment

Required variants:

```text
random
gaze_only
temporal_only
edge_only
structural_only
gaze_temporal
gaze_edge
bistar_full
```

All must support fixed-budget comparison.

Example:

```yaml
ablation:
  budget_mode: visible_patch_count

  visible_patch_counts:
    - 20
    - 40
    - 60

  variants:
    - random
    - gaze_only
    - temporal_only
    - edge_only
    - structural_only
    - gaze_temporal
    - gaze_edge
    - bistar_full
```

Alternative:

```yaml
budget_mode: encoded_bytes
target_kbps: 100
```

Fixed encoded-byte budget is preferable for final paper experiments.

---

# 20. Automatic reporting

Every run must automatically produce:

```text
config_resolved.yaml
git_state.json
environment.json
metrics.csv
summary.json
stdout.log
stderr.log
```

`git_state.json`:

```json
{
  "project_commit": "...",
  "project_dirty": false,
  "videomae_commit": "..."
}
```

`environment.json`:

```json
{
  "python": "...",
  "torch": "...",
  "cuda": "...",
  "gpu": "...",
  "ffmpeg_version": "...",
  "ffmpeg_buildconf_sha256": "...",
  "libx264_available": true,
  "libx265_available": true
}
```

---

# 21. Automated tests

Use `pytest`.

---

## 21.1 Unit tests

### Config parsing

```text
test_config_loading.py
```

Acceptance:

- all required fields validated;
- unknown critical fields rejected;
- missing required codec parameters rejected.

---

### Mask shape

```text
test_videomae_mask_adapter.py
```

Acceptance example:

Input:

```text
16 frames
224×224
patch size 16
tubelet size 2
```

Expected mask:

```text
1568 tokens
```

---

### Mask semantics

Verify:

```text
BiSTAR visible=True
```

maps to:

```text
VideoMAE mask=False
```

---

### Bitrate accounting

Create a known 10,000-byte payload over 1 second.

Expected:

```text
80 kbps
```

Tolerance:

```text
< 1e-6
```

---

### Frame alignment

Encoded and decoded frame IDs must remain aligned.

No silent frame dropping is allowed in offline experiments.

---

## 21.2 FFmpeg integration tests

Input:

```text
5-second synthetic video
224×224
15 FPS
```

Test:

```text
H.264 encode
H.264 decode
H.265 encode
H.265 decode
```

Acceptance:

- return code == 0;
- output exists;
- decoded frame count matches source frame count;
- width == 224;
- height == 224;
- FPS within tolerance;
- output byte size > 0.

---

## 21.3 VideoMAE smoke test

Test clip:

```text
16 frames
224×224
```

Mask:

```text
90% masked
```

Acceptance:

- model loads;
- no NaN;
- output shape is valid;
- reconstructed RGB range valid after post-processing;
- forward pass succeeds on GPU.

---

## 21.4 End-to-end smoke test

Run:

```text
one sequence
100 frames
H.264
H.265
BiSTAR
```

Acceptance:

- all three methods complete;
- frame results are generated;
- bitrate is non-zero;
- SSIM can be computed;
- output directory follows required schema.

---

# 22. Acceptance criteria by phase

# Phase 0 — Environment

Must pass:

```bash
python scripts/check_environment.py
```

Requirements:

- FFmpeg available;
- libx264 available;
- libx265 available;
- CUDA visible when configured;
- VideoMAE imports successfully;
- checkpoint exists;
- all Python dependencies available.

---

# Phase 1 — Baseline codecs

Deliverables:

```text
H264Adapter
H265Adapter
FFmpeg command generator
actual bitrate calculator
decode pipeline
RD sweep
```

Acceptance:

1. H.264/H.265 successfully process all smoke-test videos.
2. Actual bitrate measured from output bytes.
3. Per-frame reconstruction metrics generated.
4. FFmpeg commands stored.
5. Re-running same config yields identical bitstream size unless encoder behavior is inherently nondeterministic.

---

# Phase 2 — VideoMAE

Deliverables:

```text
VideoMAEAdapter
MaskAdapter
clip reconstruction
checkpoint loader
frame extraction
```

Acceptance:

1. 16-frame clip reconstructs successfully.
2. BiSTAR mask can be converted correctly.
3. Output frame IDs correspond to input frames.
4. GPU memory usage recorded.
5. Inference latency recorded.

---

# Phase 3 — BiSTAR

Deliverables:

```text
selectors
packetization
metadata accounting
reconstruction pipeline
```

Acceptance:

1. selector outputs valid patch indices;
2. packet serialization round-trip succeeds;
3. actual packet byte count measured;
4. VideoMAE reconstruction uses received sparse data;
5. no inaccessible original pixels are accidentally passed into the decoder.

The fifth requirement is critical.

A unit/integration test must explicitly ensure that the reconstruction module cannot access hidden ground-truth patches.

---

# Phase 4 — Fair comparison

Acceptance:

At every target bitrate:

```text
|actual_rate - target_rate| / target_rate <= tolerance
```

where possible.

If a method cannot meet the bitrate, record:

```text
status = bitrate_unreachable
```

Do not silently remove the point.

---

# Phase 5 — Ablation

Acceptance:

For fixed-patch-count experiments:

```text
selected patch count identical across compared selectors
```

For fixed-bitrate experiments:

```text
actual total bitrate within configured tolerance
```

---

# Phase 6 — Real-time extension

After offline experiments pass.

Use:

```text
GStreamer
RTP/UDP
Linux tc/netem
```

Do not implement real-time infrastructure before Phase 1–5 are stable.

---

# 23. Future GStreamer interface

The real-time layer should preserve the common method abstraction.

Required APIs:

```python
push_frame(frame)
push_gaze(gaze)
poll_packet()
push_packet(packet)
poll_decoded_frame()
```

H.264/H.265 may use GStreamer internally.

BiSTAR uses its own packetizer.

All methods must expose:

```text
capture_timestamp
send_timestamp
receive_timestamp
display_ready_timestamp
```

---

# 24. Linux network emulation

Future config:

```yaml
network:
  enabled: true

  bandwidth_kbps: 200
  delay_ms: 100
  jitter_ms: 20
  loss_percent: 1.0
```

Implementation:

```text
tc qdisc
netem
tbf or HTB
```

The script must also provide:

```bash
realtime/network/reset_netem.sh
```

to restore the interface.

Never leave network shaping enabled after tests.

---

# 25. Result tables required for paper revision

The infrastructure should eventually generate the raw data required for:

## Table A — Encoder configuration

```text
Method
Resolution
FPS
Preset
GOP
B-frames
Rate-control
Target bitrate
Actual bitrate
```

---

## Table B — Rate–distortion

```text
Method
Bitrate
PSNR
SSIM
MS-SSIM
LPIPS
```

---

## Table C — Ablation

```text
Selector
Actual bitrate
SSIM
LPIPS
ROI-SSIM
Temporal metric
```

---

## Table D — Runtime

```text
Method
Encode ms
Transmit ms
Decode/reconstruct ms
E2E P50
E2E P95
FPS
GPU memory
```

---

# 26. Implementation order for Codex

Codex must follow this order.

## Step 1

Create repository skeleton.

Do not implement model logic.

---

## Step 2

Implement config parser and common data types.

Run unit tests.

---

## Step 3

Implement FFmpeg H.264/H.265 adapters.

Run integration tests.

---

## Step 4

Implement unified metric engine.

Generate first RD curve using only H.264/H.265.

Do not proceed until this is correct.

---

## Step 5

Integrate VideoMAE as an unmodified third-party dependency.

Verify official model inference.

---

## Step 6

Implement VideoMAE mask adapter.

Verify mask dimensions and semantics.

---

## Step 7

Implement basic BiSTAR selector.

Start with:

```text
random selector
```

Then add:

```text
gaze
temporal SSIM
edge
structural
full
```

---

## Step 8

Implement BiSTAR packet serialization and bitrate accounting.

---

## Step 9

Connect:

```text
selector
→ packetizer
→ receiver
→ VideoMAE
→ reconstructed frame
→ metrics
```

---

## Step 10

Implement matched-bitrate runner.

---

## Step 11

Implement ablation runner.

---

## Step 12

Freeze Phase 1–5 experimental infrastructure.

Create a Git tag:

```text
benchmark-v1.0
```

Only then start GStreamer/realtime work.

---

# 27. Codex coding constraints

Codex must obey:

1. Do not modify `third_party/VideoMAE` unless required.
2. Prefer adapters over upstream edits.
3. No hard-coded absolute paths.
4. No experiment parameter inside Python source if it can live in YAML.
5. Every experiment must support `--config`.
6. Every run must create a resolved config copy.
7. Every important module must have tests.
8. All subprocess calls must check return codes.
9. FFmpeg stdout/stderr must be saved.
10. No catch-all exception that hides failures.
11. All random behavior must use explicit seed.
12. Never compute reported bitrate from theoretical patch count alone.
13. No use of future frames in an experiment labeled real-time/causal.
14. Never allow hidden ground-truth pixels to leak into BiSTAR reconstruction.
15. Intermediate reconstruction visualizations must not affect metric computation.

---

# 28. Required CLI interfaces

## Environment check

```bash
python scripts/check_environment.py
```

---

## Dataset manifest

```bash
python scripts/build_manifest.py \
  --input datasets/raw \
  --output datasets/manifests/bistar_eval.csv
```

---

## RD experiment

```bash
python scripts/run_rd.py \
  --config configs/experiments/rd_sweep.yaml
```

---

## Matched-bitrate

```bash
python scripts/run_matched_bitrate.py \
  --config configs/experiments/matched_bitrate.yaml
```

---

## Ablation

```bash
python scripts/run_ablation.py \
  --config configs/experiments/ablation.yaml
```

---

## Smoke test

```bash
python scripts/run_smoke_test.py \
  --config configs/experiments/smoke_test.yaml
```

---

# 29. Minimum first milestone

The first milestone should **not** include GStreamer.

Milestone M1 is complete when the following command:

```bash
python scripts/run_rd.py \
  --config configs/experiments/rd_sweep.yaml
```

automatically:

1. reads the same test videos;
2. preprocesses them once;
3. runs H.264;
4. runs H.265;
5. measures real bitstream bytes;
6. decodes both bitstreams;
7. aligns decoded frames with original frames;
8. calculates PSNR/SSIM/MS-SSIM/LPIPS;
9. saves results to CSV;
10. generates RD curves;
11. stores FFmpeg commands;
12. stores environment metadata.

This milestone establishes the comparison infrastructure before BiSTAR is integrated.

---

# 30. Minimum second milestone

Milestone M2 is complete when:

```text
same source video
    ↓
BiSTAR selector
    ↓
actual serialized patch stream
    ↓
VideoMAE adapter
    ↓
reconstructed video
    ↓
same metric engine
```

works without manually changing code.

M2 must report:

```text
actual bitrate
payload bitrate
metadata bitrate
SSIM
LPIPS
VideoMAE inference latency
```

---

# 31. Final infrastructure acceptance

The infrastructure is considered ready for formal paper experiments only if all conditions below are satisfied.

### Reproducibility

- exact config stored;
- exact commits stored;
- exact checkpoint stored or hash recorded;
- FFmpeg build config stored.

### Fairness

- same source frames;
- same resolution;
- same FPS;
- same metrics;
- actual bitrate measured;
- patch metadata counted.

### VideoMAE correctness

- mask semantics verified;
- token ordering verified;
- hidden pixels unavailable to reconstruction;
- causal/non-causal status explicit.

### Benchmark correctness

- matched-bitrate runner works;
- CRF/bitrate sweep works;
- all raw measurements exported;
- no manually edited result tables.

### Test coverage

The following must pass:

```bash
pytest tests/unit
pytest tests/integration
pytest tests/regression
```

### Paper readiness

At minimum the infrastructure can automatically generate:

```text
RD curves
codec configuration table
ablation table
runtime table
per-frame logs
per-sequence statistics
aggregate statistics
```

---

# 32. Engineering recommendation

The implementation should be divided into three independent layers:

```text
Layer 1: Codec Benchmark
        FFmpeg + H.264/H.265

Layer 2: BiSTAR Reconstruction
        Selector + Sparse Transport + VideoMAE

Layer 3: Real-Time System
        GStreamer + Network Emulation + Latency Measurement
```

Do not couple these layers prematurely.

The first scientifically meaningful checkpoint is not successful VideoMAE reconstruction.

It is:

> **A unified benchmark in which H.264 and H.265 are measured correctly and reproducibly under controlled bitrate conditions.**

Only after that benchmark is validated should BiSTAR be connected to the same evaluation interface.

This prevents the original paper's main methodological problem from recurring: comparing methods under incompletely specified or incomparable communication conditions.
