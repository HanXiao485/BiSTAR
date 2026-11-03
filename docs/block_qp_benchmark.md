# H.264 block-QP benchmark

`h264_block_qp` encodes one full-frame, standard-decodable H.264 elementary stream and supplies a source-frame-keyed delta map to libx264's public `x264_picture_t.prop.quant_offsets` interface. It is not an RL-RC-DoT reproduction, an atlas codec, or a collection of independent patch streams.

## Build and run

Build the optional bridge without replacing system FFmpeg or x264:

```bash
./scripts/build_x264_qp_bridge.sh
```

Run an RD or matched-bitrate configuration from the repository root and active Python environment:

```bash
./scripts/run_block_qp_benchmark.sh --config configs/experiments/block_qp_smoke.yaml
```

The codec configuration is `configs/codecs/h264_block_qp.yaml`. Ordinary `h264.yaml` and `h265.yaml` do not import or require the bridge.

## Providers

Uniform maps use a single delta:

```yaml
qp_control:
  provider: uniform
  mode: delta
  block_size: 16
  delta: 0.0
  delta_limits: [-12.0, 12.0]
```

Fixed ROI accepts pixel or normalized rectangles. A rectangle is `[x, y, width, height]`; `per_frame_rectangles` keys are source frame IDs. With no rectangle, the configured uniform fallback is explicit and counted.

```yaml
qp_control:
  provider: fixed_roi
  mode: delta
  block_size: 16
  roi_delta: -4.0
  non_roi: {policy: fixed_delta, delta: 4.0}
  mixed_block_policy: any_overlap
  missing_roi_policy: uniform
  missing_roi_delta: 0.0
  delta_limits: [-12.0, 12.0]
  roi:
    coordinate_space: normalized
    rectangles: [[0.25, 0.25, 0.5, 0.5]]
```

External maps are `.npz` files containing unique integer `frame_ids` and float `maps` shaped `[frames, ceil(height/16), ceil(width/16)]`. A plain `.npy` array is also accepted and implies consecutive frame IDs starting at zero.

```yaml
qp_control:
  provider: external_map
  mode: delta
  block_size: 16
  external_map_path: path/to/maps.npz
  delta_limits: [-12.0, 12.0]
```

All maps must be finite, in range, and complete for the encoded source frames. The implementation rejects absolute mode, B-frames, odd YUV420 dimensions, wrong grids, missing/duplicate frame IDs, and unavailable native bridges.

## Evidence and limits

Each encode writes `qp_requested.csv`, `qp_maps.f32`, a SHA-256 checksum, `x264_frame_trace.csv`, and `qp_metadata.json`. Elementary-stream file size is the billed byte count; research traces are not included.

The installed public x264 API exposes the requested macroblock offsets and frame-average effective CRF, but not applied per-macroblock QP values. Accordingly, `applied_block_qp_available` is `false`; requested deltas are never presented as measured applied QPs. A future acceptance claim for actual block QP requires an encoder instrumentation trace or a validated bitstream parser.
