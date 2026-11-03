# Provider, ROI, and Configuration Contract

Read this reference when implementing providers, map validation, ROI conversion, configuration, or trace output.

## Provider lifecycle

Implement the repository-appropriate equivalent of:

```python
class QPMapProvider:
    def reset(self, sequence_info, config): ...
    def select(self, context): ...  # QPDecision
    def update(self, feedback): ...
    def close(self): ...
```

`context` includes sequence ID, source frame ID, PTS/time base, original and padded dimensions, grid shape, target bitrate, available causal ROI coverage, and optional RGB/encoder features. Every optional feature has an availability indicator. Providers declare required features; missing data either raises a clear error or uses the configured fallback, never fake zeros.

`QPDecision` contains mode (`delta` or supported `absolute`), a finite map shaped `[ceil(H/16), ceil(W/16)]`, provider metadata, and a coordinate-system version. Index row-major as `row * ncols + col`. Validate dimensions, finite values, bounds, mode, source frame ID, uniqueness, and ordering before encoding.

`feedback` maps back to the source frame by PTS and may include actual bytes, encode time, cumulative/rolling rate, base QP, and applied block QP with availability flags. With delay/reordering, never associate output by call order. Do not invent per-frame bytes by dividing a sequence total.

Call `reset` for every sequence and every bitrate-matching attempt. `close` must be safe after partial failures. Load external providers by an explicit registry or controlled `module:Class` import, never `eval`.

## Required providers and ROI composition

- `uniform`: zero or one constant offset across the map.
- `fixed_roi`: configured ROI and non-ROI offsets.
- `external_map`: exact source-frame-keyed replay with strict missing/duplicate/shape checks.

Derive macroblock coverage from a causal ROI source over valid image pixels; padding must not inflate coverage. Default mixed-block behavior protects any intersecting ROI. Support `coverage_threshold` and optional `soft_blend`, e.g. `a*delta_roi + (1-a)*delta_background`.

The non-ROI policy supports fixed delta, inherit/no added background offset, or external values. Use one explicit composition stage so offsets cannot be applied twice. Define conflict precedence, such as full external map over regional composition, and reject ambiguous configurations. Retain raw regional values, composed values, and clipped values for tracing.

Keep encoding-visible ROI separate from evaluation ROI. Ground-truth eye data available only for evaluation must not reach a causal provider. Mark any deliberate oracle run in method names and outputs. For missing/invalid ROI, default to a recorded uniform fallback; preserve existing evaluation-validity rules.

## Configuration behavior

Add a new codec configuration without changing the ordinary H.264 defaults. Resolve and record backend version plus AQ, VBV, lookahead, thread, scenecut, GOP, B-frame, tune, preset, pixel format, block size, provider, mode, bounds, mixed-block and missing-ROI policy, target rate, and tracing options.

Treat example offsets and bitrates as starting values, not optima. Fail early on unknown providers, unsupported modes, illegal maps, frame mismatches, conflicting composition, and unsupported dimensions/pixel formats. If odd YUV420 dimensions are handled, document the pad/crop rule and preserve visible dimensions.

Suggested shape, adapted to the actual schema:

```yaml
codec:
  name: h264_block_qp
  implementation: libx264_qp_bridge
  pixel_format: yuv420p
  encoder:
    preset: medium
    mode: bitrate
    bitrate_kbps: 100
    gop_size: 30
    b_frames: 0
    tune: zerolatency
  qp_control:
    provider: fixed_roi
    mode: delta
    block_size: 16
    roi_delta: -4.0
    non_roi: {policy: fixed_delta, delta: 4.0}
    delta_limits: [-12.0, 12.0]
    mixed_block_policy: any_overlap
    missing_roi_policy: uniform
    external_map_path: null
    trace_requested: true
    trace_applied: false
```

QP traces include frame ID, row, column, coverage, requested delta, nullable requested absolute/base/applied QP, and availability. Never serialize unavailable numeric values as zero. Also record clipping rate, ROI fallback count, and map checksum.
