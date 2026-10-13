# Architecture

The benchmark has three independent layers: FFmpeg codec benchmarking, sparse BiSTAR transport/reconstruction, and a deferred real-time transport layer. Offline experiments consume only declared manifest rows. BiSTAR decoders receive a sparse frame reconstructed from serialized bytes plus decoder state; their API has no hidden ground-truth parameter.

VideoMAE uses `mask=True` for hidden tokens. `MaskAdapter` converts the selector's `visible=True` convention by inversion and flattens in time-major `(T, H, W)` order.

The official visualization helper uses each ground-truth patch's mean and variance to denormalize predictions. That operation leaks hidden pixels and is therefore forbidden here. The adapter zeroes hidden input pixels before inference and denormalizes predicted patches using global RGB statistics computed only from transmitted visible pixels. This adaptation is recorded in method metadata.

## GOP patch atlas transport

The real-time-oriented BiSTAR path selects one spatial mask from the first frame of each GOP and the previous GOP's final source frame. It holds that mask for the whole GOP, packs visible patches into stable atlas slots, and encodes the atlas as an H.264 elementary stream with B-frames disabled. The wire packet contains a fixed `BATL` header, a bitmask (25 bytes for a 14×14 grid), and the encoded payload. All three parts count toward bitrate.

The receiver parses the packet without source-frame state, decodes the atlas, scatters its slots into zero-filled sparse frames, and gives only those frames and the visible mask to VideoMAE. Selection is causal. Reconstruction is chunked and waits for the GOP, so latency reporting includes both the per-frame algorithmic wait and measured encode, decode, and inference time.

When `selector.patch_budget_mode` is `false`, a causal controller adjusts the patch count between GOPs. The initial count scales the configured reference count by target bitrate. Later counts use the previous GOP's complete wire size, exponential smoothing, a maximum step, and configured candidates. No mask changes within a GOP. Explicit patch counts used by ablation runs force fixed mode so transport comparisons retain identical spatial budgets.

## Spatial quality reporting

RD evaluation aggregates quality on the common 16×16 patch grid before comparison videos are encoded. Each method and target rate produces patch PSNR, patch SSIM, normalized L1, and selection frequency. Heatmaps use a shared scale across methods at the same target rate. Full-frame codecs have selection frequency one everywhere; sparse methods accumulate the masks actually used for each frame or GOP. This separates spatial reconstruction quality from the selector's allocation pattern.
