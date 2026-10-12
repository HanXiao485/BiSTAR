# Architecture

The benchmark has three independent layers: FFmpeg codec benchmarking, sparse BiSTAR transport/reconstruction, and a deferred real-time transport layer. Offline experiments consume only declared manifest rows. BiSTAR decoders receive a sparse frame reconstructed from serialized bytes plus decoder state; their API has no hidden ground-truth parameter.

VideoMAE uses `mask=True` for hidden tokens. `MaskAdapter` converts the selector's `visible=True` convention by inversion and flattens in time-major `(T, H, W)` order.

The official visualization helper uses each ground-truth patch's mean and variance to denormalize predictions. That operation leaks hidden pixels and is therefore forbidden here. The adapter zeroes hidden input pixels before inference and denormalizes predicted patches using global RGB statistics computed only from transmitted visible pixels. This adaptation is recorded in method metadata.
