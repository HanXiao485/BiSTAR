# Experiment modes

`smoke_test.yaml` creates a deterministic 224×224, 15 FPS source. `rd_sweep.yaml` and `matched_bitrate.yaml` use actual raw-bitstream bytes. `ablation.yaml` counts the serialized header, indices, codec identifier, payload length, and compressed payload. `realtime.yaml` is deliberately disabled until offline acceptance and requires an explicit network interface.
