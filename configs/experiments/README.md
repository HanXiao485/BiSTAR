# Experiment modes

`smoke_test.yaml` creates a deterministic 224×224, 15 FPS source. `rd_sweep.yaml` and `matched_bitrate.yaml` compare H.264, H.265, per-frame BiSTAR, and the GOP-stable H.264 patch atlas using actual wire bytes. `ablation.yaml` compares `jpeg_patch` and `h264_atlas` for every selector and visible-patch budget. `realtime.yaml` validates the real-time dependencies and records the atlas protocol settings; network shaping still requires an explicit interface and command.
