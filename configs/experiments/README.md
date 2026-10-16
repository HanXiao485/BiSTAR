# Experiment modes

`smoke_test.yaml` creates a deterministic 224×224, 15 FPS source. `rd_sweep.yaml` and `matched_bitrate.yaml` compare H.264, H.265, per-frame BiSTAR, and the GOP-stable H.264 patch atlas using actual wire bytes. `ablation.yaml` compares `jpeg_patch` and `h264_atlas` for every selector and visible-patch budget. `realtime.yaml` validates the real-time dependencies and records the atlas protocol settings; network shaping still requires an explicit interface and command.

`roi_config_search.yaml` searches codec and sparse-transport configurations on independently annotated task ROIs, selects configurations on `val`, and evaluates frozen winners on `test`. `roi_config_search_single_video.yaml` uses a temporal holdout for exploratory experiments on the current video. Both require complete ROI CSV annotations and preserve actual-rate match failures. See `docs/roi_config_search.md` for annotation, grids, plots, and provenance.
