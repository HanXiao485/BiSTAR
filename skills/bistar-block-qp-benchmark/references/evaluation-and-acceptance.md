# Evaluation, Accounting, and Acceptance

Read this reference before modifying runner/report paths and before declaring completion.

## Evaluation invariants

- Integrate RD, matched-bitrate, codec search, and ROI search through existing dispatch and report mechanisms. Tune only on existing calibration/validation splits, never held-out evaluation data.
- Keep source, resolution, frame interval, frame rate, and pixel format identical in comparisons.
- Preserve the FFmpeg H.264 baseline. Add a no-map/zero-map control using the same native backend and resolved settings. Attribute effects to maps only within that same backend; disclose all differences across backends.
- Charge actual complete elementary-stream bytes, including NAL and QP signaling. Generated local maps/logs are free only if they are not transmitted. Charge any side channel. Never charge visualization MP4s or research logs as codec bytes.
- Reset provider and encoder for each rate-match attempt and use actual output bytes. Preserve existing tolerance, iteration-limit, and unreachable-target semantics. Equal target settings do not imply equal achieved rate.
- Describe iterative offline matching as offline selection, not online control.

Append rather than replace result fields: provider, backend/version, ROI source, achieved bitrate, ROI/non-ROI/global quality, provider and encode timing, and QP trace path. Preserve existing field meanings and script exit codes.

Cache signatures cover source data, ROI, provider config/code, backend/version, encoder settings, and external-map content hash. Old caches must not satisfy the new method. Store raw map/quality/RD CSV or NPZ plus plotting metadata. Separate throughput from end-to-end latency and disclose lookahead, policy, and bridge overhead.

## Required acceptance checks

| Check | Evidence required |
|---|---|
| Applied QP | Encode equal-complexity left/right synthetic texture, swap side deltas, and prove the intended macroblock QPs swap using encoder trace or bitstream parsing. Size/PSNR alone is insufficient. |
| Zero control | Same backend/settings with zero map versus no custom-offset path are equivalent, or the repeatable difference is explained. Cross-backend bitstream identity is not required. |
| Dynamic map | Alternate ROI sides per source frame and verify maps, trace, PTS, and frame IDs have no one-frame shift. Test reorder/flush if B-frames are claimed. |
| Dimensions | Test non-multiples of 16, pad/crop, visible dimensions, frame count, and PTS. Reject or explicitly handle incompatible odd YUV420 dimensions. |
| Bad input | Wrong shape, NaN/Inf, range overflow, absent external frame, duplicate frame ID, unknown provider, and conflicting configuration fail with actionable errors. |
| ROI edges | Full, empty, invalid, and mixed-block ROI follow configured behavior and do not leak evaluation-only labels. |
| Rate/state | Recompute bitrate from actual bytes, surface unreachable matching, and demonstrate reset prevents trial-to-trial state leakage. |
| Compatibility | Small ordinary H.264/H.265 and relevant BiSTAR/Atlas paths still work; without the native library, disabled block-QP code does not affect imports or legacy execution. |
| Extensibility | A minimal custom provider runs without backend/runner edits; an external map can be replayed deterministically. |
| Outputs | Resolved config, versions, traces, checksums, logs, quality tables, and plot metadata are complete; resume uses only compatible caches. |

Run focused tests and smoke inputs by default. Do not launch formal sweeps or training. If the environment prevents a native build or a check, retain the implementation and reproducible command, label that check `not run` or `unverified`, and report the concrete blocker. Never fabricate passing evidence or performance gains.
