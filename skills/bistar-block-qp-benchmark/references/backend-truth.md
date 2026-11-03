# Backend Truth and Capability Contract

Read this reference when assessing, implementing, or validating the encoder backend.

## Supported first version

- H.264/AVC, 8-bit YUV420p, 16×16 macroblocks only.
- One standard-decodable full-frame elementary stream with per-frame block offsets.
- Conceptual semantics: `QP[t, i] ≈ frame_base_qp[t] + delta_qp[t, i]`. Requested values are not evidence of applied values because rounding, clipping, AQ, VBV, frame type, lookahead, and encoder decisions intervene.
- H.265 remains an unchanged baseline; do not project H.264 macroblock assumptions onto CTUs.

## Backend decision

For libx264, verify against the exact installed public headers/source:

- the `x264_picture_t` quant-offset interface and required AQ settings;
- buffer ownership and lifetime through delayed output;
- macroblock grid dimensions and row-major ordering;
- whether values are delta or absolute and their valid range;
- PTS/source-frame mapping, flush behavior, threads, B-frames, and lookahead;
- whether actual base/block QP tracing is available.

If a patch is unavoidable, pin its upstream revision and keep a minimal patch plus isolated build instructions. Do not replace system libraries.

For FFmpeg AVFrame ROI side data, verify the checked-out/installed libavcodec and libx264 wrapper source, dynamic per-frame injection, coordinate semantics, and qoffset conversion. A successful command is not proof of arbitrary grid support.

Expose backend capabilities in machine-readable form: delta/absolute modes, block size, runtime status, applied-QP tracing, base-QP availability, frame-level base control, supported pixel formats, and reorder support. Reject unsupported modes. Never derive an absolute map from an unknown base QP.

The provider may receive `base_qp=None`; x264 rate control may not expose the current frame's base QP before selection. A future policy requiring it needs a real encoder callback/extraction path, not a fabricated value.

## Loading and failures

Load native components only when the block-QP method is selected. A missing or failed optional build must not break imports or ordinary H.264/H.265/BiSTAR/Atlas paths. It must, however, fail the selected block-QP method clearly; never silently fall back to ordinary H.264 while reporting block-QP success.

Use argument arrays for subprocesses, retain stderr, close encoders on all error paths, clean temporary artifacts, and write durable results atomically.
