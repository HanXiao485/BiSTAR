#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT="${ROOT}/build/x264_qp_bridge"
mkdir -p "$(dirname "${OUTPUT}")"
cc -std=c99 -O2 -Wall -Wextra \
  $(pkg-config --cflags x264) \
  "${ROOT}/src/codecs/native/x264_qp_bridge.c" \
  -o "${OUTPUT}" $(pkg-config --libs x264)
printf '%s\n' "${OUTPUT}"
