#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="${2:?--config requires a path}"; shift 2 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ -n "${CONFIG}" ]] || { printf '%s\n' 'Usage: scripts/run_block_qp_benchmark.sh --config PATH' >&2; exit 2; }
cd "${ROOT}"
exec python scripts/run_block_qp_benchmark.py --config "${CONFIG}"
