#!/usr/bin/env bash
set -euo pipefail
interface=${1:?usage: reset_netem.sh INTERFACE}
tc qdisc del dev "$interface" root 2>/dev/null || true
