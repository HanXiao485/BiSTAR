#!/usr/bin/env bash
set -euo pipefail
interface=${1:?usage: setup_netem.sh INTERFACE BANDWIDTH_KBPS DELAY_MS JITTER_MS LOSS_PERCENT}
bandwidth=${2:?}; delay=${3:?}; jitter=${4:?}; loss=${5:?}
tc qdisc replace dev "$interface" root handle 1: tbf rate "${bandwidth}kbit" burst 32kbit latency 400ms
tc qdisc replace dev "$interface" parent 1:1 handle 10: netem delay "${delay}ms" "${jitter}ms" loss "${loss}%"
