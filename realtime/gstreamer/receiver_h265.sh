#!/usr/bin/env bash
set -euo pipefail
port=${1:?usage: receiver_h265.sh PORT}
gst-launch-1.0 udpsrc port="$port" caps='application/x-rtp,media=video,encoding-name=H265,payload=96' ! rtph265depay ! avdec_h265 ! videoconvert ! autovideosink sync=true
