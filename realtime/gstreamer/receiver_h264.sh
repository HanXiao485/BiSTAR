#!/usr/bin/env bash
set -euo pipefail
port=${1:?usage: receiver_h264.sh PORT}
gst-launch-1.0 udpsrc port="$port" caps='application/x-rtp,media=video,encoding-name=H264,payload=96' ! rtph264depay ! avdec_h264 ! videoconvert ! autovideosink sync=true
