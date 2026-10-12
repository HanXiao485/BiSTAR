#!/usr/bin/env bash
set -euo pipefail
source_uri=${1:?usage: sender_h264.sh SOURCE_URI HOST PORT}; host=${2:?}; port=${3:?}
gst-launch-1.0 uridecodebin uri="$source_uri" ! videoconvert ! x264enc tune=zerolatency bframes=0 ! rtph264pay ! udpsink host="$host" port="$port"
