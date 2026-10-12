#!/usr/bin/env bash
set -euo pipefail
source_uri=${1:?usage: sender_h265.sh SOURCE_URI HOST PORT}; host=${2:?}; port=${3:?}
gst-launch-1.0 uridecodebin uri="$source_uri" ! videoconvert ! x265enc tune=zerolatency bframes=0 ! rtph265pay ! udpsink host="$host" port="$port"
