#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
raw_root="${project_root}/datasets/raw/EyeTrackUAV2/RAW_DATA/RAW_DATA_OrigRes_DomEye"
listing="${raw_root}/.listing"

if [[ ! -f "${listing}" ]]; then
  echo "Missing official FTP listing: ${listing}" >&2
  exit 1
fi

awk '$1 ~ /^d/ && $9 !~ /^\./ {print $9}' "${listing}" \
  | tr -d '\r' \
  | xargs -P 8 -I{} /usr/bin/wget \
      --mirror --continue --no-host-directories --cut-dirs=3 \
      --directory-prefix="${raw_root}" --no-verbose --timeout=60 --tries=10 \
      "ftp://ftp.ivc.polytech.univ-nantes.fr/EyeTrackUAV2/RAW_DATA/RAW_DATA_OrigRes_DomEye/{}/{}_TASK/"

cd "${project_root}"
/home/xiaohan/anaconda3/envs/bistar/bin/python scripts/audit_eyetrackuav2.py \
  --condition TASK --checksums
