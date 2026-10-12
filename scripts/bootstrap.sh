#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
env_prefix="/home/xiaohan/anaconda3/envs/bistar"
conda env create --prefix "$env_prefix" -f "$project_root/environment.yml"
echo "Run: conda activate $env_prefix"
