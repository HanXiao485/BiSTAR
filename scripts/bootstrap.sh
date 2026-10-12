#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
env_prefix="$project_root/.conda/bistar-full"
CONDA_PKGS_DIRS="$project_root/.conda/pkgs" conda env create --prefix "$env_prefix" -f "$project_root/environment.yml"
echo "Run: conda activate $env_prefix"
