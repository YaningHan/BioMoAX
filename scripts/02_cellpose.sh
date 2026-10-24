#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 1 ]] || { echo 'Usage: bash scripts/02_cellpose.sh /absolute/path/config.env' >&2; exit 2; }
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/run_pipeline.sh" "$1" cellpose
