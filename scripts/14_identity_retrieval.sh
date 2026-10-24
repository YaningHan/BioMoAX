#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONDONTWRITEBYTECODE=1
exec "${CORE_PYTHON:-python}" "$repo/pipelines/identity_retrieval.py" "$@"
