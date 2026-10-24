#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${CORE_PYTHON:-python}" -m biomoax.cli ridge-predict "$@"
