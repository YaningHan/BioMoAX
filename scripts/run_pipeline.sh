#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo 'Usage: bash scripts/run_pipeline.sh /absolute/path/config.env prepare|cellpose|cellprofiler|deepprofiler|train|tasks|evaluate|preprocess' >&2
  exit 2
fi
config=$1
stage=$2
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
[[ -f "$config" ]] || { echo "Missing config: $config" >&2; exit 2; }
# Configuration is executable shell code; use only a trusted local file.
source "$config"
: "${RUN_ROOT:?Set RUN_ROOT}"
[[ "$RUN_ROOT" = /* ]] || { echo 'RUN_ROOT must be absolute' >&2; exit 2; }
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
if [[ "$stage" = preprocess ]]; then
  for step in prepare cellpose cellprofiler deepprofiler; do
    bash "$repo/scripts/run_pipeline.sh" "$config" "$step"
  done
  exit 0
fi
case "$stage" in
  prepare|cellpose|cellprofiler|deepprofiler|train|evaluate|tasks) ;;
  *) echo "Unknown stage: $stage" >&2; exit 2 ;;
esac
mkdir -p "$RUN_ROOT/logs"
# Atomic stage lock: failures retain logs; no automatic overwrite or resume.
lock="$RUN_ROOT/logs/$stage.lock"
mkdir "$lock" || { echo "Stage already attempted: $stage. Inspect logs before retrying in a new RUN_ROOT." >&2; exit 1; }
exec > >(tee "$RUN_ROOT/logs/$stage.log") 2>&1
trap 'rc=$?; if ((rc != 0)); then echo "Stage failed ($rc): $stage"; fi' EXIT
core=${CORE_PYTHON:-python}
crops=${CROPS_H5:-$RUN_ROOT/cellpose/crops.h5}
cd "$repo"
case "$stage" in
  tasks)
    "$core" pipelines/prepare_records.py --project-root "${PROJECT_ROOT:?}" --output "$RUN_ROOT/records"
    "$core" pipelines/prepare_moa_tasks.py --records "$RUN_ROOT/records/drug_records.csv" --output "$RUN_ROOT/tasks"
    "$core" pipelines/prepare_external_tasks.py --project-root "$PROJECT_ROOT" --records "$RUN_ROOT/records/drug_records.csv" --crispr-gallery-keys "${CRISPR_GALLERY_KEYS:-$repo/manifests/crispr_a549_gallery_keys.json}" --output "$RUN_ROOT/external_tasks"
    ;;
  prepare)
    "$core" pipelines/check_manifest.py --manifest "${SITE_MANIFEST:?Set SITE_MANIFEST}" --output "$RUN_ROOT/input_check.json"
    ;;
  cellpose)
    gpu=(); [[ ${USE_GPU:-0} != 1 ]] || gpu=(--gpu)
    "${CELLPOSE_PYTHON:-python}" pipelines/segment.py --manifest "${SITE_MANIFEST:?}" --weights "${CELLPOSE_WEIGHTS:?}" --output "$RUN_ROOT/cellpose" "${gpu[@]}"
    ;;
  cellprofiler)
    "${CP_PYTHON:-python}" pipelines/cellprofiler.py --input "$crops" --executable "${CP_EXECUTABLE:-cellprofiler}" --output "$RUN_ROOT/cellprofiler"
    ;;
  deepprofiler)
    "${DP_PYTHON:-python}" pipelines/deepprofiler.py --input "$crops" --weights "${DP_WEIGHTS:?}" --output "$RUN_ROOT/deepprofiler"
    ;;
  train)
    "$core" -m biomoax.cli check-data --project-root "${PROJECT_ROOT:?}"
    preflight=(); [[ ${TRAIN_PREFLIGHT:-1} != 1 ]] || preflight=(--preflight)
    "$core" -m biomoax.cli train --project-root "$PROJECT_ROOT" --output "$RUN_ROOT/training" --seed "${TRAIN_SEED:-43}" "${preflight[@]}"
    ;;
  evaluate)
    if [[ -n ${TASK_DIR:-} ]]; then
      QUERY_IDS="$TASK_DIR/query_ids.csv"
      GALLERY_IDS="$TASK_DIR/gallery_ids.csv"
      POSITIVE_PAIRS="$TASK_DIR/positive_pairs.csv"
      EXCLUDED_PAIRS="$TASK_DIR/excluded_pairs.csv"
      QUERY_FEATURES="$RUN_ROOT/query.npy"
      GALLERY_FEATURES="$RUN_ROOT/gallery.npy"
      "$core" pipelines/select_task_features.py --features "${QUERY_POOL:?}" --feature-ids "${QUERY_POOL_IDS:?}" --task-ids "$QUERY_IDS" --output "$QUERY_FEATURES"
      "$core" pipelines/select_task_features.py --features "${GALLERY_POOL:?}" --feature-ids "${GALLERY_POOL_IDS:?}" --task-ids "$GALLERY_IDS" --output "$GALLERY_FEATURES"
    fi
    exclusions=(); [[ -z ${EXCLUDED_PAIRS:-} ]] || exclusions=(--excluded-pairs "$EXCLUDED_PAIRS")
    "$core" pipelines/build_retrieval.py --query "${QUERY_FEATURES:?}" --gallery "${GALLERY_FEATURES:?}" --query-ids "${QUERY_IDS:?}" --gallery-ids "${GALLERY_IDS:?}" --positive-pairs "${POSITIVE_PAIRS:?}" "${exclusions[@]}" --output "$RUN_ROOT/retrieval.npz"
    "$core" -m biomoax.cli evaluate --input "$RUN_ROOT/retrieval.npz" --output "$RUN_ROOT/metrics.json"
    ;;
esac
touch "$RUN_ROOT/logs/$stage.complete"
echo "Completed: $stage"
