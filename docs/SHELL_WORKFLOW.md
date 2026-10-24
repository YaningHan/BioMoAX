# Shell workflow

Use Bash on Linux (or a configured Linux environment under WSL). Scripts work from any directory. Copy `configs/pipeline.env.example` outside the repository, set absolute paths and the Python executable for each environment. Configuration is sourced as Bash code: only use trusted files. Do not put secrets in it.

```bash
bash scripts/01_prepare_data.sh /absolute/path/pipeline.env
bash scripts/02_cellpose.sh /absolute/path/pipeline.env
bash scripts/03_cellprofiler.sh /absolute/path/pipeline.env
bash scripts/04_deepprofiler.sh /absolute/path/pipeline.env
```

Or run those four stages sequentially:

```bash
bash scripts/run_pipeline.sh /absolute/path/pipeline.env preprocess
```

Stage 1 checks the site manifest and local input file presence; it does **not** download images, generate paper splits, or verify that your sample selection matches the paper. Stage 2 creates `RUN_ROOT/cellpose/crops.h5`; stages 3 and 4 consume it automatically. For an existing crop file, set `CROPS_H5` and run stages 3/4 individually. See [pipeline details](../pipelines/README.md) for channel order, route scope and compatible environments.

## Training

First assemble existing processed inputs into a new training root:

```bash
bash scripts/10_assemble_training.sh --source-root /path/to/prepared_project \
  --output /path/to/new_training_root
```

For files stored in different locations, use `--manifest inputs.json` instead. The JSON maps the exact required training paths returned by `pipelines/assemble_training.py:required_files`, relative to `data/processed/`, to their source files. [DATA.md](DATA.md) describes their layout; it also lists evaluation inputs not required by this training stage. Source paths may be absolute or relative to the JSON file. This stage organizes processed inputs; it does not infer compound annotations, reconstruct missing arrays, or replace the source-specific preprocessing steps.

Large HDF5/NumPy arrays are symlinked by default; metadata are copied so training does not modify originals. Keep linked source files available, or use `--copy-arrays` for a standalone copy with sufficient disk space. The command checks the input contract and core record/partition mappings; only `assembly_complete.json` marks success. Failed output directories are retained for inspection, not automatically overwritten.

Set `PROJECT_ROOT` in your configuration to the new training root, then run:

```bash
bash scripts/05_train.sh /absolute/path/pipeline.env
```

This first validates `PROJECT_ROOT` against [the complete training contract](DATA.md). It is not assembled automatically from stage 2: expression arrays, compound mappings, annotations, and dataset-specific partitions are also required. `TRAIN_PREFLIGHT=1` defaults to two update checks; set it to `0` explicitly for full training. Use a fresh `RUN_ROOT` for each attempt. The historical training code requires a compatible CUDA/BF16 device and sufficient host memory.

## Condition features

Create a CSV with unique `record_id,label` pairs for one crop HDF5 collection. Labels must match its `labels` dataset; record IDs must match the retrieval task IDs. Output row order follows this CSV.

```bash
# BioMoAX: specify a model checkpoint and encode each condition.
bash scripts/09_condition_features.sh image --crops runs/cellpose/crops.h5 \
  --records conditions.csv --checkpoint /path/to/model.pt --head moa \
  --device cuda --max-crops 100 --output runs/image_conditions

# CellProfiler: aggregate crop features using their explicit source-row mapping.
bash scripts/09_condition_features.sh baseline --crops runs/cellpose/crops.h5 \
  --records conditions.csv --features runs/cellprofiler/features.npy \
  --source-rows runs/cellprofiler/source_rows.npy \
  --valid-rows runs/cellprofiler/valid_rows.npy --output runs/cp_conditions
```

For DeepProfiler, use its `features.npy` and `source_rows.npy`; omit `--valid-rows` unless you have an explicit mask. Baselines default to the raw feature mean over all finite valid crops. `--aggregation normalized-mean` instead normalizes each crop before averaging and normalizes the result. Neither option applies native-profile standardization or learns a cross-modal mapping.

Image encoding averages normalized crop embeddings and normalizes the mean. The default takes the first 100 crops in ascending source-row order per condition; `--max-crops 0` uses all crops. This deterministic selection is recorded, not a reconstruction of historical random draws. For matched-crop comparisons, supply the same selected crop rows to every method.

Each output includes `features.npy`, `feature_ids.csv`, `coverage.csv`, and `selected_crops.csv`. Invalid baseline crop rows are counted and omitted; a condition with no usable crops fails the run instead of disappearing from evaluation. Set `QUERY_POOL`/`GALLERY_POOL` and their ID CSVs to these outputs for the task-based retrieval mode below. Same-modality pools must use compatible feature processing; cross-modal baselines still require a training-only mapping.

Run `python pipelines/test_condition_features.py -v` to check grouping, source-row alignment, masks, and shell-to-task selection. Image grouping uses a stub encoder in this test, not a trained checkpoint.

## Retrieval

Core within-cell, cross-cell and HT29 MOA task files are bundled under `manifests/tasks/`. Cross-batch and A549 CRISPR files are under `manifests/external_tasks/`. Their actual eligibility counts and mapping provenance are described in [the manifest guide](../manifests/README.md). To regenerate both sets from processed data and annotations, run `bash scripts/07_prepare_tasks.sh /absolute/path/pipeline.env`. The CRISPR common-gallery mask defaults to the bundled record; optionally set `CRISPR_GALLERY_KEYS` explicitly.

For automatic ID-based selection, set `TASK_DIR` to one task directory and supply `QUERY_POOL`, `QUERY_POOL_IDS`, `GALLERY_POOL`, and `GALLERY_POOL_IDS`. Pools are condition-level feature `.npy` files, each with its ordered `record_id` CSV. Stage 6 selects the required rows, constructs masks and evaluates. Query and gallery pools may represent different modalities; their embeddings must share the same comparison space. This mode overrides the direct task-input variables below.

```bash
bash scripts/06_evaluate.sh /absolute/path/pipeline.env
```

Set `QUERY_FEATURES` and `GALLERY_FEATURES` to finite, nonzero condition-level embedding matrices with equal feature dimensions. `QUERY_IDS` and `GALLERY_IDS` are CSV files with a `record_id` column, in exactly the same row order as their respective matrices. IDs must be unique in each file; use source/cell-qualified IDs to avoid accidental collisions.

`POSITIVE_PAIRS` is a CSV with `query_id,gallery_id` columns specifying the task's positive links. `EXCLUDED_PAIRS`, when supplied, has the same format. Empty exclusion configuration means **no exclusions**; for same-cell MOA retrieval, explicitly include the same-record pairs. The runner does not infer MOA overlap, chemical identity or target annotations from names. CRISPR target anchors/one-hop MOA expansion must be resolved upstream. Unknown IDs fail, rather than being silently dropped.

The stage writes `retrieval.npz` and `metrics.json`. Queries without eligible positives are handled by the existing evaluator. ID arrays are preserved in the NPZ for traceability. Run condition-feature preparation above before evaluation. Ridge mappings and checkpoint choice remain explicit; the core `encode` and `ridge-fit` commands are also available separately.

## Logs and failure behavior

Logs live under `RUN_ROOT/logs`. Each attempt creates a stage lock; successful completion additionally creates a `.complete` marker. Errors stop the stage and the sequential runner. Existing attempts are not overwritten or silently resumed. Inspect failed logs and use a fresh run root for a clean rerun; do not treat partial arrays as completed results.

`python pipelines/test_shell_workflow.py` tests synthetic retrieval through the actual shell entry, including paths with spaces, duplicate-run refusal, and unknown-ID rejection. It does not execute full training or segmentation.
