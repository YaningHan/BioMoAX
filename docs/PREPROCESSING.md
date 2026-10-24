# Data preprocessing

Download inputs from [the original providers](DATA_SOURCES.md). Commands and input formats are described in [the pipeline guide](../pipelines/README.md); use separate [environments](../environments/README.md) for the image-processing tools.

## Image crops

`segment.py` applies illumination correction, segments the DNA channel with Cellpose, and extracts five-channel 128 × 128 crops. Supply a site manifest with image paths, illumination files, and condition labels. Output channel order is DNA, ER, RNA, AGP, Mito. Existing crop files must have an explicit compatible channel order; do not infer it from array shape.

## CellProfiler and DeepProfiler

`cellprofiler.py` measures each crop with the included CellProfiler pipeline and returns median nuclear features, nuclei counts, and validity flags. `deepprofiler.py` uses the external CellPainting-CNN model to produce a 1,280-dimensional vector per crop. Both preserve source-row mappings.

These commands produce crop-level features. Native site-level profiles use a different measurement and aggregation unit and should not be substituted with crop averages when reproducing the corresponding paper baselines.

### Native CellProfiler profile tables

For U2OS well-level profiles, first aggregate wells by compound:

```bash
bash scripts/12_aggregate_native_cp.sh --wells cpg0012_CP_self_well.csv \
  --columns pipelines/assets/native_cp_columns.txt --output runs/cp_compounds
```

This entry takes existing well-level medians. It computes the median across wells for each `brd_core`, retains its `int_label`, and fills missing compound-feature values using the column median. All-missing feature columns and ambiguous label mappings fail. It writes `profiles.csv`, which can be passed to stage 11 below. `--reference cpg0012_CP_self.csv` optionally checks the result against an existing compound table. This rule is specific to the U2OS native profiles; auxiliary-cell profiles use different aggregation rules. Raw-image measurement and cell-to-well aggregation are separate upstream stages.

Use `scripts/11_native_cp.sh` to align existing condition-level CellProfiler tables and reconstruct the native scaled matrices:

```bash
bash scripts/11_native_cp.sh --profiles native_profiles.csv \
  --records conditions.csv --label-column new_int_label \
  --columns pipelines/assets/native_cp_columns.txt --output runs/native_cp
```

`conditions.csv` contains unique `record_id,label` pairs in the desired output order. Use the task record IDs and the native matrix row labels (for the bundled drug records, `expression_row`, not necessarily `h5_label`). For U2OS, pass `--label-column int_label --mapping labels.csv`, where `labels.csv` contains `source_label,label` derived from the source manifest's `int_label,new_int_label` columns. Repeated identical mapping pairs are accepted; conflicting mappings are rejected.

The input tables are `cpg0012_CP_self.csv` for U2OS and `cpg0004_CP_self_1039.csv` for A549. This command starts from aggregated profiles, not raw images or single-cell measurements. The supplied 1,130-column list fixes feature order. Outputs include `raw.npy`, robust-scaled `features.npy`, IDs, a coverage table and a validity mask. Missing profiles retain zero placeholders in `raw.npy` and are flagged invalid; scaling follows the native matrix convention and includes these placeholders. Apply the validity mask explicitly in downstream tasks.

Use `raw.npy`, **not** the robust-scaled matrix, when preparing training pairs for `biomoax ridge-fit`; that command fits its own training-only standardization. The native robust scaler is fitted to all supplied records and is not a training-only transform. An optional `--reference saved_matrix.npy` checks numerical agreement before writing outputs.

Run `python pipelines/test_native_cp.py -v` for alignment and input-validation tests.

## Expression profiles

`expression.py` matches Broad compound IDs and cell type to Level-3 L1000 instances, averages the matched profiles, and preserves the GCTX gene order. See [expression inputs and outputs](L1000_PROVENANCE.md).

## Training and retrieval

The [shell workflow](SHELL_WORKFLOW.md) provides commands to assemble processed training inputs, encode image conditions, aggregate crop features, and evaluate ID-aligned retrieval tasks. Training assembly requires the complete annotations and partitions in [DATA.md](DATA.md); preprocessing commands do not infer these from images alone.
