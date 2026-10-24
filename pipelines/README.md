# Data preprocessing

## L1000 expression

Use the core environment and a compound CSV with unique `record_id` and explicit `broad_id` columns. Rows must belong to the requested cell type.

```bash
bash scripts/08_extract_expression.sh --cell U2OS --records compounds.csv \
  --instinfo /path/to/instinfo_beta.txt \
  --gctx /path/to/level3_beta_trt_cp_n1805898x12328.gctx \
  --output runs/expression_u2os
```

Change `--cell` and the compound CSV for other cell types. The command averages same-cell compound instances without dose/time or QC filtering. It preserves input record order and the GCTX gene order, and saves `expression.npy`, availability flags, and sample mappings. Missing profiles are marked NaN, not measured zeros. Use a new output directory and retain the identifiers for downstream alignment. Check sample selections and numerical compatibility before using new arrays with the paper's processed inputs.

Test with `python pipelines/test_expression.py -v`.

## Image preprocessing

After extraction, use [condition-feature preparation](../docs/SHELL_WORKFLOW.md#condition-features) to encode image conditions or aggregate crop-level baseline vectors into ID-indexed retrieval pools.

These portable commands expose specific processing routes from the project. They do not automatically reconstruct every paper dataset. Download images, illumination arrays and third-party weights from [the original providers](../docs/DATA_SOURCES.md). Keep each tool in its own compatible environment; the core package does not install Cellpose, TensorFlow or CellProfiler.

| Entry | Input | Output | Scope |
|---|---|---|---|
| `segment.py` | Explicit site CSV, images, illumination arrays, Cellpose weights | Crops, labels, site ranges, cell centers | Chemical/genetic reconstruction route |
| `cellprofiler.py` | Channel-annotated crop HDF5 | Nuclei median features and QC masks | Matched-crop features, not native site-level tables |
| `deepprofiler.py` | Same crop HDF5, CellPainting-CNN weights | One 1,280-dimensional vector per crop | Audited min–max crop route |

Run these commands from the repository root. All output directories must be new. Failed runs may leave partial outputs; only a `run.json` with `status: complete` indicates completion. Automatic resume is not provided. Inspect outputs before deleting a failed directory.

## 1. Cellpose

This pipeline uses `cellpose==4.0.9` with NumPy, SciPy, h5py, Pillow and tifffile in a compatible PyTorch environment. Supply the downloaded `cpsam` weights explicitly; the script rejects a fallback model.

```bash
python pipelines/segment.py --manifest sites.csv --weights /path/to/cpsam --output runs/segmented --gpu
```

The CSV needs `plate,well,site,kind,label` plus `DNA,ER,RNA,AGP,Mito` and `illum_DNA,illum_ER,illum_RNA,illum_AGP,illum_Mito`. Image/illumination fields are paths relative to the CSV or absolute paths. Use one row per site. `kind` is `crispr` or `compound`; integer `label` must match your condition metadata. Include a separate manifest per cell/source when plate/well/site identifiers could collide.

CRISPR correction divides by illumination; compound correction also multiplies by the illumination median. DNA is percentile-normalized for segmentation; diameter is 40 for CRISPR and automatic for compound images. Crops retain uint16 intensities after clipping, use integer-truncated mask centroids, and exclude boundary-crossing windows. Output channels are DNA, ER, RNA, AGP, Mito. Zero-crop sites remain recorded and require review.

This command does not implement all earlier multicell `nuclei` routes or the separate expansion-PNG route. Already illumination-corrected images must not be fed through correction a second time.

## 2. CellProfiler

The wrapper requires NumPy, pandas, h5py and tifffile. The external measurement program was checked with CellProfiler 4.2.8.1. Install/configure Java as required by that environment; pass its executable explicitly when using a separate environment.

```bash
python pipelines/cellprofiler.py --input runs/segmented/crops.h5 --executable /path/to/cellprofiler --output runs/cp
```

`assets/crop.cppipe` defines the measurement pipeline and `assets/cp_columns.txt` selects the output features. The wrapper supplies the load-data CSV and output folder. Auxiliary brightfield names point to AGP for pipeline compatibility; output features use the five fluorescence channels.

`features.npy` preserves every input crop row. `nuclei_counts.npy` and `valid_rows.npy` flag measurement failures/nonfinite features; invalid rows are not silently zero-filled. Keep the same valid rows across compared methods. TIFFs and measurement CSVs are retained under batch directories for inspection, so plan disk space accordingly. No imputation or native-profile aggregation is implied by this crop command.

## 3. DeepProfiler

The tested model environment uses TensorFlow 2.13.1 and `efficientnet==1.1.1` with h5py and NumPy. Use the checkpoint-native `efficientnet.tfkeras` implementation; do not replace it with a different Keras backbone. The external weights are `Cell_Painting_CNN_v1.hdf5` from [Zenodo](https://zenodo.org/records/7114558).

```bash
python pipelines/deepprofiler.py --input runs/segmented/crops.h5 --weights /path/to/Cell_Painting_CNN_v1.hdf5 --output runs/dp
```

The extractor validates backbone weights and rejects missing or incompatible tensors. Crops are reordered from DNA, ER, RNA, AGP, Mito to DNA, RNA, ER, AGP, Mito, then min–max scaled per crop/channel. Features are not automatically averaged or L2-normalized; downstream condition aggregation must match the intended task. Min–max-scaled features are not interchangeable with percentile-scaled features.

## Validation and provenance

```bash
python pipelines/test_preprocessing.py -v
```

The four CPU tests cover channel order, scaling, illumination correction, crop boundaries, input contracts and overwrite protection.

For input sources and the distinction between crop-level features and native site-level profiles, see [preprocessing](../docs/PREPROCESSING.md).
