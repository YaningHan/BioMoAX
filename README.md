# BioMoAX

BioMoAX learns representations of Cell Painting images and L1000 expression profiles using compound correspondence, mechanism-of-action (MOA) annotations, and genetic–chemical image pairs. This package accompanies the current RECOMB manuscript.

## What is included

- The image and expression encoders, four projection heads, contrastive losses, and image normalization extracted from the paper training archive.
- A portable entry point to the archived five-loss training recipe, including fixed splits and validation random seeds.
- Checkpoint loading, profile encoding, cosine retrieval, and bidirectional ridge baselines.
- Summary tables underlying the main retrieval figures, checkpoint configuration, source hashes, and a synthetic runnable example.
- English and [Chinese documentation](README_zh.md), plus [data specifications](docs/DATA.md) and [reproduction notes](docs/REPRODUCTION.md).

This is a local upload candidate, not an already published repository. Large input data and pretrained weights are not included, and public download locations have not yet been assigned. Full training and complete paper-figure regeneration have not been rerun from this distribution. Current smoke-test results are recorded in `VERIFICATION.json`; earlier verification is preserved in `docs/VERIFICATION_20261017.json`. See [the upload checklist](docs/UPLOAD_CHECKLIST.md) before publishing.

## Repository layout

- `src/biomoax/`: model, encoding, retrieval metrics, ridge baselines, and CLI.
- `src/biomoax/legacy/`: portable archived training implementation.
- `configs/` and `splits/`: paper configuration and fixed genetic split.
- `examples/` and `tests/`: synthetic example and CPU tests.
- `results/manuscript_20261024/`: current manuscript numerical tables and selected MOA color mappings; see its README for column conventions.
- `docs/`: input specifications, reproduction scope, and upload checklist.

## Installation

Use Python 3.10 or newer in a fresh environment. Install a CUDA-compatible PyTorch build for GPU training, then install this package:

```bash
python -m pip install -e .
python -m biomoax.cli --help
```

The exact local versions used for verification are in `environment_verified.json`; this is an observed environment, not a claim that every listed version combination was used for the original training.

## Quick start without research data

```bash
python examples/synthetic_retrieval.py
biomoax evaluate --input runs/synthetic.npz --output runs/synthetic_metrics.json
python -m unittest discover -s tests -v
```

The example reports 100% Top-1 on two eligible synthetic queries; the third has no positive and is excluded. It tests the software interface and is not a biological result.

## Model and objectives

The image backbone is a 12-block ViT with width 384, six attention heads and 16-pixel patches. Inputs are five-channel 128×128 crops. The expression backbone maps 12,328 features through widths 1,024 and 512. Each modality has compound and MOA projection heads producing normalized 256-dimensional vectors.

The paper objective is:

```
L = 2 L_compound + 1.7 L_image_MOA + 3 L_expression_MOA
    + 2 L_cross_modal_MOA + L_gene_image
```

Expression embeddings are detached in the cross-modal MOA term. Genetic and chemical images share the image backbone and image MOA head. The paper recipe disables the CRISPR-to-expression loss. The legacy modules contain historical defaults that are overridden by the training entry point; use `biomoax train`, not individual legacy scripts.

## Training

Prepare the processed files described in `docs/DATA.md`, then run:

```bash
biomoax check-data --project-root /path/to/project_inputs
biomoax train --project-root /path/to/project_inputs --output runs/preflight --seed 43 --preflight
biomoax train --project-root /path/to/project_inputs --output runs/training --seed 43
```

Preflight runs two optimizer updates with reduced crop caches; it still requires the complete metadata and compatible crop files. Training requires a CUDA GPU with BF16 support and at least 100 GiB of host RAM under the archived memory checks. The schedule is 60 epochs of 300 updates. Existing run configurations are protected against overwrite; automatic resume is not provided by this entry point.

## Inference and retrieval

For a trusted paper checkpoint, encode all crops for **one condition** into one mean profile:

```bash
biomoax encode --checkpoint checkpoints/seed43_ep029.pt --input one_condition_crops.npy --modality image --head moa --device cuda --output runs/profile.npy
```

Crop input has shape `[N,5,128,128]`, in the original uint16 intensity convention. This command does not group crops from different conditions; run it per condition. Expression input has shape `[N,12328]` and produces one embedding per row. Historical checkpoints contain Python/NumPy RNG state and require trusted-source loading.

`biomoax evaluate` accepts an NPZ containing `query`, `gallery`, boolean `positive`, and optional boolean `excluded`. Construct positives and same-record exclusions from metadata before calling it. See `docs/DATA.md` for shapes and conventions. This explicit interface also supports CRISPR-to-drug retrieval using the paper's target/MOA positive definitions.

## Ridge baseline

```bash
biomoax ridge-fit --input training_pairs.npz --output runs/ridge.joblib
```

The input has `image` and `expression` matrices containing training pairs only. The two directions are fitted independently with alpha 1, an intercept, and training-fitted StandardScaler transformations. MOA labels are not used for fitting. The caller supplies the documented CellProfiler or DeepProfiler input profiles, paired rows, and missing-feature handling.

## Paper results and scope

`results/` contains exported tables, not outputs from the synthetic example. Files directly under `results/` are historical exports; the current manuscript snapshot is in `results/manuscript_20261024/`. Main model summaries cover seed/epoch pairs 43/29, 45/28 and 46/28. The RTK network uses 45/28; the PDE network uses 42/28. Network coordinates are graph layouts of retrieval results. HT29 retrieval includes the 71 annotated drugs with a shared-MOA gallery positive. The current visualization uses an A549 reference map and separately transforms U2OS and HT29 profiles.

Model error bars describe variation between checkpoint means. Baseline retrieval intervals describe query-bootstrap uncertainty; batch-mixing baseline errors use paired jackknife standard errors. These are different sources of uncertainty.

To visualize the exported cross-cell means and CRISPR curves without raw images:

```bash
python scripts/plot_result_tables.py --output runs/table_plots
```

These compact plots check the tables and do not regenerate the composed manuscript figures.

## Distribution and licensing

The original project does not supply a confirmed top-level software license. No new license is assigned in this review package. Confirm code ownership, third-party terms, and the intended license before public release. No public repository URL, DOI, or checkpoint download link has been invented. The Chinese manuscript is an internal language counterpart; this archive contains code and result tables rather than manuscript PDFs.
