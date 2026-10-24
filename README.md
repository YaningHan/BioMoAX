# BioMoAX

BioMoAX learns representations of Cell Painting images and L1000 expression profiles using compound correspondence, mechanism-of-action (MOA) annotations, and genetic–chemical image pairs.

## What is included

- Image and expression encoders, four projection heads, contrastive losses, and image normalization.
- A training entry point with the paper's fixed data splits and evaluation settings.
- Checkpoint loading, profile encoding, cosine retrieval, and bidirectional ridge baselines.
- Summary tables underlying the main retrieval figures, run configuration, and a synthetic runnable example.
- [Data specifications](docs/DATA.md), [pipeline instructions](docs/SHELL_WORKFLOW.md), and [reproduction notes](docs/REPRODUCTION.md).
- [Native CellProfiler profile alignment](docs/PREPROCESSING.md#native-cellprofiler-profile-tables), with fixed feature order and coverage masks.

Repository: https://github.com/YaningHan/BioMoAX. Download source data from the original providers listed in [data sources](docs/DATA_SOURCES.md); raw images and large processed arrays are not redistributed here. See [preprocessing](docs/PREPROCESSING.md) for processing instructions.

## Repository layout

- `src/biomoax/`: model, encoding, retrieval metrics, ridge baselines, and CLI.
- `src/biomoax/legacy/`: training implementation used by the CLI.
- `configs/` and `splits/`: paper configuration and fixed genetic split.
- `manifests/`: core drug partitions, processed image/expression row mappings and MOA query/gallery lists.
- `examples/` and `tests/`: synthetic example and CPU tests.
- `scripts/`: Bash entry points for preprocessing, training, encoding, and evaluation.
- `pipelines/`: [Cellpose, CellProfiler and DeepProfiler commands](pipelines/README.md), with explicit route scope and external dependencies.
- `environments/`: dependency lists for the core package and preprocessing tools.
- `results/paper/`: current manuscript numerical tables and selected MOA color mappings; see its README for column conventions.
- `docs/`: data sources, preprocessing, input specifications, and reproduction notes.

## Installation

See [environment setup](environments/README.md) for module-specific dependencies.

Use Python 3.10 or newer in a fresh environment. Install a CUDA-compatible PyTorch build for GPU training, then install this package:

```bash
git clone https://github.com/YaningHan/BioMoAX.git
cd BioMoAX
python -m pip install -e .
python -m biomoax.cli --help
```

Module-specific dependencies are listed under `environments/`.

## Quick start without research data

```bash
python examples/synthetic_retrieval.py
biomoax evaluate --input runs/synthetic.npz --output runs/synthetic_metrics.json
python -m unittest discover -s tests -v
```

The synthetic example demonstrates the retrieval interface and positive/exclusion masks without downloading research data.

## Data and workflow

1. Download inputs from the providers listed in [data sources](docs/DATA_SOURCES.md).
2. Follow [preprocessing](docs/PREPROCESSING.md) and the [input specifications](docs/DATA.md) to prepare images, expression profiles, and metadata.
3. Use the [shell workflow](docs/SHELL_WORKFLOW.md) to train, encode profiles, and evaluate retrieval. CSV templates are in [examples/inputs](examples/inputs/README.md).

Model architecture, loss weights, and task-specific evaluation rules are described in [method and evaluation details](docs/REPRODUCTION.md).

## Training

For per-module Bash launchers and centralized environment/path configuration, see [the shell workflow](docs/SHELL_WORKFLOW.md).

Prepare the processed files described in `docs/DATA.md`, then run:

```bash
biomoax check-data --project-root /path/to/project_inputs
biomoax train --project-root /path/to/project_inputs --output runs/preflight --preflight
biomoax train --project-root /path/to/project_inputs --output runs/training
```

Training uses a reproducible default; pass `--seed` to change the training random seed. Paper-run settings and checkpoint provenance are documented in [reproduction notes](docs/REPRODUCTION.md#randomness-and-paper-runs).

Preflight runs two optimizer updates with reduced crop caches and requires the complete metadata and compatible crop files. Training requires a CUDA GPU with BF16 support and at least 100 GiB of host RAM. The schedule is 60 epochs of 300 updates. Use a new output directory for each run; automatic resume is not provided by this entry point.

## Inference and retrieval

Specify a model checkpoint to encode all crops for **one condition** into one mean profile:

```bash
biomoax encode --checkpoint /path/to/model.pt --input one_condition_crops.npy --modality image --head moa --device cuda --output runs/profile.npy
```

Crop input has shape `[N,5,128,128]` with uint16 intensities. For multiple conditions, use the [batch encoding command](docs/SHELL_WORKFLOW.md#condition-features). Expression input has shape `[N,12328]` and produces one embedding per row. Load checkpoints only from trusted sources.

For identity retrieval, use `--head compound --per-crop` to export one image embedding per crop, preserving input order. See [the identity workflow](docs/SHELL_WORKFLOW.md#drug-identity) for the subsequent crop splits and evaluation.

`biomoax evaluate` accepts an NPZ containing `query`, `gallery`, boolean `positive`, and optional boolean `excluded`. Construct positives and same-record exclusions from metadata before calling it. See `docs/DATA.md` for shapes and conventions. This explicit interface also supports CRISPR-to-drug retrieval using the paper's target/MOA positive definitions.

## Ridge baseline

```bash
biomoax ridge-fit --input training_pairs.npz --output runs/ridge.joblib
```

The input has `image` and `expression` matrices containing training pairs only. The two directions are fitted independently with alpha 1, an intercept, and training-fitted StandardScaler transformations. MOA labels are not used for fitting. The caller supplies the documented CellProfiler or DeepProfiler input profiles, paired rows, and missing-feature handling.

Use `scripts/13_ridge_predict.sh` to map queries and target-gallery profiles before retrieval. `scripts/14_identity_retrieval.sh` implements repeated disjoint crop splits for drug identity. Commands are in [the shell workflow](docs/SHELL_WORKFLOW.md#retrieval), with CSV schemas in [input templates](examples/inputs/README.md).

## Paper results and scope

Paper result tables are in `results/paper/`; see its README for column definitions and uncertainty conventions. Run settings and checkpoint mappings are recorded in `configs/paper.json`.

To visualize the current paper A549 identity summary and CRISPR curves without raw images:

```bash
python scripts/plot_result_tables.py --output runs/table_plots
```

This command produces standalone plots from the result tables.

## Distribution and licensing

BioMoAX code is licensed under the [MIT License](LICENSE). Third-party code, assets, datasets, and models remain subject to their respective licenses and terms.
