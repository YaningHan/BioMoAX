# BioMoAX

BioMoAX learns a shared functional space for Cell Painting images, L1000 expression profiles and genetic perturbations. Five complementary contrastive objectives connect compound identity, shared mechanisms of action and gene–drug image pairs.

This repository contains training and inference code, ridge baselines, the current English manuscript, numerical results, and a self-contained figure/statistics reproduction package. Raw training data and model weights are separate from Git source.

## Installation and checks

```bash
python -m pip install -e '.[reproduce]'
python -m unittest discover -s tests -v
python examples/synthetic_retrieval.py
```

Use `biomoax --help` for training, encoding, retrieval and ridge commands. See [data specifications](docs/DATA.md), [preprocessing](docs/PREPROCESSING.md), the [shell workflow](docs/SHELL_WORKFLOW.md), and the [release validation record](docs/VALIDATION.md).

## Reproduce the paper

```bash
python reproduce/run.py statistics
python reproduce/run.py figures
python reproduce/run.py paper
```

Outputs are written under `reproduce/build/`; bundled inputs and manuscript sources remain unchanged. Paper compilation requires pdfLaTeX and latexmk. The complete sequence is `python reproduce/run.py all`.

The bundled observations, UMAP coordinates and small image examples reproduce the figures and statistical calculations without downloading training data or checkpoints. They do not rerun feature extraction, UMAP fitting or training. See [reproduction instructions](reproduce/README.md).

`paper/main.tex` is the current English manuscript, using the supplied LNCS class, one-inch margins and splncs04 bibliography. Its five approved PDF figures are included. Redrawing uses installed Liberation Sans or DejaVu Sans; font glyphs can differ from the approved figures, which contain embedded Arial subsets.

## Pretrained model

Download the [BioMoAX pretrained model](https://github.com/YaningHan/BioMoAX/releases/tag/weights-v1.0) and encode your image crops:

```bash
python scripts/download_checkpoint.py
biomoax encode --input crops.npy --modality image --output embeddings.npy
```

The download is verified automatically using its file size and SHA-256. Encoding uses the pretrained model and MOA head by default. Image input is a uint16 array of shape `[N, 5, 128, 128]`; the output is a normalized 256-dimensional condition embedding. The model also encodes expression profiles. See [model usage](docs/CHECKPOINTS.md) for input details and [evaluation protocols](docs/REPRODUCTION.md) for the paper settings.

## Training data

```bash
biomoax check-data --project-root /path/to/data-root
biomoax train --project-root /path/to/data-root --output /path/to/run --preflight
```

Training uses the static modules in `src/biomoax/training/`. It requires the processed input contract, CUDA with BF16 support and sufficient host memory. Preflight performs two updates; it is not a full training run. See the CLI help for exact arguments and [protocol notes](docs/REPRODUCTION.md) for data splits, checkpoint selection, uncertainties and baseline feature units.

## Repository contents

| Directory | Contents |
|---|---|
| `src/biomoax/` | Encoders, losses, inference, metrics, ridge and static training modules |
| `configs/`, `splits/`, `manifests/` | Paper settings, record/gene splits and checkpoint hashes |
| `results/paper/` | Current numerical summaries and Wilcoxon tables |
| `reproduce/` | Fixed observations, figure assets, rendering and statistical verification |
| `paper/` | Current English manuscript and approved figures |
| `docs/`, `pipelines/`, `scripts/` | Input preparation and execution instructions |

Code is distributed under the existing MIT license. Third-party datasets, pretrained models and fonts retain their own terms; see [third-party notes](THIRD_PARTY.md). No third-party weights or complete proprietary fonts are bundled.
