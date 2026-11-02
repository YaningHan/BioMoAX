# Pretrained model

The [BioMoAX pretrained model](https://github.com/YaningHan/BioMoAX/releases/tag/weights-v1.0) includes the image and expression encoders and their compound/MOA projection heads. The weight file is approximately 138.3 MiB. Its architecture is described in the release's `model_config.json`; file size and SHA-256 are recorded in `manifests/checkpoints.json`.

From the repository root:

```bash
python scripts/download_checkpoint.py
biomoax encode --input crops.npy --modality image --output image_embedding.npy
biomoax encode --input expression.npy --modality expression --output expression_embeddings.npy
```

Image input is a uint16 NumPy array with shape `[N, 5, 128, 128]`. Crops from one condition are averaged to produce a normalized `[1, 256]` embedding. Expression input has shape `[N, 12328]` and produces `[N, 256]` embeddings. Both commands use the same pretrained model and default MOA head. Add `--device cuda` for GPU inference. Use `--head compound` for compound-identity representations.

The Python interface also uses the pretrained model by default:

```python
from biomoax.inference import load_models, encode_condition

models = load_models()
embedding = encode_condition(models, crops, modality="image")
```

The downloaded file is saved under `checkpoints/`. To place it elsewhere, pass `--project-root /path/to/project` to the download script and `--checkpoint /path/to/model.pt` to the encoding command. To verify an existing download, run `python scripts/check_checkpoints.py`.

Paper checkpoint assignments and result aggregation are documented in [REPRODUCTION.md](REPRODUCTION.md), `configs/paper.json`, and `manifests/paper_checkpoints.json`. Third-party baseline weights are obtained from their original providers.
