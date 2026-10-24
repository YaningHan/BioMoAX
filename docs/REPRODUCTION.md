# Reproduction and source mapping

## Source layers

`src/biomoax/model.py` contains the model definitions. `metrics.py`, `baselines.py`, `inference.py` and `cli.py` provide the public interfaces. `legacy/` contains the training implementation used by `biomoax train`.

The training chain is:

```
CLI → train_rebuilt_five_loss → fixed validation configuration
    → crop configuration → base model and data loaders
    → five-loss gate + resident RAM reader → training loop
```

The training entry applies the 90/40 gene split to both cells, uses resident uint16 caches, disables CRISPR-to-expression training and external test evaluation inside the training loop, and checks finite gradients.

## Randomness and paper runs

The public training command exposes one `--seed` option (default: 43). It controls training randomness, not the fixed data split or the validation sampling streams. Omitting it in the quick-start commands preserves the original default behavior.

`configs/paper.json` records training seeds, the fixed data seed, validation seeds, and checkpoint-to-result mappings. It is a reference file, not a configuration automatically loaded by the CLI.

For repeated training, choose a seed from the recorded training list and a separate output directory for each run. Main-result and network checkpoints are identified in the configuration.

## Paper evaluation

| Task | Query/gallery | Positive rule |
|---|---|---|
| Within cell | Held-out record queries, same-cell gallery | Shared MOA, query record excluded |
| Cross cell | U2OS → A549 and A549 → U2OS | Shared MOA |
| Cross batch | 32 shared-compound and 115 MOA-only cpg0030 queries → 690 cpg0012 drugs | Shared MOA in both groups |
| Unseen cell | 71 eligible HT29 queries → pooled U2OS/A549 | Shared MOA |
| Genetic to chemical | 29 eligible held-out A549 genes → 1,038 drugs | Target anchors plus one MOA expansion |

For each condition the original evaluation samples up to 100 crops, projects each crop, averages projected embeddings, and normalizes the mean. `encode` performs this aggregation on the crops supplied by the caller; it does not reproduce the original random draws automatically. Preparing paper-specific query/gallery arrays, baseline native profiles and positive masks remains necessary before `evaluate`.

Results are aggregated across three checkpoints after averaging crop draws within each checkpoint. Table columns and uncertainty conventions are described in `results/paper/README.md`.

