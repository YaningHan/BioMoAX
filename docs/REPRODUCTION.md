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

## Baseline feature units

Crop-level features and native condition profiles serve different tasks. A shared method name does not imply identical input crops or aggregation across panels.

| Evaluation | CellProfiler inputs and calculation |
|---|---|
| Fig2A, drug identity | Matched crop features; five disjoint 20/80 query/gallery splits per drug. Aggregate each side by feature-wise median, replace nonfinite aggregate values using gallery medians (zero if no finite gallery value), L2-normalize, then rank by cosine similarity. |
| Fig2B and Fig4, image-to-image MOA retrieval | Native condition-level profiles with source-specific aggregation and the common feature columns; rank by cosine similarity. |
| Fig2B and Fig4, cross-modal retrieval | Raw native CP profiles paired with expression. Fit independent image-to-expression and expression-to-image ridge models with training-only standardization, alpha 1 and an intercept. |
| Fig3, cross-batch retrieval | Native cpg0030/cpg0012 profile baseline. Crop-level spatial displays are separate from this retrieval calculation. |
| Fig5A, CRISPR-to-drug retrieval | Align gene and drug profiles on shared CP columns. Fit RobustScaler to the full native drug gallery and apply it to both sides, then rank normalized profiles using the common eligible CP/DP gallery and target/MOA positives. |

For native U2OS CP profiles, compute per-well medians from cell measurements, then per-compound medians across wells; fill missing compound-feature values using the column median. The supplied [aggregation and alignment commands](PREPROCESSING.md#native-cellprofiler-profile-tables) start from well-level or condition-level tables. Native U2OS/A549 matrix reconstruction uses RobustScaler fitted to the supplied condition rows, including recorded zero placeholders. This transform is distinct from the training-only StandardScaler used by ridge baselines.

Auxiliary-cell native CP profiles use finite single-cell vectors averaged within each compound and plate, followed by an equal-weight average across available plates. This is not a crop-level median or a cell-count-weighted average across plates. Native DP profiles use source-specific mean aggregation; the generic crop extractor and aggregator should not be treated as an interchangeable replacement for those inputs.

Use crop-level CP/DP commands for matched-crop comparisons and spatial displays. Native-profile evaluations require their condition-level inputs and identifier mappings. Preserve explicit missing-feature masks rather than treating placeholder rows as measured profiles.

