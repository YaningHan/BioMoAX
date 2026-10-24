# Reproduction and source mapping

## Source layers

`src/biomoax/model.py` is extracted from the archived model definitions, preserving parameter names and forward calculations. `metrics.py`, `baselines.py`, `inference.py` and `cli.py` provide focused public interfaces. `legacy/` preserves the original training chain, including its historical file names. `SOURCE_MANIFEST.json` gives hashes and adaptations for each exported source and result table.

The training chain is:

```
CLI → train_rebuilt_five_loss → fixed validation configuration
    → crop configuration → base model and data loaders
    → five-loss gate + resident RAM reader → training loop
```

The top-level entry overrides historical auxiliary-loss and held-out gene defaults. It installs the 90/40 split for both cells, uses resident uint16 caches, disables CRISPR-to-expression training and external test evaluation inside the training loop, and checks finite gradients. Source transformations used to disable branches are guarded by exact-source assertions inherited from the experiment. This structure is retained for traceability; the public model module can be read independently.

## Paper evaluation

| Task | Query/gallery | Positive rule |
|---|---|---|
| Within cell | Held-out record queries, same-cell gallery | Shared MOA, query record excluded |
| Cross cell | U2OS → A549 and A549 → U2OS | Shared MOA |
| Cross batch | 32 shared-compound and 115 MOA-only cpg0030 queries → 690 cpg0012 drugs | Shared MOA in both groups |
| Unseen cell | 71 eligible HT29 queries → pooled U2OS/A549 | Shared MOA |
| Genetic to chemical | 29 eligible held-out A549 genes → 1,038 drugs | Target anchors plus one MOA expansion |

For each condition the original evaluation samples up to 100 crops, projects each crop, averages projected embeddings, and normalizes the mean. `encode` performs this aggregation on the crops supplied by the caller; it does not reproduce the original random draws automatically. Preparing paper-specific query/gallery arrays, baseline native profiles and positive masks remains necessary before `evaluate`.

Results are aggregated across three recorded checkpoints after averaging crop draws within each checkpoint. The exported tables preserve their original column conventions; inspect headers because some archived tables use `mean` or `value`, and MRR columns may remain in historical exports although they are not displayed in the current main figures.

## Available and separate artifacts

The source package permits model/loss inspection, synthetic testing, inference with supplied weights, feature-level evaluation, baseline fitting, and execution of the archived training entry with prepared data. It includes numerical figure sources. It does not provide a one-command raw-data download/segmentation pipeline or all historical figure renderers. Pretrained weights, processed inputs, native CP/DP feature generation, full plotting dependencies, and public hosting must be supplied separately for end-to-end paper reproduction.

The original numerical experiments remain the evidence for paper performance. CPU smoke tests and source parity checks validate the release interfaces; they do not establish independent reproduction of training outcomes.
