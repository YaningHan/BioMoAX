# Reproduction and protocols

## Model and training

The image encoder is a 12-block ViT with width 384, six attention heads and 16-pixel patches, taking five-channel 128×128 crops. The expression encoder maps 12,328 features through widths 1,024 and 512. Independent compound and MOA heads produce normalized 256-dimensional embeddings for each modality.

The joint objective is `2 L_compound + 1.7 L_image_MOA + 3 L_expression_MOA + 2 L_cross_modal_MOA + L_gene_image`. Expression embeddings are detached for cross-modal MOA alignment. Genetic and chemical images share the image encoder and MOA head; CRISPR-to-expression loss is disabled.

The public CLI launches `biomoax.training.entry`. Static modules in `src/biomoax/training/` implement configuration, fixed validation, site splits, data readers, loss policy and the training loop. No historical script directory is required at runtime. Full training uses 60 epochs and 300 updates per epoch; the preflight executes two updates with capped resident crop caches. It requires CUDA BF16 support and checks available host memory.

## Data splits

Training expects the processed contract described in [DATA.md](DATA.md), rooted at `data/processed/training_inputs/`. The committed split CSVs identify cpg0030 training/evaluation records and the shared 90/40 training/evaluation gene split used in both cells.

cpg0030 contributes 154 training records. Cross-source MOA retrieval uses 115 queries whose compounds are excluded from training, against cpg0012 references; it is not an entirely unseen source. A separate 32-query shared-compound setting is used for mixing analysis. Keep these two evaluation groups distinct.

Within-cell drug-identity retrieval uses disjoint crop query/gallery sets. MOA retrieval excludes same-compound matches where specified. Cross-cell tasks use U2OS/A549 in both directions; HT29 supplies the unseen-cell queries. Genetic-to-chemical retrieval evaluates 29 eligible held-out A549 genes against 1,038 drugs using target anchors and one-hop MOA positives. Source-specific eligibility masks remain necessary.

## Checkpoint and query summaries

The three retained checkpoints were selected from five training seeds after validation and downstream comparisons. Their variation is not an estimate over unselected independent training runs. `configs/paper.json` records assignments and fixed sampling seeds, and `manifests/paper_checkpoints.json` records the corresponding weight hashes; the CLI does not automatically load this reference configuration.

Figure 2B uses seed46. Figures 3A, 4 retrieval and 5A summarize the retained checkpoints with their recorded crop draws. Figure 3B uses seed46 crop draw 9, selected by the highest mean of four mixing scores among ten draws; the display is not a mean over those draws. Figures 5B/C are selected illustrative cases from seed45/seed43 respectively.

The current statistical verification uses paired per-query observations for Wilcoxon tests. Multiple-comparison adjustment and the comparison families are recorded in the exported test tables and verification code. Query-bootstrap intervals condition on the evaluated representations and query population; they do not include retraining uncertainty. Preserve each panel's actual interval and aggregation convention rather than assigning one uncertainty model to every figure.

Encoding averages projected crop embeddings and normalizes the mean. The CLI processes the supplied crops; it does not automatically recover the original paper-specific crop sampling, native baseline profiles or positive masks. The bundled [fixed-input package](../reproduce/README.md) reproduces the statistical calculations and figures from their retained observations and coordinates.

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
