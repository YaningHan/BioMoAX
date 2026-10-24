# Core drug records and MOA tasks

`drug_records.csv` links 1,978 processed drug/cell records to crop HDF5 labels and expression rows, with compound identifiers, MOA sets, split assignments, crop counts and expression availability. It describes processed records, not raw L1000 sample selections; generating expression arrays requires explicit source sample IDs and aggregation settings.

Partitions follow the shipped training implementation. The pure MOA split function is extracted from that source without importing or executing training. U2OS excludes Tocris records from training/query selection, while retaining them as gallery candidates. The 473 U2OS training labels are independently compared with the saved site split. A549 uses its original `int_label` for crop lookup and `new_int_label`/row index for expression lookup; these must not be conflated.

| Cell | Train | Held-out | Gallery-only |
|---|---:|---:|---:|
| U2OS | 473 | 90 | 127 |
| A549 | 738 | 301 | 0 |
| A375 | 80 | 6 | 0 |
| MCF7 | 79 | 6 | 0 |
| HT29 | 0 | 78 | 0 |

Paths in the CSV are relative to `PROJECT_ROOT/data/processed`. For the `local_cpg0016` inputs, `image_plate_sources.csv` lists 11 public image directories and their local cell labels. Exact well/crop selections and external cell annotations require separate alignment.

## Task lists

Each directory under `tasks/` contains ordered query/gallery IDs (with record metadata), positive pairs and excluded pairs. Positives use nonempty pipe-separated MOA intersection; same-record pairs are excluded for within-cell retrieval. No query is chosen using a model score. Counts are in `tasks/task_summary.json`:

- Within U2OS: 90 queries; 690 gallery rows, with the query record excluded for each query (689 candidates).
- Within A549: 301 queries; 1,039 gallery rows, with the query record excluded (1,038 candidates).
- U2OS → A549: 90 annotation-eligible queries.
- A549 → U2OS: 257 of the 301 held-out records have a gallery MOA positive.
- HT29 → pooled U2OS/A549: 71 of 78 records have a gallery MOA positive.

These are annotation-defined core tasks. Method-specific missing-feature coverage is not silently filtered. Missing/zero feature vectors are rejected by the feature selector and must be resolved with an explicit, documented coverage rule. Do not assume all directions use the full held-out query count.

Identity retrieval uses crop-level splitting, so these drug-level task files must not be substituted for it.

## Cross-batch and genetic tasks

`external_tasks/` contains the same query/gallery/pair interfaces for:

- `cross_batch_moa`: 115 queries against 690 U2OS records (886 positive pairs).
- `cross_batch_shared`: 32 held-out shared-compound queries against the same gallery (474 MOA-positive pairs). This is a separate task, not an identity-retrieval positive definition.
- `crispr_a549`: 29 eligible held-out genes against 1,038 drug records (341 positive pairs).

`cross_batch_crop_rows.csv` maps each cross-batch query to the rows in `crops_cpg0030_u2os.h5`. Query IDs follow alphabetical compound ordering within each match group. Use those IDs when preparing embedding pools.

CRISPR query metadata includes its source HDF5 gene label. `positive_evidence.csv` records direct-target membership, shared MOA labels and contributing anchor drug IDs. Anchors are defined against the full 1,039-row A549 collection; one-hop MOA expansion occurs before applying the frozen common-baseline gallery mask. Do not recompute expansion only after dropping gallery records. `heldout_coverage.csv` retains all 40 held-out genes, including those without positives.

`crispr_a549_gallery_keys.json` defines the finite/nonzero common-baseline gallery mask used for CRISPR evaluation. `full:N` refers to expression/source row N in A549, not a crop HDF5 label.

`heldout_coverage.csv` reports positive counts for all 40 held-out genes. These task files define retrieval pairs; they do not specify raw L1000 sample selections.

Rebuild from the processed inputs with `bash scripts/07_prepare_tasks.sh /absolute/path/config.env`. This requires those inputs; existing bundled CSVs can be used without rerunning the export. It does not download or regenerate the underlying image or expression arrays.
