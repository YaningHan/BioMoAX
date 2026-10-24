# Available partitions

Fixed data partitions used in the paper.

- `crispr_gene_split.csv`: shared gene train/held-out assignment.
- `u2os_training_site_split.csv`: site-level crop allocation for the 473 training drug records; not the complete U2OS compound train/evaluation partition.
- `cross_batch_train.csv` and `cross_batch_eval.csv`: original cpg0030 partition records. Their labels belong to that source, not to cpg0012.

These files preserve the identifiers used in the experiments. The training input layout uses the filenames listed in `docs/DATA.md`; these reference partitions are not automatically installed into a training tree.

Core A549/U2OS and auxiliary-cell compound partitions, processed image–expression row mappings, and retrieval task lists are under `manifests/`. Raw expression sample selections are not included in these partition files.
