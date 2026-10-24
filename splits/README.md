# Available partitions

These are preserved experiment records, not newly sampled partitions.

- `crispr_gene_split.csv`: shared gene train/held-out assignment.
- `u2os_training_site_split.csv`: site-level crop allocation for the 473 training drug records; not the complete U2OS compound train/evaluation partition.
- `cross_batch_train.csv` and `cross_batch_eval.csv`: original cpg0030 partition records. Their labels belong to that source, not to cpg0012.

Copied partition sources and hashes are in `partition_sources.json`. Stable release filenames do not change the identifiers inside the files. The legacy training contract still expects its documented original filenames; these exports are reference records, not automatically installed into a training tree.

Core A549/U2OS and auxiliary-cell compound partitions, processed image–expression row mappings, and core/external task lists are under `manifests/`. Exact raw-source download manifests and raw expression signature matching remain separate work. Do not infer those missing source lists from the gene split or site split.
