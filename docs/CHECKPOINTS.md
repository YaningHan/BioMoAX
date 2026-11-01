# Paper checkpoints

The retained checkpoints are `seed43_ep029`, `seed45_ep028` and `seed46_ep028`. Together they occupy 435,001,736 bytes. They were selected after validation and downstream comparisons from runs with five training seeds; they are not three prespecified independent replicates.

`manifests/checkpoints.json` gives filenames, byte counts and SHA-256 hashes. URLs remain null until weight assets are published. The source repository contains no model weights. Third-party baseline weights must be obtained from their original providers under their respective terms.

```bash
python scripts/check_checkpoints.py --project-root /path/to/data-root
```

See `configs/paper.json` for panel assignments. Figure 2B uses seed46; Figure 3B uses crop draw 9 of seed46, selected by the mean of four mixing scores among ten draws. Figure 5B uses seed45 and Figure 5C seed43. Other main retrieval summaries use the three retained checkpoints as recorded in the configuration.
