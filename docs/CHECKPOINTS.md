# Paper checkpoints

The retained checkpoints are `seed43_ep029`, `seed45_ep028` and `seed46_ep028`. Together they occupy 435,001,736 bytes. They were selected after validation and downstream comparisons from runs with five training seeds; they are not three prespecified independent replicates.

`manifests/checkpoints.json` gives filenames, byte counts and SHA-256 hashes. The public model release distributes **seed46_ep028** only: [download](https://github.com/YaningHan/BioMoAX/releases/download/weights-v1.0/seed46_ep028.pt), [release notes and model configuration](https://github.com/YaningHan/BioMoAX/releases/tag/weights-v1.0). It includes the image and expression encoders and both projection-head pairs. Load it with `biomoax.inference.load_models(path, head="moa")` or `head="compound"`.

```bash
python scripts/download_checkpoint.py --project-root /path/to/data-root
python scripts/check_checkpoints.py --project-root /path/to/data-root --id seed46_ep028
```

The other two retained checkpoints are documented here for paper provenance and have no public download configured in this release. A single checkpoint does not reproduce three-checkpoint averages or case networks drawn from seed43/seed45. The fixed-observation figure/statistics package reproduces the published summaries without checkpoint downloads. Third-party baseline weights must be obtained from their original providers under their respective terms.

```bash
python scripts/check_checkpoints.py --project-root /path/to/data-root
```

See `configs/paper.json` for panel assignments. Figure 2B uses seed46; Figure 3B uses crop draw 9 of seed46, selected by the mean of four mixing scores among ten draws. Figure 5B uses seed45 and Figure 5C seed43. Other main retrieval summaries use the three retained checkpoints as recorded in the configuration.
