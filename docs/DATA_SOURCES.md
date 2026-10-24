# Data and third-party model sources

Download data directly from the original providers. This repository does not mirror raw images or large processed arrays. Follow each provider's terms and cite the source datasets.

| Resource | Official download entry | Use |
|---|---|---|
| Cell Painting Gallery | https://github.com/broadinstitute/cellpainting-gallery | Images, illumination files, dataset metadata and available profiles |
| LINCS / Connectivity Map | https://clue.io/data | L1000 expression profiles and compound annotations |
| CellPainting-CNN weights | https://zenodo.org/records/7114558 | External pretrained model for the DeepProfiler baseline; not BioMoAX weights |
| DeepProfiler | https://github.com/cytomining/DeepProfiler | Upstream software and profiling documentation |
| Cellpose | https://github.com/MouseLand/cellpose | Upstream segmentation software; use the compatible historical model/API, not an unverified replacement |

The local source records identify these Gallery prefixes:

| Local source | S3 prefix |
|---|---|
| U2OS compound images | `s3://cellpainting-gallery/cpg0012-wawer-bioactivecompoundprofiling/` |
| A549 compound images | `s3://cellpainting-gallery/cpg0004-lincs-sinewave/` |
| Cross-batch images | `s3://cellpainting-gallery/cpg0030-gustafsdottir-cellpaintingpilot/` |
| Chemical/genetic images | `s3://cellpainting-gallery/cpg0000-jump-pilot/` |

These prefixes come from the project source inventory; they are not a complete paper-specific download manifest. Use the Gallery catalog to inspect available objects before downloading. Do not recursively download the entire Gallery.

## Additional image plates

[image_plate_sources.csv](../manifests/image_plate_sources.csv) lists 11 plates present in the local A375/MCF7/HT29 processing inputs, with their public image-directory names. The S3 base is `s3://cellpainting-gallery/cpg0016-jump/source_4/`; image prefixes are `images/{batch}/images/{image_directory}/`. `local_cell` preserves the processing-file labels. This plate inventory is not a per-well or per-crop selection manifest.

For example, inspect the first listed plate before downloading selected objects:

```bash
aws s3 ls --no-sign-request \
  s3://cellpainting-gallery/cpg0016-jump/source_4/images/2021_06_21_Batch7/images/BR00121424__2021-06-29T07_13_33-Measurement1/
```

The download layout also uses `images/{batch}/illum/{plate}/` for illumination files and `workspace/load_data_csv/{batch}/{plate}/load_data_with_illum.csv` for load-data tables. Check object availability and retain the plate/well/site metadata when preparing your site CSV.

## L1000 files

Use the [CMap 2020 data page](https://clue.io/data/CMap2020) to locate these files by name:

| File | Purpose |
|---|---|
| `level3_beta_trt_cp_n1805898x12328.gctx` | Compound-treatment Level-3 expression matrix |
| `instinfo_beta.txt` | Instance/sample IDs and cell/compound annotations |
| `compoundinfo_beta.txt` | Compound, target and MOA annotations |
| `geneinfo_beta.txt` | Gene annotations; matrix column order comes from GCTX gene IDs |

The extraction command reads sample IDs from `instinfo_beta.txt` and gene order from the GCTX itself; see [expression processing](L1000_PROVENANCE.md). File names identify the source inputs, not a verified reconstruction of every archived paper array. Preserve explicit sample selections when generating new arrays.

Retain sample identifiers, plate/well/site information, compound/gene mappings, channel order, train/evaluation splits, and checksums alongside downloaded files. These small metadata artifacts are distinct from the raw data and are required for faithful reproduction. The included genetic split is under `splits/`; the complete input contract is in [DATA.md](DATA.md).
