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

Remaining source mapping: the local A375/MCF7/HT29 files are labeled `cpg0016`, but their exact external plate/object mapping must still be established. The exact L1000 release, file identifiers and expression row order also need a pinned manifest. A portal URL alone does not specify the paper input subset.

Retain sample identifiers, plate/well/site information, compound/gene mappings, channel order, train/evaluation splits, and checksums alongside downloaded files. These small metadata artifacts are distinct from the raw data and are required for faithful reproduction. The included genetic split is under `splits/`; the complete input contract is in [DATA.md](DATA.md).
