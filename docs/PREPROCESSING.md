# Data preprocessing

Download inputs from [the original providers](DATA_SOURCES.md). Commands and input formats are described in [the pipeline guide](../pipelines/README.md); use separate [environments](../environments/README.md) for the image-processing tools.

## Image crops

`segment.py` applies illumination correction, segments the DNA channel with Cellpose, and extracts five-channel 128 × 128 crops. Supply a site manifest with image paths, illumination files, and condition labels. Output channel order is DNA, ER, RNA, AGP, Mito. Existing crop files must have an explicit compatible channel order; do not infer it from array shape.

## CellProfiler and DeepProfiler

`cellprofiler.py` measures each crop with the included CellProfiler pipeline and returns median nuclear features, nuclei counts, and validity flags. `deepprofiler.py` uses the external CellPainting-CNN model to produce a 1,280-dimensional vector per crop. Both preserve source-row mappings.

These commands produce crop-level features. Native site-level profiles use a different measurement and aggregation unit and should not be substituted with crop averages when reproducing the corresponding paper baselines.

## Expression profiles

`expression.py` matches Broad compound IDs and cell type to Level-3 L1000 instances, averages the matched profiles, and preserves the GCTX gene order. See [expression inputs and outputs](L1000_PROVENANCE.md).

## Training and retrieval

The [shell workflow](SHELL_WORKFLOW.md) provides commands to assemble processed training inputs, encode image conditions, aggregate crop features, and evaluate ID-aligned retrieval tasks. Training assembly requires the complete annotations and partitions in [DATA.md](DATA.md); preprocessing commands do not infer these from images alone.
