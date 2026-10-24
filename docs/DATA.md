# Input data

Download original data from the providers in [DATA_SOURCES.md](DATA_SOURCES.md). Raw images and large processed arrays are not redistributed in this repository. Cell Painting images and L1000 profiles are matched through compound metadata. See [PREPROCESSING.md](PREPROCESSING.md) for processing routes. Public plate directories corresponding to the local `cpg0016` processing inputs are listed in [image_plate_sources.csv](../manifests/image_plate_sources.csv); per-well/crop selections remain separate metadata.

## Training root

The `--project-root` directory must contain `data/processed/` with the following layout. File names preserve the original recipe to keep splits, offsets, and labels traceable.

```
data/processed/
  crops/
    crops_cpg0012_690.h5
    crops_cpg0004_a549x1055.h5
    crops_cpg0016_A375.h5
    crops_cpg0016_MCF7.h5
    crops_cpg0016_HT29.h5
  crops_cpg0030_u2os.h5
  meta_cpg0030_u2os.csv
  features/
    l1000_u2os_690.npy
    l1000_a549_1039.npy
    l1000_a375_86.npy
    l1000_mcf7_85.npy
    l1000_ht29_78.npy
  metadata/
    L1000_JumpCP_meta.csv
    l1000_a549_1039_meta.csv
    cpg0016_A375_meta.csv
    cpg0016_MCF7_meta.csv
    cpg0016_HT29_meta.csv
    cpg0012_profiles_merged.csv
    compoundinfo_beta.txt
    cpg0030_train_split.csv
    cpg0030_eval_split.csv
    crispr_gene_split_v8_90_40_seed42.csv
    u2os_cpg0012_train473_site_split_seed42.csv
    SingleCell_Manifest_JumpCP_690.csv
  cpg0000_crispr/
    meta_cpg0000_A549.csv
    meta_cpg0000_U2OS.csv
  cpg0000_compound/
    meta_cpg0000_cpd_A549.csv
    meta_cpg0000_cpd_U2OS.csv
  retraining_candidate_v1/
    complete.json
    cpg0000_crispr/
      crops_cpg0000_A549.h5
      crops_cpg0000_U2OS.h5
      meta_cpg0000_A549.csv
      meta_cpg0000_U2OS.csv
    cpg0000_compound/
      crops_cpg0000_cpd_A549.h5
      crops_cpg0000_cpd_U2OS.h5
      meta_cpg0000_cpd_A549.csv
      meta_cpg0000_cpd_U2OS.csv
```

This list is the required input contract, not a bundled dataset. The archived loaders validate additional source columns and identities; preserve original processed metadata rather than fabricating it. The training program hashes `complete.json` for provenance, and may materialize the gene split and temporary cache information under the supplied root.

## Shapes and identifiers

- Crop HDF5 files contain `crops`, a uint16 array `[N,5,128,128]`, and usually `labels`, an integer array `[N]` linking crop rows to the source metadata. The cpg0030 file contains only `crops`; `meta_cpg0030_u2os.csv` supplies row-aligned `int_label` values instead.
- L1000 arrays contain float32 profiles `[records,12328]`. Zero placeholder rows do not represent measured expression.
- Compound metadata use source-specific `new_int_label` or `int_label`, compound identifiers/names, and pipe-delimited `moa` strings. Genetic metadata contain `int_label` and `gene`.
- Gene partitions contain 130 genes shared across A549/U2OS, with 90 training genes and 40 held-out genes. The reference CSV is included under `splits/`.
- Whole-site U2OS exclusions depend on plate, well, site and label columns in both the manifest and the site split. An 80:20 crop split alone does not reconstruct this exclusion.

Cell-specific records are not globally compound-disjoint across all cells. HT29 is not used for encoder training. Auxiliary expression profiles are averaged by compound and cell type, without requiring matching dose or exposure time across assays.

## Retrieval NPZ

`biomoax evaluate` reads:

| Key | Shape | Meaning |
|---|---|---|
| `query` | `[Q,D]` | Finite nonzero query embeddings |
| `gallery` | `[G,D]` | Finite nonzero gallery embeddings |
| `positive` | `[Q,G]`, boolean | Valid positive links for the task |
| `excluded` | `[Q,G]`, boolean, optional | Gallery entries to omit for each query |

Same-cell retrieval excludes the query record. Drug MOA positives require nonempty annotation intersection. CRISPR positives consist of target-annotated anchors and one expansion to drugs sharing their MOA. A query contributes only if it has a positive after exclusions. Rank ties use stable input order. Percentages are on a 0–100 scale.

## Baseline training NPZ

`biomoax ridge-fit` reads `image: [N,D_image]` and `expression: [N,12328]`. Include only training pairs, pooled over U2OS/A549. The paper used 1,211 DeepProfiler pairs and 1,210 CellProfiler pairs after omitting one missing CellProfiler training pair. Native baseline profiles and matched-crop figure features are different inputs. The CLI does not silently infer missing-feature or training partitions.
