"""BioMoAX v12: A549-only CRISPR supervision on the original drug data.

This controlled run keeps the v10 drug recipe unchanged while removing every
U2OS CRISPR image from training.  The shared deterministic 90/40 gene split is
retained for A549: 90 A549 genes train the auxiliary CRISPR losses and 40 are
held out.  All 130 U2OS CRISPR genes are held out, enabling two post-hoc test
slices:

* 90 seen genes in an unseen CRISPR cellular context;
* 40 unseen genes in an unseen CRISPR cellular context.

The 912-compound expansion is deliberately disabled.
"""

from __future__ import annotations
import os

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import train_moax_v7 as base
import train_moax_v8_crispr130 as v8


ROOT = Path(os.environ['BIOMOAX_PROJECT_ROOT'])
SITE_SPLIT = (
    ROOT / "data/processed/metadata/"
    "u2os_cpg0012_train473_site_split_seed42.csv"
)
U2OS_MANIFEST = (
    ROOT / "data/processed/metadata/SingleCell_Manifest_JumpCP_690.csv"
)


def heldout_u2os_source_rows() -> np.ndarray:
    """Return H5 row indices belonging to protocol-held-out whole sites."""
    split = pd.read_csv(SITE_SPLIT)
    query = split.loc[split["split"].eq("query")].copy()
    manifest = pd.read_csv(U2OS_MANIFEST)
    key_cols = ["new_int_label", "plate", "well", "site"]
    for frame in (query, manifest):
        frame["new_int_label"] = frame["new_int_label"].astype(int)
        frame["site"] = frame["site"].astype(int)
        frame["plate"] = frame["plate"].astype(str)
        frame["well"] = frame["well"].astype(str)
    query_keys = pd.MultiIndex.from_frame(query[key_cols])
    manifest_keys = pd.MultiIndex.from_frame(manifest[key_cols])
    rows = np.flatnonzero(manifest_keys.isin(query_keys)).astype(np.int64)
    expected = int(query["n_crops"].sum())
    if len(rows) != expected or expected != 34841:
        raise RuntimeError(
            f"Expected 34,841 site-held-out U2OS crops, got {len(rows)} "
            f"(CSV sum={expected})"
        )
    labels = set(manifest.iloc[rows]["new_int_label"].astype(int))
    if len(labels) != 473:
        raise RuntimeError(f"Expected held-out sites for 473 compounds, got {len(labels)}")
    return rows


def configure(seed: int) -> tuple[set[str], set[str], Path]:
    run_name = f"moax_v12_siteholdout_a549_crispr_only_seed{seed}"
    log_dir = ROOT / "logs" / run_name
    # Reuse the directory name from the aborted pre-site-holdout run.  The
    # disk-backed reader recreates it from scratch, so no invalid cache data
    # survive and no additional ~57 GB directory is leaked.
    memmap_dir = ROOT / "data/tmp" / f"moax_v12_a549_crispr_only_seed{seed}"
    split = v8.build_or_validate_split()
    v8.configure_base(split)

    train_genes = set(split.loc[split["split"].eq("train"), "gene"].astype(str))
    test_genes = set(split.loc[split["split"].eq("test"), "gene"].astype(str))

    # Controlled CRISPR-cell ablation.
    base.HELD_OUT_GENES_A549 = test_genes
    base.HELD_OUT_GENES_U2OS = "ALL"
    heldout_rows = heldout_u2os_source_rows()
    base.EXCLUDED_SOURCE_ROWS_BY_CELL = {"U2OS": heldout_rows}

    # Reproduce v10 on the original drug pool only.
    base.TRAINING_EXPANSION = {}
    base.L1000_AVAILABLE_IDS = None
    base.BATCH_DRUGS = 1000
    base.K_PER_EPOCH = 300
    base.N_CROPS_BATCH = 1
    base.USE_MOA_BALANCED_SAMPLER = True
    base.MIN_DRUGS_PER_MOA = 2
    base.MOA_BALANCED_FRACTION = 0.5
    base.EPOCHS = 60
    base.RANDOM_SEED = seed

    # The dual-cell v10 batch contributed roughly 24 gene-i2i samples from
    # each cell line.  Keeping 24 here avoids doubling A549 supervision when
    # U2OS is removed.  CRISPR->L1000 was already A549-only in v10.
    base.GENE_I2I_PER_BATCH = 24
    base.CRISPR_I2E_PER_BATCH = 48

    base.LOG_DIR = log_dir
    v8.MEMMAP_DIR = memmap_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    # v7 applies its default seed at import time.  Reseed after configuration
    # so the two runs differ only in stochastic training, not in data splits.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    config = {
        "run": run_name,
        "seed": seed,
        "parent_recipe": "v10_b1000",
        "training_expansion": 0,
        "drug_data": "original v10 only",
        "epochs": base.EPOCHS,
        "batch_drugs": base.BATCH_DRUGS,
        "steps_per_epoch": base.K_PER_EPOCH,
        "crispr": {
            "A549_train_genes": len(train_genes),
            "A549_heldout_genes": len(test_genes),
            "U2OS_train_genes": 0,
            "U2OS_heldout_genes": len(train_genes | test_genes),
            "shared_gene_split": str(v8.SPLIT_PATH),
            "gene_i2i_per_batch": base.GENE_I2I_PER_BATCH,
            "crispr_i2e_per_batch": base.CRISPR_I2E_PER_BATCH,
        },
        "u2os_seen_compound_site_holdout": {
            "compounds": 473,
            "heldout_crops": len(heldout_rows),
            "split_manifest": str(SITE_SPLIT),
            "used_for_fitting": False,
        },
    }
    (log_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return train_genes, test_genes, log_dir


def preflight(train_genes: set[str], test_genes: set[str]) -> None:
    if len(train_genes) != 90 or len(test_genes) != 40:
        raise RuntimeError(
            f"Expected A549 90/40 split, got {len(train_genes)}/{len(test_genes)}"
        )
    if train_genes & test_genes:
        raise RuntimeError("CRISPR train and held-out gene sets overlap")

    meta_a = pd.read_csv(v8.META_A549)
    meta_u = pd.read_csv(v8.META_U2OS)
    genes_a = set(meta_a["gene"].astype(str))
    genes_u = set(meta_u["gene"].astype(str))
    if genes_a != genes_u or genes_a != train_genes | test_genes:
        raise RuntimeError("A549/U2OS 130-gene universes do not match the split")
    if base.TRAINING_EXPANSION:
        raise RuntimeError("The 912-compound expansion must remain disabled")
    if base.HELD_OUT_GENES_U2OS != "ALL":
        raise RuntimeError("Every U2OS CRISPR gene must be held out")
    excluded = getattr(base, "EXCLUDED_SOURCE_ROWS_BY_CELL", {}).get("U2OS")
    if excluded is None or len(excluded) != 34841:
        raise RuntimeError("The whole-site U2OS holdout was not installed")


def print_configuration(
    train_genes: set[str], test_genes: set[str], seed: int, log_dir: Path
) -> None:
    print("=" * 76)
    print(f"v12 seed={seed}: original v10 drugs + A549-only CRISPR supervision")
    print("  drug data:              original v10 only (expansion=0)")
    print(f"  A549 CRISPR:            train={len(train_genes)} held-out={len(test_genes)}")
    print("  U2OS CRISPR:            train=0 held-out=130")
    print("  U2OS site holdout:       34,841 crops across 473 seen compounds")
    print(f"  gene-i2i samples/batch: {base.GENE_I2I_PER_BATCH}")
    print(f"  CRISPR-i2e/batch:       {base.CRISPR_I2E_PER_BATCH}")
    print(f"  drug batch/steps:       {base.BATCH_DRUGS}/{base.K_PER_EPOCH}")
    print(f"  epochs:                 {base.EPOCHS}")
    print(f"  split manifest:         {v8.SPLIT_PATH}")
    print(f"  output:                 {log_dir}")
    print("=" * 76)


def main(seed: int, check_only: bool = False) -> None:
    train_genes, test_genes, log_dir = configure(seed)
    preflight(train_genes, test_genes)
    print_configuration(train_genes, test_genes, seed, log_dir)
    if check_only:
        print("Preflight passed; training was not started.")
        return
    v8.install_disk_backed_reader()
    base.main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    main(seed=args.seed, check_only=args.check_only)
