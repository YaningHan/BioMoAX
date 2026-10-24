"""
BioMoAX v8: CRISPR-130 retraining
=================================

This is a deliberately minimal update of train_moax_v7.py:

* all non-CRISPR datasets, splits, losses, and hyperparameters stay unchanged;
* A549 and U2OS use the same 90 train / 40 held-out gene split;
* the original 10 A549 held-out genes remain in the new 40-gene test set;
* checkpoints are written to logs/moax_v8_crispr130 and never overwrite v7.

The split is deterministic (seed 42) and materialized as:
data/processed/metadata/crispr_gene_split_v8_90_40_seed42.csv
"""

from __future__ import annotations

import argparse
import functools
import gc
import mmap
import os
import re
import shutil
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

import train_moax_v7 as base


RUN_NAME = "moax_v8_crispr130"
SPLIT_SEED = 42
N_TEST = 40
N_CRISPR_CACHE = 500

BASE = Path(os.environ['BIOMOAX_PROJECT_ROOT'])
META_A549 = BASE / "data/processed/cpg0000_crispr/meta_cpg0000_A549.csv"
META_U2OS = BASE / "data/processed/cpg0000_crispr/meta_cpg0000_U2OS.csv"
COMPOUNDINFO = BASE / "data/processed/metadata/compoundinfo_beta.txt"
SPLIT_PATH = (
    BASE
    / "data/processed/metadata/crispr_gene_split_v8_90_40_seed42.csv"
)
LOG_DIR = BASE / "logs/moax_v8_crispr130"
MEMMAP_DIR = BASE / "data/tmp/moax_v8_memmap"

ORIGINAL_A549_TEST = {
    "ADA",
    "ALDH2",
    "BTK",
    "GUCY1B1",
    "IL1B",
    "KDR",
    "MMP2",
    "PRKCB",
    "PTK2B",
    "TNF",
}


def build_or_validate_split() -> pd.DataFrame:
    """Create one shared deterministic gene split for A549 and U2OS."""
    meta_a = pd.read_csv(META_A549)
    meta_u = pd.read_csv(META_U2OS)
    genes_a = set(meta_a["gene"].astype(str))
    genes_u = set(meta_u["gene"].astype(str))

    if genes_a != genes_u:
        raise RuntimeError(
            "A549/U2OS CRISPR gene sets differ: "
            f"A549-only={sorted(genes_a - genes_u)}, "
            f"U2OS-only={sorted(genes_u - genes_a)}"
        )
    if len(genes_a) != 130:
        raise RuntimeError(f"Expected 130 shared genes, found {len(genes_a)}")
    if not ORIGINAL_A549_TEST <= genes_a:
        missing = sorted(ORIGINAL_A549_TEST - genes_a)
        raise RuntimeError(f"Original held-out genes missing from metadata: {missing}")

    genes = sorted(genes_a)

    # Use exactly the same target -> MOA mapping rule as train_moax_v7.py.
    compoundinfo = pd.read_csv(COMPOUNDINFO, sep="\t", low_memory=False)
    compoundinfo["target_up"] = compoundinfo["target"].fillna("").str.upper()
    gene_to_moa: dict[str, str] = {}
    for gene in genes:
        hits = compoundinfo[
            compoundinfo["target_up"].str.contains(
                rf"\b{re.escape(gene.upper())}\b",
                regex=True,
                na=False,
            )
        ]
        moas = hits["moa"].dropna().astype(str)
        if len(moas):
            gene_to_moa[gene] = moas.value_counts().index[0]

    mapped_genes = set(gene_to_moa)
    if len(mapped_genes) != 90:
        raise RuntimeError(
            f"Expected 90 genes with a target-derived MOA, found {len(mapped_genes)}"
        )
    if not ORIGINAL_A549_TEST <= mapped_genes:
        missing = sorted(ORIGINAL_A549_TEST - mapped_genes)
        raise RuntimeError(f"Original held-out genes without MOA mapping: {missing}")

    rng = np.random.default_rng(SPLIT_SEED)
    candidates = sorted(mapped_genes - ORIGINAL_A549_TEST)
    extra_test = set(
        rng.choice(
            candidates,
            N_TEST - len(ORIGINAL_A549_TEST),
            replace=False,
        ).tolist()
    )
    test_genes = ORIGINAL_A549_TEST | extra_test

    expected = pd.DataFrame(
        {
            "gene": genes,
            "split": ["test" if gene in test_genes else "train" for gene in genes],
            "seed": SPLIT_SEED,
            "moa_mapped": [gene in mapped_genes for gene in genes],
            "mapped_moa": [gene_to_moa.get(gene, "") for gene in genes],
            "preserved_ep28_test": [
                gene in ORIGINAL_A549_TEST for gene in genes
            ],
        }
    )

    if SPLIT_PATH.exists():
        existing = pd.read_csv(SPLIT_PATH)
        compare_cols = [
            "gene",
            "split",
            "seed",
            "moa_mapped",
            "mapped_moa",
            "preserved_ep28_test",
        ]
        # Empty MOA strings round-trip through CSV as NaN by default.
        existing["mapped_moa"] = existing["mapped_moa"].fillna("")
        if not existing[compare_cols].equals(expected[compare_cols]):
            raise RuntimeError(
                f"Existing split does not match deterministic v8 split: {SPLIT_PATH}"
            )
    else:
        SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        expected.to_csv(SPLIT_PATH, index=False)

    counts = expected["split"].value_counts().to_dict()
    if counts != {"train": 90, "test": 40}:
        raise RuntimeError(f"Unexpected split counts: {counts}")
    if not expected.loc[expected["split"] == "test", "moa_mapped"].all():
        raise RuntimeError("Every held-out test gene must have a mapped MOA")
    return expected


def configure_base(split: pd.DataFrame) -> set[str]:
    """Apply only the CRISPR split and output-directory overrides."""
    test_genes = set(split.loc[split["split"] == "test", "gene"])
    base.HELD_OUT_GENES_A549 = test_genes
    base.HELD_OUT_GENES_U2OS = test_genes
    # The container is limited to 120 GB. Five hundred crops per training gene
    # preserve ample sampling diversity while avoiding a transient allocation
    # spike when both 130-gene H5 files are loaded.
    base.N_CRISPR_CACHE = N_CRISPR_CACHE
    base.LOG_DIR = LOG_DIR
    base.LOG_DIR.mkdir(parents=True, exist_ok=True)
    return test_genes


def evict_file_cache(path: Path | str) -> None:
    """Ask Linux to discard clean cache pages after an H5 is copied to RAM."""
    if not hasattr(os, "posix_fadvise"):
        return
    path = Path(path)
    if not path.exists():
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


class DiskBackedMultiH5Reader:
    """v7-equivalent sampling backed by local memmap files instead of RAM."""

    COPY_CHUNK = 2000

    def __init__(self):
        self._caches: dict[str, np.memmap] = {}
        self._labels: dict[str, np.ndarray] = {}
        self._source_indices: dict[str, np.ndarray] = {}
        self._cache_paths: dict[str, Path] = {}
        shutil.rmtree(MEMMAP_DIR, ignore_errors=True)
        MEMMAP_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Disk-backed crop cache: {MEMMAP_DIR}")

    def _allocate(self, cell: str, shape: tuple[int, ...], dtype) -> np.memmap:
        path = MEMMAP_DIR / f"{cell}.uint16.mmap"
        raw = np.memmap(path, mode="w+", dtype=dtype, shape=shape)
        self._cache_paths[cell] = path
        return raw

    def _copy_selected(
        self,
        ds,
        idx: np.ndarray,
        raw: np.memmap,
    ) -> None:
        pos = 0
        for start in range(0, ds.shape[0], self.COPY_CHUNK):
            end = min(start + self.COPY_CHUNK, ds.shape[0])
            sel = idx[(idx >= start) & (idx < end)]
            if len(sel) == 0:
                continue
            block = ds[start:end, : base.IMG_CHANNELS]
            raw[pos : pos + len(sel)] = block[sel - start]
            pos += len(sel)
            del block
        if pos != len(idx):
            raise RuntimeError(f"Copied {pos} crops, expected {len(idx)}")

    @staticmethod
    def _finish(raw: np.memmap, source_h5: Path | str) -> None:
        raw.flush()
        # Discard freshly written mapped pages; training faults them back in
        # on demand without keeping a second anonymous copy.
        try:
            raw._mmap.madvise(mmap.MADV_DONTNEED)
        except (AttributeError, OSError):
            pass
        evict_file_cache(source_h5)

    def load_cell(
        self,
        cell,
        h5_path,
        needed_labels=None,
        max_crops_per_label=None,
    ):
        print(
            f"Loading {cell}: {Path(h5_path).name}"
            + (
                f" (max {max_crops_per_label}/drug, disk-backed)"
                if max_crops_per_label
                else " (disk-backed)"
            )
        )
        t0 = time.time()
        rng = np.random.default_rng(base.RANDOM_SEED)
        with h5py.File(h5_path, "r") as f:
            all_labels = f["labels"][:].astype(np.int32)
            ds = f["crops"]
            if needed_labels is not None:
                wanted = np.fromiter(needed_labels, dtype=np.int32)
                base_idx = np.where(np.isin(all_labels, wanted))[0]
            else:
                base_idx = np.arange(len(all_labels))
            excluded = getattr(base, "EXCLUDED_SOURCE_ROWS_BY_CELL", {}).get(cell)
            if excluded is not None:
                excluded = np.asarray(excluded, dtype=np.int64)
                before = len(base_idx)
                base_idx = base_idx[~np.isin(base_idx, excluded)]
                removed = before - len(base_idx)
                if removed != len(excluded):
                    raise RuntimeError(
                        f"{cell}: expected to exclude {len(excluded)} source rows, "
                        f"but removed {removed}"
                    )
                print(f"  {cell}: excluded {removed} protocol-held-out source crops")
            if max_crops_per_label is not None:
                keep = []
                for lbl in np.unique(all_labels[base_idx]):
                    rows = base_idx[all_labels[base_idx] == lbl]
                    if len(rows) > max_crops_per_label:
                        rows = np.sort(
                            rng.choice(rows, max_crops_per_label, replace=False)
                        )
                    keep.append(rows)
                idx = (
                    np.sort(np.concatenate(keep))
                    if keep
                    else np.array([], dtype=np.int64)
                )
            else:
                idx = base_idx
            shape = (len(idx), base.IMG_CHANNELS) + ds.shape[2:]
            raw = self._allocate(cell, shape, ds.dtype)
            self._copy_selected(ds, idx, raw)
        self._finish(raw, h5_path)
        self._caches[cell] = raw
        self._labels[cell] = all_labels[idx]
        self._source_indices[cell] = idx
        print(
            f"  {cell}: {raw.shape}  {raw.nbytes / 1024**3:.1f} GB memmap"
            f"  ({time.time() - t0:.0f}s)"
        )

    def load_cpg0030(
        self,
        h5_path,
        meta_csv_path,
        train_intlbls,
        intlbl_to_global,
        max_crops=100,
    ):
        print(
            f"Loading cpg0030: {Path(h5_path).name} "
            f"(max {max_crops}/drug, disk-backed)"
        )
        t0 = time.time()
        rng = np.random.default_rng(base.RANDOM_SEED)
        meta = pd.read_csv(meta_csv_path)
        meta["_row"] = np.arange(len(meta))
        pairs = []
        for ilbl, grp in meta[meta["int_label"].isin(train_intlbls)].groupby(
            "int_label"
        ):
            rows = grp["_row"].values
            if len(rows) > max_crops:
                rows = rng.choice(rows, max_crops, replace=False)
            g_id = intlbl_to_global[ilbl]
            pairs.extend((int(row), g_id) for row in rows)
        pairs.sort(key=lambda item: item[0])
        all_idx = np.array([item[0] for item in pairs], dtype=np.int64)
        all_glbl = np.array([item[1] for item in pairs], dtype=np.int32)
        with h5py.File(h5_path, "r") as f:
            ds = f["crops"]
            raw = self._allocate(
                "cpg0030",
                (len(all_idx), base.IMG_CHANNELS, 128, 128),
                np.uint16,
            )
            self._copy_selected(ds, all_idx, raw)
        self._finish(raw, h5_path)
        self._caches["cpg0030"] = raw
        self._labels["cpg0030"] = all_glbl
        print(
            f"  cpg0030: {raw.shape}  {raw.nbytes / 1024**3:.1f} GB memmap"
            f"  ({time.time() - t0:.0f}s)"
        )

    def load_with_global_offset(
        self,
        cell,
        h5_path,
        h5_label_col,
        global_offset,
        max_crops_per_label=200,
    ):
        print(
            f"Loading {cell}: {Path(h5_path).name} "
            f"(offset={global_offset}, max {max_crops_per_label}/drug, disk-backed)"
        )
        t0 = time.time()
        rng = np.random.default_rng(base.RANDOM_SEED)
        with h5py.File(h5_path, "r") as f:
            all_labels = f[h5_label_col][:].astype(np.int32)
            ds = f["crops"]
            keep = []
            for lbl in np.unique(all_labels):
                rows = np.where(all_labels == lbl)[0]
                if len(rows) > max_crops_per_label:
                    rows = np.sort(
                        rng.choice(rows, max_crops_per_label, replace=False)
                    )
                keep.append(rows)
            idx = (
                np.sort(np.concatenate(keep))
                if keep
                else np.array([], dtype=np.int64)
            )
            shape = (len(idx), base.IMG_CHANNELS) + ds.shape[2:]
            raw = self._allocate(cell, shape, ds.dtype)
            self._copy_selected(ds, idx, raw)
        self._finish(raw, h5_path)
        self._caches[cell] = raw
        self._labels[cell] = all_labels[idx] + global_offset
        print(
            f"  {cell}: {raw.shape}  {raw.nbytes / 1024**3:.1f} GB memmap"
            f"  ({time.time() - t0:.0f}s)"
        )

    def unload_cell(self, cell):
        raw = self._caches.pop(cell, None)
        self._labels.pop(cell, None)
        self._source_indices.pop(cell, None)
        path = self._cache_paths.pop(cell, None)
        if raw is not None:
            try:
                raw._mmap.close()
            except (AttributeError, OSError):
                pass
            del raw
            gc.collect()
        if path is not None:
            path.unlink(missing_ok=True)

    def get_label_to_rows(self, cell):
        labels = self._labels[cell]
        result = {}
        for i, label in enumerate(labels):
            result.setdefault(int(label), []).append(i)
        return {key: np.asarray(rows) for key, rows in result.items()}

    def read(self, cell, rows):
        return self._caches[cell][rows]


def install_disk_backed_reader() -> None:
    """Install the memory-safe reader and retain source-cache eviction."""
    base.MultiH5Reader = DiskBackedMultiH5Reader

    original_crispr_loader = base.load_crispr_h5

    @functools.wraps(original_crispr_loader)
    def wrapped_crispr_loader(h5_path, *args, **kwargs):
        result = original_crispr_loader(h5_path, *args, **kwargs)
        evict_file_cache(h5_path)
        return result

    base.load_crispr_h5 = wrapped_crispr_loader

    # Remove cache left by an interrupted previous startup before allocating
    # any new arrays.
    for h5_path in list(base.H5.values()) + [
        base.CRISPR_DIR / "crops_cpg0000_A549.h5",
        base.CRISPR_DIR / "crops_cpg0000_U2OS.h5",
    ]:
        evict_file_cache(h5_path)


def main(check_only: bool = False) -> None:
    split = build_or_validate_split()
    test_genes = configure_base(split)
    train_genes = set(split.loc[split["split"] == "train", "gene"])

    print("=" * 72)
    print(f"{RUN_NAME}: shared CRISPR split")
    print(f"  split manifest: {SPLIT_PATH}")
    print(f"  A549: train={len(train_genes)} held-out={len(test_genes)}")
    print(f"  U2OS: train={len(train_genes)} held-out={len(test_genes)}")
    print(
        "  held-out genes with mapped MOA: "
        f"{split.loc[split['split'] == 'test', 'moa_mapped'].sum()}/"
        f"{len(test_genes)}"
    )
    print(f"  CRISPR crop cache: max {N_CRISPR_CACHE}/training gene/cell line")
    print(
        "  original ep28 A549 test genes preserved: "
        f"{len(ORIGINAL_A549_TEST & test_genes)}/{len(ORIGINAL_A549_TEST)}"
    )
    print(f"  output: {LOG_DIR}")
    print("  all non-CRISPR settings inherited unchanged from train_moax_v7.py")
    print("=" * 72)

    if check_only:
        print("Preflight passed; training was not started.")
        return

    install_disk_backed_reader()
    base.main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate and materialize the split without loading training data.",
    )
    args = parser.parse_args()
    main(check_only=args.check_only)
