"""
train_moax_v7.py
================
v6 → v7 变动：

新增数据（均无L1000，只参与i2i/gene_i2i）：
  cpg0000 compound A549   260 cpd, 每基因2个，与CRISPR A549直接基因配对
  cpg0000 compound U2OS   260 cpd, 68基因与CRISPR U2OS配对
  cpg0000 CRISPR A549     120 训练基因 (held-out 10: ADA/ALDH2/BTK/GUCY1B1/IL1B/KDR/MMP2/PRKCB/PTK2B/TNF)
  cpg0000 CRISPR U2OS     68 基因，全量训练

新损失 loss_gene_i2i：
  CRISPR图像(基因X) ↔ cpg0000化合物图像(靶基因X)
  替代 v6 的 loss_crispr_i2i（v6用药名匹配cpg0012/0004，v7直接按基因对齐）

Checkpoint 条件简化：
  v7: e2e_U2OS≥22.2% AND e2e_A549≥17.9% AND i2i_U2OS≥14.4% AND i2i_A549≥13.6%
  去掉 i2e/e2i 门槛，每个满足条件的 epoch 都保存为 qualified_ep{epoch:03d}.pt

Global ID offset:
  U2OS(cpg0012) 0-689    A549(cpg0004) 690-1728
  HT29 1729-1806          A375 1807-1892   MCF7 1893-1977
  cpg0030 1978-2132       cpd_A549 2133-2392
  cpd_U2OS 2393-2652      N_TOTAL=2653
"""

import os

import math, random, time, threading, queue
from torch.utils.checkpoint import checkpoint as grad_ckpt
import numpy as np
import pandas as pd
import h5py, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from pathlib import Path
from tqdm import tqdm

BASE    = Path(os.environ['BIOMOAX_PROJECT_ROOT'])
PROC    = BASE / "data/processed"
META_D  = PROC / "metadata"
CPD_DIR = PROC / "cpg0000_compound"
CRISPR_DIR = PROC / "cpg0000_crispr"
LOG_DIR = BASE / "logs/moax_v7"

H5 = {
    "U2OS":    PROC / "crops/crops_cpg0012_690.h5",
    "A549":    PROC / "crops/crops_cpg0004_a549x1055.h5",
    "A375":    PROC / "crops/crops_cpg0016_A375.h5",
    "MCF7":    PROC / "crops/crops_cpg0016_MCF7.h5",
    "HT29":    PROC / "crops/crops_cpg0016_HT29.h5",
    "cpg0030": PROC / "crops_cpg0030_u2os.h5",
    "cpd_A549": CPD_DIR / "crops_cpg0000_cpd_A549.h5",
    "cpd_U2OS": CPD_DIR / "crops_cpg0000_cpd_U2OS.h5",
}
L1K = {
    "U2OS": PROC / "features/l1000_u2os_690.npy",
    "A549": PROC / "features/l1000_a549_1039.npy",
    "A375": PROC / "features/l1000_a375_86.npy",
    "MCF7": PROC / "features/l1000_mcf7_85.npy",
    "HT29": PROC / "features/l1000_ht29_78.npy",
}
META = {
    "U2OS": META_D / "L1000_JumpCP_meta.csv",
    "A549": META_D / "l1000_a549_1039_meta.csv",
    "A375": META_D / "cpg0016_A375_meta.csv",
    "MCF7": META_D / "cpg0016_MCF7_meta.csv",
    "HT29": META_D / "cpg0016_HT29_meta.csv",
}
PROFILES_MERGED = PROC / "metadata/cpg0012_profiles_merged.csv"

# CRISPR held-out（不参与训练，作泛化/零样本测试）
#   A549: 部分留出（有 L1000，主力 gene_i2i 训练 120 基因）
#   U2OS: 全部 68 基因留出（无 L1000，纯跨细胞系零样本测试，不进训练）
HELD_OUT_GENES_A549 = {
    'ADA','ALDH2','BTK','GUCY1B1','IL1B','KDR','MMP2','PRKCB','PTK2B','TNF'
}
HELD_OUT_GENES_U2OS = "ALL"   # 全部 U2OS CRISPR 基因均 held-out

# ── Offsets ───────────────────────────────────────────────────────────────────
OFFSET = {
    "U2OS":    0,
    "A549":    690,
    "HT29":    1729,
    "A375":    1807,
    "MCF7":    1893,
    "cpg0030": 1978,
    "cpd_A549": 2133,
    "cpd_U2OS": 2393,
}
N_CPD     = {"U2OS": 690, "A549": 1039, "HT29": 78, "A375": 86, "MCF7": 85}
N_CPD_NEW = {"cpd_A549": 260, "cpd_U2OS": 260}
NO_L1K_OFFSET = OFFSET["cpg0030"]   # global_id >= 此值的 drugs 无 L1000

# Optional compound-pair expansion used by newer wrappers.  Historical runs
# leave this empty, preserving their offsets, sampling, and checkpoints.
# Each entry must provide: offset, n, h5, l1k, meta, label_col, cell_line.
TRAINING_EXPANSION = {}

# Historical datasets used the contiguous ``NO_L1K_OFFSET`` convention.  A
# sparse explicit set is required once paired compounds are appended after
# image-only datasets.  ``None`` retains the historical convention exactly.
L1000_AVAILABLE_IDS = None

# ── Hyperparameters ───────────────────────────────────────────────────────────
EMBED_DIM     = 256
VIT_DIM       = 384
EXPR_TRUNK    = 512

BATCH_DRUGS   = 500
K_PER_EPOCH   = 300
N_CROPS_BATCH = 10

# Optional MOA-aware sampling.  Historical v7/v8 runs keep the original
# uniform sampler; newer wrappers can enable this without changing old run
# semantics or checkpoint reproducibility.
USE_MOA_BALANCED_SAMPLER = False
MIN_DRUGS_PER_MOA = 2
MOA_BALANCED_FRACTION = 0.5

ALPHA_CPD        = 2.0
ALPHA_MOA_I      = 1.7
ALPHA_MOA_E      = 3.0
ALPHA_MOA_CROSS  = 2.0
ALPHA_GENE_I2I   = 1.0   # CRISPR图像↔cpg0000化合物图像 基因级配对
ALPHA_CRISPR_I2E = 1.5   # CRISPR图像↔药物L1000 (靶向+MOA)，补 CRISPR→L1000 短板

# ── Checkpoint 阈值（去掉 i2e/e2i 门槛）────────────────────────────────────────
THRESH_E2E_U2OS = 0.222   # L1000 baseline
THRESH_E2E_A549 = 0.179
THRESH_I2I_U2OS = 0.144   # best DP single-modal baseline
THRESH_I2I_A549 = 0.136   # best CP single-modal baseline

GENE_I2I_PER_BATCH   = 24   # 每 batch 采样的基因对数（v6=16，v7扩至24）
CRISPR_I2E_PER_BATCH = 24   # 每 batch 采样的 CRISPR→药物L1000 基因数

TRAIN_CROP_RATIO = 0.8
RANDOM_SEED      = 42
N_CRISPR_CACHE   = 1000   # 每基因缓存的 crops 数

EPOCHS       = 150
LR           = 1e-4
WARMUP       = 5
INIT_TEMP    = 0.07
WEIGHT_DECAY = 1e-2

IMG_CHANNELS  = 5
VIT_PATCH     = 16
VIT_DEPTH     = 12
VIT_HEADS     = 6
VIT_MLP_RATIO = 4

N_CROPS_EVAL  = 50
USE_AMP       = True


ZEROSHOT_FREQ   = 20

LOG_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

class MultiH5Reader:
    def __init__(self):
        self._caches = {}
        self._labels = {}

    def load_cell(self, cell, h5_path, needed_labels=None, max_crops_per_label=None):
        print(f"Loading {cell}: {Path(h5_path).name}"
              + (f" (max {max_crops_per_label}/drug)" if max_crops_per_label else ""))
        t0 = time.time()
        rng = np.random.default_rng(RANDOM_SEED)
        with h5py.File(h5_path, "r") as f:
            all_labels = f["labels"][:].astype(np.int32)
            ds = f["crops"]
            if needed_labels is not None:
                base_idx = np.where(np.isin(all_labels, np.fromiter(needed_labels, dtype=np.int32)))[0]
            else:
                base_idx = np.arange(len(all_labels))
            if max_crops_per_label is not None:
                keep = []
                for lbl in np.unique(all_labels[base_idx]):
                    rows = base_idx[all_labels[base_idx] == lbl]
                    if len(rows) > max_crops_per_label:
                        rows = rng.choice(rows, max_crops_per_label, replace=False)
                        rows = np.sort(rows)
                    keep.append(rows)
                idx = np.sort(np.concatenate(keep)) if keep else np.array([], dtype=np.int64)
            else:
                idx = base_idx
            raw = np.empty((len(idx), IMG_CHANNELS) + ds.shape[2:], dtype=ds.dtype)
            CHUNK = 20000; pos = 0
            for start in range(0, ds.shape[0], CHUNK):
                end = min(start + CHUNK, ds.shape[0])
                sel = idx[(idx >= start) & (idx < end)]
                if len(sel) == 0: continue
                raw[pos:pos+len(sel)] = ds[start:end, :IMG_CHANNELS][sel - start]
                pos += len(sel)
            self._caches[cell] = raw
            self._labels[cell] = all_labels[idx]
        gb = raw.nbytes / 1024**3
        print(f"  {cell}: {raw.shape}  {gb:.1f} GB  ({time.time()-t0:.0f}s)")

    def load_cpg0030(self, h5_path, meta_csv_path, train_intlbls, intlbl_to_global,
                     max_crops=100):
        print(f"Loading cpg0030: {Path(h5_path).name} (max {max_crops}/drug)")
        t0 = time.time()
        rng = np.random.default_rng(RANDOM_SEED)
        meta = pd.read_csv(meta_csv_path)
        meta['_row'] = np.arange(len(meta))
        idx_label_pairs = []
        for ilbl, grp in meta[meta['int_label'].isin(train_intlbls)].groupby('int_label'):
            rows = grp['_row'].values
            if len(rows) > max_crops:
                rows = rng.choice(rows, max_crops, replace=False)
            g_id = intlbl_to_global[ilbl]
            for r in rows:
                idx_label_pairs.append((int(r), g_id))
        idx_label_pairs.sort(key=lambda x: x[0])
        all_idx  = np.array([p[0] for p in idx_label_pairs], dtype=np.int64)
        all_glbl = np.array([p[1] for p in idx_label_pairs], dtype=np.int32)
        with h5py.File(h5_path, "r") as f:
            ds = f["crops"]
            raw = np.empty((len(all_idx), IMG_CHANNELS, 128, 128), dtype=np.uint16)
            CHUNK = 20000; pos = 0
            for start in range(0, ds.shape[0], CHUNK):
                end = min(start + CHUNK, ds.shape[0])
                sel_mask = (all_idx >= start) & (all_idx < end)
                sel = all_idx[sel_mask]
                if len(sel) == 0: continue
                raw[pos:pos+len(sel)] = ds[start:end, :IMG_CHANNELS][sel - start]
                pos += len(sel)
        self._caches["cpg0030"] = raw
        self._labels["cpg0030"] = all_glbl
        gb = raw.nbytes / 1024**3
        print(f"  cpg0030: {raw.shape}  {gb:.1f} GB  ({time.time()-t0:.0f}s)")

    def load_with_global_offset(self, cell, h5_path, h5_label_col, global_offset,
                                 max_crops_per_label=200):
        """通用加载：H5 labels 0-indexed，global_id = label + global_offset。"""
        print(f"Loading {cell}: {Path(h5_path).name} (offset={global_offset}, max {max_crops_per_label}/drug)")
        t0 = time.time()
        rng = np.random.default_rng(RANDOM_SEED)
        with h5py.File(h5_path, "r") as f:
            all_labels = f["labels"][:].astype(np.int32)
            ds = f["crops"]
            keep = []
            for lbl in np.unique(all_labels):
                rows = np.where(all_labels == lbl)[0]
                if len(rows) > max_crops_per_label:
                    rows = rng.choice(rows, max_crops_per_label, replace=False)
                    rows = np.sort(rows)
                keep.append(rows)
            idx = np.sort(np.concatenate(keep)) if keep else np.array([], dtype=np.int64)
            raw = np.empty((len(idx), IMG_CHANNELS) + ds.shape[2:], dtype=ds.dtype)
            CHUNK = 20000; pos = 0
            for start in range(0, ds.shape[0], CHUNK):
                end = min(start + CHUNK, ds.shape[0])
                sel = idx[(idx >= start) & (idx < end)]
                if len(sel) == 0: continue
                raw[pos:pos+len(sel)] = ds[start:end, :IMG_CHANNELS][sel - start]
                pos += len(sel)
            self._caches[cell] = raw
            self._labels[cell] = all_labels[idx] + global_offset
        gb = raw.nbytes / 1024**3
        print(f"  {cell}: {raw.shape}  {gb:.1f} GB  ({time.time()-t0:.0f}s)")

    def unload_cell(self, cell):
        self._caches.pop(cell, None)
        self._labels.pop(cell, None)

    def get_label_to_rows(self, cell):
        labels = self._labels[cell]
        d = {}
        for i, l in enumerate(labels):
            d.setdefault(int(l), []).append(i)
        return {k: np.array(v) for k, v in d.items()}

    def read(self, cell, rows):
        return self._caches[cell][rows]


def load_crispr_h5(h5_path, meta_csv, held_out_genes=None, n_per_gene=1000):
    """
    从 H5 加载 CRISPR crops，缓存到 dict。
    返回: {gene_idx: np.array(N, 5, 128, 128)}, all_gene_names
    """
    meta = pd.read_csv(meta_csv)
    all_genes = meta['gene'].tolist()
    if held_out_genes == "ALL":
        held_out = set(all_genes)
    else:
        held_out = set(held_out_genes or [])
    rng = np.random.default_rng(RANDOM_SEED)
    gene_crops = {}
    print(f"  加载 {Path(h5_path).name}  ({len(all_genes)} 基因, held-out={len(held_out)})...")
    t0 = time.time()
    with h5py.File(h5_path, 'r') as f:
        all_labels = f['labels'][:].astype(np.int32)
        ds = f['crops']
        for gene_idx, gene in enumerate(all_genes):
            if gene in held_out:
                continue
            rows = np.where(all_labels == gene_idx)[0]
            if len(rows) == 0:
                continue
            if len(rows) > n_per_gene:
                rows = np.sort(rng.choice(rows, n_per_gene, replace=False))
            crops = np.empty((len(rows), IMG_CHANNELS, 128, 128), dtype=np.uint16)
            CHUNK = 5000; pos = 0
            for start in range(0, ds.shape[0], CHUNK):
                end = min(start + CHUNK, ds.shape[0])
                sel = rows[(rows >= start) & (rows < end)]
                if len(sel) == 0: continue
                crops[pos:pos+len(sel)] = ds[start:end, :IMG_CHANNELS][sel - start]
                pos += len(sel)
            gene_crops[gene_idx] = crops
    total_gb = sum(c.nbytes for c in gene_crops.values()) / 1e9
    print(f"    {len(gene_crops)} 训练基因  {total_gb:.1f} GB  ({time.time()-t0:.0f}s)")
    return gene_crops, all_genes


# ═══════════════════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════════════════

class PatchEmbed(nn.Module):
    def __init__(self, in_ch=5, patch_size=16, img_size=128, dim=384):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x): return self.proj(x).flatten(2).transpose(1, 2)

class Attention(nn.Module):
    def __init__(self, dim, heads=6, dropout=0.0):
        super().__init__()
        self.heads = heads; self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim); self.drop = nn.Dropout(dropout)
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = self.drop((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(B, N, C))

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=6, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim); self.attn = Attention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(mlp_dim, dim), nn.Dropout(dropout))
    def forward(self, x):
        x = x + self.attn(self.norm1(x)); return x + self.mlp(self.norm2(x))

class ImageBackbone(nn.Module):
    def __init__(self, in_ch=5, patch_size=16, img_size=128,
                 vit_dim=384, depth=12, heads=6, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.inst_norm   = nn.InstanceNorm2d(in_ch, affine=True)
        self.patch_embed = PatchEmbed(in_ch, patch_size, img_size, vit_dim)
        n_patches        = self.patch_embed.n_patches
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, vit_dim))
        self.pos_embed   = nn.Parameter(torch.zeros(1, n_patches + 1, vit_dim))
        self.pos_drop    = nn.Dropout(dropout)
        self.blocks      = nn.Sequential(*[TransformerBlock(vit_dim, heads, mlp_ratio, dropout)
                                           for _ in range(depth)])
        self.norm        = nn.LayerNorm(vit_dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def _augment(self, x):
        if torch.rand(1) > 0.5: x = x.flip(-1)
        if torch.rand(1) > 0.5: x = x.flip(-2)
        k = torch.randint(0, 4, (1,)).item()
        if k > 0: x = torch.rot90(x, k, [-2, -1])
        x = torch.clamp(x * (0.8 + torch.rand(IMG_CHANNELS, 1, 1).to(x.device) * 0.4), 0, 1)
        if torch.rand(1) > 0.5:
            sigma = 0.5 + torch.rand(1).item(); k_size = 3
            coords = torch.arange(k_size, dtype=torch.float32) - k_size // 2
            g = torch.exp(-coords ** 2 / (2 * sigma ** 2)); g = g / g.sum()
            kernel = g.outer(g).to(x.device).expand(IMG_CHANNELS, 1, k_size, k_size)
            x = torch.clamp(F.conv2d(x.unsqueeze(0), kernel, padding=k_size//2,
                                     groups=IMG_CHANNELS).squeeze(0), 0, 1)
        return x

    def forward(self, x, augment=False):
        if augment and self.training:
            x = torch.stack([self._augment(xi) for xi in x])
        x = self.inst_norm(x)
        B = x.shape[0]; x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = self.pos_drop(torch.cat([cls, x], dim=1) + self.pos_embed)
        for block in self.blocks: x = grad_ckpt(block, x, use_reentrant=False)
        return self.norm(x[:, 0])

class ExprBackbone(nn.Module):
    def __init__(self, in_dim, trunk_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(1024, trunk_dim), nn.LayerNorm(trunk_dim), nn.GELU(), nn.Dropout(0.1))
    def forward(self, x): return self.net(x)

class ProjHead(nn.Module):
    def __init__(self, in_dim, out_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(in_dim, out_dim))
        self.apply(self._init)
    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self, x): return F.normalize(self.net(x), dim=-1)

class QuadTemp(nn.Module):
    def __init__(self, init_temp=0.07):
        super().__init__()
        self.log_temp_cpd       = nn.Parameter(torch.tensor(math.log(init_temp)))
        self.log_temp_i2i       = nn.Parameter(torch.tensor(math.log(init_temp)))
        self.log_temp_e2e       = nn.Parameter(torch.tensor(math.log(init_temp)))
        self.log_temp_moa_cross = nn.Parameter(torch.tensor(math.log(init_temp)))
    @property
    def temp_cpd(self):       return self.log_temp_cpd.exp().item()
    @property
    def temp_i2i(self):       return self.log_temp_i2i.exp().item()
    @property
    def temp_e2e(self):       return self.log_temp_e2e.exp().item()
    @property
    def temp_moa_cross(self): return self.log_temp_moa_cross.exp().item()


# ═══════════════════════════════════════════════════════════════════════════════
# Loss functions
# ═══════════════════════════════════════════════════════════════════════════════

def supcon(embs, pos_mask, log_temp):
    B = embs.shape[0]; device = embs.device
    has_pos = pos_mask.any(dim=1)
    if not has_pos.any(): return torch.zeros([], device=device)
    temp = log_temp.exp().clamp(1e-4, 100)
    sim = (embs @ embs.T) / temp
    self_mask = torch.eye(B, dtype=torch.bool, device=device)
    log_denom  = torch.logsumexp(sim.masked_fill(self_mask, float('-inf')), dim=1)
    log_prob   = sim - log_denom.unsqueeze(1)
    n_pos      = pos_mask.float().sum(dim=1).clamp(min=1)
    loss_per   = -(pos_mask.float() * log_prob).sum(dim=1) / n_pos
    return loss_per[has_pos].mean()

def cross_supcon(embs_a, embs_b, pos_mask, log_temp):
    temp = log_temp.exp().clamp(1e-4, 100)
    sim  = (embs_a @ embs_b.T) / temp
    n_pos_a    = pos_mask.float().sum(dim=1).clamp(min=1)
    log_prob_a = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    loss_a     = -(pos_mask.float() * log_prob_a).sum(dim=1) / n_pos_a
    n_pos_b    = pos_mask.float().sum(dim=0).clamp(min=1)
    log_prob_b = sim.T - torch.logsumexp(sim.T, dim=1, keepdim=True)
    loss_b     = -(pos_mask.T.float() * log_prob_b).sum(dim=1) / n_pos_b
    return (loss_a.mean() + loss_b.mean()) / 2

def make_moa_mask(batch_moa, device):
    B = len(batch_moa)
    mask = torch.zeros(B, B, dtype=torch.bool, device=device)
    for i in range(B):
        if not batch_moa[i]: continue
        for j in range(B):
            if i != j and batch_moa[i] & batch_moa[j]:
                mask[i, j] = True
    return mask

def make_moa_mask_asym(q_drugs, g_drugs, moa_sets, device):
    Q, G = len(q_drugs), len(g_drugs)
    mask = torch.zeros(Q, G, dtype=torch.bool, device=device)
    for i, q in enumerate(q_drugs):
        if not moa_sets[q]: continue
        for j, g in enumerate(g_drugs):
            if moa_sets[q] & moa_sets[g]:
                mask[i, j] = True
    return mask


# ═══════════════════════════════════════════════════════════════════════════════
# Data utilities
# ═══════════════════════════════════════════════════════════════════════════════

def gpu_normalize(imgs):
    B, C, H, W = imgs.shape; flat = imgs.reshape(B * C, H * W); n = H * W
    k_lo = max(1, int(0.01 * n)); k_hi = min(n, int(0.99 * n))
    lo = torch.kthvalue(flat, k_lo, dim=1).values.view(B, C, 1, 1)
    hi = torch.kthvalue(flat, k_hi, dim=1).values.view(B, C, 1, 1)
    return torch.clamp((imgs - lo) / (hi - lo + 1e-8), 0, 1)

def split_drug_crops(all_rows, train_ratio=0.8, seed=42):
    rng = np.random.default_rng(seed)
    train_rows = {}; eval_rows = {}
    for d, (cell, rows) in all_rows.items():
        perm = rng.permutation(len(rows))
        n_tr = max(1, int(len(rows) * train_ratio))
        train_rows[d] = (cell, rows[perm[:n_tr]])
        eval_rows[d]  = (cell, rows[perm[n_tr:]] if n_tr < len(rows) else rows[perm[:1]])
    return train_rows, eval_rows

def fetch_batch_v2(drug_batch, drug_to_rows, multi_h5, l1000_combined, n_crops):
    crops_list = []; l1k_list = []; l1k_indices = []
    for i, d in enumerate(drug_batch):
        cell, rows = drug_to_rows[d]
        sampled = np.sort(np.random.choice(rows, n_crops, replace=(len(rows) < n_crops)))
        crops_list.append(multi_h5.read(cell, sampled).astype(np.float32) / 65535.0)
        has_l1000 = (
            d in L1000_AVAILABLE_IDS
            if L1000_AVAILABLE_IDS is not None
            else d < NO_L1K_OFFSET
        )
        if has_l1000:
            l1k_list.append(l1000_combined[d])
            l1k_indices.append(i)
    imgs  = torch.from_numpy(np.stack(crops_list, 0))
    exprs = torch.from_numpy(np.array(l1k_list, dtype=np.float32)) if l1k_list else torch.empty(0, l1000_combined.shape[1])
    return imgs, exprs, l1k_indices

class RandomDrugSampler:
    def __init__(self, drug_list, batch_size, k_per_epoch):
        self.drug_list   = drug_list
        self.batch_size  = min(batch_size, len(drug_list))
        self.k_per_epoch = k_per_epoch
    def __len__(self): return self.k_per_epoch
    def __iter__(self):
        for _ in range(self.k_per_epoch):
            yield random.sample(self.drug_list, self.batch_size)


class MoaBalancedDrugSampler:
    """Build unique-drug batches with explicit within-batch MOA positives.

    Each eligible MOA contributes at least ``min_per_moa`` distinct drugs
    while capacity remains.  Remaining positions are filled uniformly from
    all training drugs, so compounds without MOA annotations still train the
    compound-pair objective.
    """

    def __init__(self, drug_list, batch_size, k_per_epoch, moa_sets,
                 min_per_moa=2, balanced_fraction=0.5):
        self.drug_list = list(drug_list)
        self.batch_size = min(batch_size, len(self.drug_list))
        self.k_per_epoch = k_per_epoch
        self.min_per_moa = min_per_moa
        self.balanced_slots = min(
            self.batch_size,
            max(self.min_per_moa, int(self.batch_size * balanced_fraction)),
        )

        moa_to_drugs = {}
        for drug in self.drug_list:
            for moa in sorted(moa_sets[drug]):
                moa_to_drugs.setdefault(moa, []).append(drug)
        self.moa_groups = [
            moa_to_drugs[moa] for moa in sorted(moa_to_drugs)
            if len(set(moa_to_drugs[moa])) >= self.min_per_moa
        ]
        print(
            f"MOA-balanced sampler: {len(self.moa_groups)} eligible MOAs, "
            f"min {self.min_per_moa} drugs/MOA, "
            f"balanced slots<={self.balanced_slots}/{self.batch_size}"
        )

    def __len__(self):
        return self.k_per_epoch

    def __iter__(self):
        for _ in range(self.k_per_epoch):
            groups = self.moa_groups.copy()
            random.shuffle(groups)
            selected = []
            selected_set = set()

            for group in groups:
                if len(selected) + self.min_per_moa > self.balanced_slots:
                    break
                available = sorted(set(group) - selected_set)
                if len(available) < self.min_per_moa:
                    continue
                chosen = random.sample(available, self.min_per_moa)
                selected.extend(chosen)
                selected_set.update(chosen)

            remaining = self.batch_size - len(selected)
            if remaining:
                pool = [drug for drug in self.drug_list if drug not in selected_set]
                selected.extend(random.sample(pool, remaining))
            random.shuffle(selected)
            yield selected

class PrefetchLoaderV2:
    def __init__(self, sampler, fetch_fn, queue_size=1):
        self._sampler = sampler; self._fetch_fn = fetch_fn
        self._queue = queue.Queue(maxsize=queue_size)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
    def _worker(self):
        try:
            for batch in self._sampler:
                imgs, exprs, l1k_idx = self._fetch_fn(batch)
                exprs_pin = exprs.pin_memory() if exprs.numel() > 0 else exprs
                self._queue.put((batch, imgs.pin_memory(), exprs_pin, l1k_idx))
        finally: self._queue.put(None)
    def __iter__(self):
        while True:
            item = self._queue.get()
            if item is None: break
            yield item
    def __len__(self): return len(self._sampler)

def fetch_gene_i2i_batch(sample_list, crispr_a549, crispr_u2os,
                          gene_to_cpd_rows_a549, gene_to_cpd_rows_u2os,
                          multi_h5):
    """
    sample_list: [(cell, gene_idx), ...]
    crispr_a549/u2os: {gene_idx: (N,5,128,128)} 预缓存
    gene_to_cpd_rows_a549/u2os: {gene_idx: [(cpd_cell, rows), ...]}
    返回 (crispr_imgs, cpd_imgs, pos_mask)，对角线为正样本
    """
    cr_list, drug_list = [], []
    for cell, gi in sample_list:
        cache       = crispr_a549 if cell == "A549" else crispr_u2os
        cpd_map     = gene_to_cpd_rows_a549 if cell == "A549" else gene_to_cpd_rows_u2os
        if gi not in cache: continue
        crispr_gene = cache[gi]                         # (N, 5, 128, 128)
        entries     = cpd_map.get(gi, [])
        if not entries: continue
        cpd_cell, cpd_rows = random.choice(entries)
        if len(cpd_rows) == 0: continue
        cr_idx     = np.random.randint(0, len(crispr_gene))
        cpd_idx    = np.sort(np.random.choice(cpd_rows, 1, replace=False))
        cr_list.append(crispr_gene[cr_idx:cr_idx+1].astype(np.float32) / 65535.0)
        drug_list.append(multi_h5.read(cpd_cell, cpd_idx).astype(np.float32) / 65535.0)
    if len(cr_list) < 2: return None, None, None
    cr_imgs   = torch.from_numpy(np.concatenate(cr_list,   0))
    drug_imgs = torch.from_numpy(np.concatenate(drug_list, 0))
    pos_mask  = torch.eye(len(cr_list), dtype=torch.bool)
    return cr_imgs, drug_imgs, pos_mask


def fetch_crispr_i2e_batch(sample_list, crispr_a549, gene_to_pos_drugs):
    """
    CRISPR图像 → 药物L1000 (i2e方向) 采样。
    sample_list: [(gene_idx, gene_name), ...]  A549 训练基因
    gene_to_pos_drugs: {gene_name: set(药物global_id)}  靶向+同MOA的有L1K药物
    返回 (crispr_imgs, drug_gids, gene_names)；正样本掩码在 run_epoch 里按 MOA/靶向构建
    """
    cr_list, drug_gids, gene_names = [], [], []
    for gene_idx, gene_name in sample_list:
        if gene_idx not in crispr_a549: continue
        pos = gene_to_pos_drugs.get(gene_name)
        if not pos: continue
        cr = crispr_a549[gene_idx]
        ci = np.random.randint(0, len(cr))
        cr_list.append(cr[ci:ci+1].astype(np.float32) / 65535.0)
        drug_gids.append(int(random.choice(tuple(pos))))
        gene_names.append(gene_name)
    if len(cr_list) < 2: return None, None, None
    cr_imgs = torch.from_numpy(np.concatenate(cr_list, 0))
    return cr_imgs, drug_gids, gene_names


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def topk_metrics(sim, labels, moa_sets, valid_drug_mask=None, query_moa_list=None):
    G = sim.shape[1]; m = {}
    vmask = valid_drug_mask if valid_drug_mask is not None else torch.ones(len(labels), dtype=torch.bool, device=sim.device)
    if vmask.any():
        vsim = sim[vmask]; vlabels = labels[vmask]
        for k in (1, 5, 10):
            topk_idx = vsim.topk(min(k, G), dim=1).indices
            m[f"d_top{k}"] = (topk_idx == vlabels.unsqueeze(1)).any(1).float().mean().item()
        ranks = (vsim.argsort(dim=1, descending=True) == vlabels.unsqueeze(1)).int().argmax(dim=1) + 1
        m["d_mrr"] = (1.0 / ranks.float()).mean().item()
    else:
        for k in (1, 5, 10): m[f"d_top{k}"] = 0.0
        m["d_mrr"] = 0.0
    if moa_sets is not None:
        sorted_idx  = sim.argsort(dim=1, descending=True).cpu().tolist()
        labels_list = labels.cpu().tolist()
        h1s, h5s, h10s, aps = [], [], [], []
        for i, (q_label, q_sorted) in enumerate(zip(labels_list, sorted_idx)):
            q_moa = query_moa_list[i] if query_moa_list is not None else moa_sets[q_label]
            if not q_moa: continue
            h1 = h5 = h10 = False; hits = 0; ap_sum = 0.0; rank_adj = 0
            for gi in q_sorted:
                if gi == q_label: continue
                rank_adj += 1                          # 跳过自身后重新计数(修 off-by-one)
                if q_moa & moa_sets[gi]:
                    hits += 1; ap_sum += hits / rank_adj
                    if rank_adj <= 1:  h1  = True
                    if rank_adj <= 5:  h5  = True
                    if rank_adj <= 10: h10 = True
            n_pos = sum(1 for gi in range(G) if gi != q_label and q_moa & moa_sets[gi])
            if n_pos > 0:
                h1s.append(float(h1)); h5s.append(float(h5)); h10s.append(float(h10))
                aps.append(ap_sum / n_pos)
        if h1s:
            m["m_top1"]  = float(np.mean(h1s))
            m["m_top5"]  = float(np.mean(h5s))
            m["m_top10"] = float(np.mean(h10s))
            m["m_map"]   = float(np.mean(aps))
    return m

@torch.no_grad()
def evaluate_i2e(fn_img, fn_expr, query_drugs, query_rows,
                 gallery_drugs, l1000_combined, multi_h5, n_crops, moa_sets):
    all_expr = torch.from_numpy(l1000_combined[np.array(gallery_drugs)]).to(DEVICE)
    with autocast("cuda", enabled=USE_AMP):
        gallery_embs = F.normalize(fn_expr(all_expr).float(), dim=-1)
    gal_moa = [moa_sets[d] for d in gallery_drugs]
    gal_idx = {d: i for i, d in enumerate(gallery_drugs)}
    img_embs, labels, valid_qd = [], [], []
    for drug in query_drugs:
        if drug not in query_rows: continue
        cell, rows = query_rows[drug]
        sampled = np.sort(np.random.choice(rows, min(n_crops, len(rows)), replace=False))
        crops = multi_h5.read(cell, sampled).astype(np.float32) / 65535.0
        imgs = gpu_normalize(torch.from_numpy(crops).to(DEVICE))
        with autocast("cuda", enabled=USE_AMP): embs = fn_img(imgs)
        img_embs.append(F.normalize(embs.mean(0, keepdim=True).float(), dim=-1))
        labels.append(gal_idx.get(drug, -1)); valid_qd.append(drug)
    if not img_embs: return {}
    img_embs  = torch.cat(img_embs, 0)
    labels_t  = torch.tensor(labels, device=DEVICE)
    sim       = img_embs @ gallery_embs.T
    valid_mask = labels_t >= 0
    q_moa_list = [moa_sets[d] for d in valid_qd]
    m = topk_metrics(sim, labels_t, gal_moa, valid_drug_mask=valid_mask, query_moa_list=q_moa_list)
    return {f"i2e_{k}": v for k, v in m.items()}


def valid_l1000_drug_ids(
    query_drugs,
    l1000_combined,
    l1000_available=None,
):
    """Return query IDs that address an explicitly available L1000 row.

    ``l1000_available`` can be supplied by callers that maintain a separate
    availability mask.  The combined training matrix uses all-zero placeholder
    rows for datasets without L1000, so that convention is used by default.
    """
    if l1000_combined.ndim != 2:
        raise ValueError(
            f"l1000_combined must be 2-D, got shape={l1000_combined.shape}"
        )
    if l1000_available is None:
        l1000_available = (
            np.isfinite(l1000_combined).all(axis=1)
            & np.any(l1000_combined != 0, axis=1)
        )
    else:
        l1000_available = np.asarray(l1000_available, dtype=bool)
        if l1000_available.shape != (l1000_combined.shape[0],):
            raise ValueError(
                "l1000_available must have one entry per L1000 row, got "
                f"{l1000_available.shape}"
            )
    return [
        int(d)
        for d in query_drugs
        if isinstance(d, (int, np.integer))
        and 0 <= int(d) < l1000_combined.shape[0]
        and l1000_available[int(d)]
    ]


@torch.no_grad()
def evaluate_e2i(fn_expr, fn_img, query_drugs, gallery_drugs,
                 gallery_rows, l1000_combined, multi_h5, n_crops, moa_sets):
    """Evaluate expression-to-image retrieval for L1000-backed query drugs.

    Drug IDs are row indices into ``l1000_combined``.  Do not use Python's
    ``d in l1000_combined`` here: for a 2-D NumPy array that tests whether the
    scalar value occurs anywhere in the expression matrix, rather than whether
    row ``d`` exists.
    """
    def build_img_embs(drug_list, rows_dict):
        embs = []
        for d in drug_list:
            if d not in rows_dict:
                embs.append(torch.zeros(1, EMBED_DIM, device=DEVICE)); continue
            cell, rows = rows_dict[d]
            sampled = np.sort(np.random.choice(rows, min(n_crops, len(rows)), replace=False))
            crops = multi_h5.read(cell, sampled).astype(np.float32) / 65535.0
            imgs = gpu_normalize(torch.from_numpy(crops).to(DEVICE))
            with autocast("cuda", enabled=USE_AMP): e = fn_img(imgs)
            embs.append(F.normalize(e.mean(0, keepdim=True).float(), dim=-1))
        return torch.cat(embs, 0) if embs else torch.zeros(0, EMBED_DIM, device=DEVICE)
    gallery_embs = build_img_embs(gallery_drugs, gallery_rows)
    gal_idx = {d: i for i, d in enumerate(gallery_drugs)}
    gal_moa = [moa_sets[d] for d in gallery_drugs]
    valid_qd = valid_l1000_drug_ids(query_drugs, l1000_combined)
    if not valid_qd: return {}
    q_expr = torch.from_numpy(l1000_combined[np.array(valid_qd)]).to(DEVICE)
    with autocast("cuda", enabled=USE_AMP):
        query_embs = F.normalize(fn_expr(q_expr).float(), dim=-1)
    labels_t   = torch.tensor([gal_idx.get(d, -1) for d in valid_qd], device=DEVICE)
    q_moa_list = [moa_sets[d] for d in valid_qd]
    m = topk_metrics(query_embs @ gallery_embs.T, labels_t, gal_moa,
                     valid_drug_mask=(labels_t >= 0), query_moa_list=q_moa_list)
    return {f"e2i_{k}": v for k, v in m.items()}

@torch.no_grad()
def evaluate_e2e(fn_expr, query_drugs, gallery_drugs, l1000_combined, moa_sets):
    all_expr = torch.from_numpy(l1000_combined[np.array(gallery_drugs)]).to(DEVICE)
    with autocast("cuda", enabled=USE_AMP): gallery_embs = F.normalize(fn_expr(all_expr).float(), dim=-1)
    q_expr = torch.from_numpy(l1000_combined[np.array(query_drugs)]).to(DEVICE)
    with autocast("cuda", enabled=USE_AMP): query_embs = F.normalize(fn_expr(q_expr).float(), dim=-1)
    gal_idx   = {d: i for i, d in enumerate(gallery_drugs)}
    labels_t  = torch.tensor([gal_idx.get(d, -1) for d in query_drugs], device=DEVICE)
    gal_moa   = [moa_sets[d] for d in gallery_drugs]
    q_moa_list = [moa_sets[d] for d in query_drugs]
    m = topk_metrics(query_embs @ gallery_embs.T, labels_t, gal_moa,
                     valid_drug_mask=(labels_t >= 0), query_moa_list=q_moa_list)
    return {f"e2e_{k}": v for k, v in m.items()}

@torch.no_grad()
def evaluate_i2i(fn_img, query_drugs, gallery_drugs, query_rows,
                 multi_h5, n_crops, moa_sets, gallery_rows_override=None):
    gal_rows = gallery_rows_override if gallery_rows_override is not None else query_rows
    gal_idx  = {d: i for i, d in enumerate(gallery_drugs)}
    gal_moa  = [moa_sets[d] for d in gallery_drugs]
    def build_embs(drug_list, rows_dict):
        embs = []
        for d in drug_list:
            if d not in rows_dict:
                embs.append(torch.zeros(1, EMBED_DIM, device=DEVICE)); continue
            cell, rows = rows_dict[d]
            sampled = np.sort(np.random.choice(rows, min(n_crops, len(rows)), replace=False))
            crops = multi_h5.read(cell, sampled).astype(np.float32) / 65535.0
            imgs  = gpu_normalize(torch.from_numpy(crops).to(DEVICE))
            with autocast("cuda", enabled=USE_AMP): e = fn_img(imgs)
            embs.append(F.normalize(e.mean(0, keepdim=True).float(), dim=-1))
        return torch.cat(embs, 0)
    gal_img  = build_embs(gallery_drugs, gal_rows)
    qry_img  = build_embs(query_drugs, query_rows)
    labels_t = torch.tensor([gal_idx.get(d, -1) for d in query_drugs], device=DEVICE)
    q_moa_list = [moa_sets[d] for d in query_drugs]
    m = topk_metrics(qry_img @ gal_img.T, labels_t, gal_moa,
                     valid_drug_mask=(labels_t >= 0), query_moa_list=q_moa_list)
    return {f"i2i_{k}": v for k, v in m.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# Training epoch
# ═══════════════════════════════════════════════════════════════════════════════

def run_epoch(img_bb, expr_bb,
              head_img_cpd, head_expr_cpd,
              head_img_i2i, head_expr_e2e,
              quad_temp, optimizer, scaler, sampler,
              train_rows, multi_h5, l1000_combined, moa_sets,
              valid_gene_i2i_indices, crispr_a549, crispr_u2os,
              gene_to_cpd_rows_a549, gene_to_cpd_rows_u2os,
              crispr_i2e_samples, gene_to_pos_drugs):

    img_bb.train(); expr_bb.train()
    head_img_cpd.train(); head_expr_cpd.train()
    head_img_i2i.train(); head_expr_e2e.train()

    losses = dict(total=0, i2e_cpd=0, i2i_moa=0, e2e_moa=0, moa_cross=0,
                  gene_i2i=0, crispr_i2e=0)

    loader = PrefetchLoaderV2(
        sampler,
        lambda b: fetch_batch_v2(b, train_rows, multi_h5, l1000_combined, N_CROPS_BATCH),
    )

    for batch_drugs, imgs_cpu, exprs_cpu, l1k_idx in tqdm(loader, total=len(loader), leave=False, ncols=80):
        imgs  = imgs_cpu.to(DEVICE, non_blocking=True)
        exprs = exprs_cpu.to(DEVICE, non_blocking=True) if exprs_cpu.numel() > 0 else None

        l1k_drugs = [batch_drugs[i] for i in l1k_idx]
        l1k_idx_t = torch.tensor(l1k_idx, dtype=torch.long, device=DEVICE)
        B = len(batch_drugs); B_l1k = len(l1k_idx)

        batch_moa_all = [moa_sets[d] for d in batch_drugs]
        batch_moa_l1k = [moa_sets[d] for d in l1k_drugs]

        with autocast("cuda", enabled=USE_AMP):
            _, K, C, H, W = imgs.shape
            imgs_norm = gpu_normalize(imgs.reshape(B*K, C, H, W)).reshape(B, K, C, H, W)
            if callable(globals().get("DRUG_IMAGE_ENCODER")):
                trunk_img_all = DRUG_IMAGE_ENCODER(img_bb, imgs_norm)
            else:
                trunk_img_all = img_bb(imgs_norm[:, 0], augment=True)

            zero = torch.zeros([], device=DEVICE)

            # ── CPD: image ↔ L1000 (仅有L1000的药物) ──────────────────────
            loss_i2e_cpd = zero
            if B_l1k >= 2 and exprs is not None:
                trunk_expr_l1k = expr_bb(exprs)
                z_img_cpd  = head_img_cpd(trunk_img_all[l1k_idx_t])
                z_expr_cpd = head_expr_cpd(trunk_expr_l1k)
                drug_id_l1k = torch.eye(B_l1k, dtype=torch.bool, device=DEVICE)
                loss_i2e_cpd = cross_supcon(z_img_cpd, z_expr_cpd, drug_id_l1k,
                                            quad_temp.log_temp_cpd)
            elif exprs is not None and B_l1k >= 1:
                trunk_expr_l1k = expr_bb(exprs)
            else:
                trunk_expr_l1k = None

            # ── MOA i2i: 所有 batch drugs（含无L1000的新化合物）────────────
            moa_mask_all  = make_moa_mask(batch_moa_all, DEVICE)
            has_moa_all   = moa_mask_all.any()
            z_img_i2i_all = head_img_i2i(trunk_img_all)
            loss_i2i = supcon(z_img_i2i_all, moa_mask_all,
                              quad_temp.log_temp_i2i) if has_moa_all else zero

            # ── e2e MOA: 仅 L1K drugs ──────────────────────────────────────
            z_expr_e2e_l1k = None
            loss_e2e = zero
            if trunk_expr_l1k is not None and B_l1k >= 2:
                moa_mask_l1k = make_moa_mask(batch_moa_l1k, DEVICE)
                if moa_mask_l1k.any():
                    z_expr_e2e_l1k = head_expr_e2e(trunk_expr_l1k)
                    loss_e2e = supcon(z_expr_e2e_l1k, moa_mask_l1k, quad_temp.log_temp_e2e)

            # ── moa_cross: 图像 → L1K 表达（表达侧 stop-gradient，表达做 teacher）──
            #   表达空间由纯 e2e 塑造保持稳定，图像单向向其对齐，
            #   防止后期变强的图像信号把 e2e 的 MOA 聚类拽偏（run1 e2e 退化根因）
            loss_moa_cross = zero
            if trunk_expr_l1k is not None and B_l1k >= 1 and has_moa_all:
                if z_expr_e2e_l1k is None:
                    z_expr_e2e_l1k = head_expr_e2e(trunk_expr_l1k)
                moa_mask_cross = make_moa_mask_asym(batch_drugs, l1k_drugs, moa_sets, DEVICE)
                if moa_mask_cross.any():
                    loss_moa_cross = cross_supcon(z_img_i2i_all, z_expr_e2e_l1k.detach(),
                                                  moa_mask_cross, quad_temp.log_temp_moa_cross)

            # ── gene i2i: CRISPR(基因X) ↔ cpg0000化合物(靶基因X) ───────────
            loss_gene_i2i = zero
            if valid_gene_i2i_indices:
                gi_batch = random.sample(valid_gene_i2i_indices,
                                         min(GENE_I2I_PER_BATCH, len(valid_gene_i2i_indices)))
                gi_cr, gi_cpd, gi_pos = fetch_gene_i2i_batch(
                    gi_batch, crispr_a549, crispr_u2os,
                    gene_to_cpd_rows_a549, gene_to_cpd_rows_u2os, multi_h5)
                if gi_cr is not None:
                    gi_cr_norm  = gpu_normalize(gi_cr.to(DEVICE))
                    gi_cpd_norm = gpu_normalize(gi_cpd.to(DEVICE))
                    gi_cr_emb  = head_img_i2i(img_bb(gi_cr_norm,  augment=True))
                    gi_cpd_emb = head_img_i2i(img_bb(gi_cpd_norm, augment=True))
                    loss_gene_i2i = cross_supcon(gi_cr_emb, gi_cpd_emb,
                                                 gi_pos.to(DEVICE), quad_temp.log_temp_i2i)

            # ── crispr i2e: CRISPR图像(基因X) → 药物L1000(靶向X或同MOA) ──────
            #   表达侧 detach（药物L1000空间做teacher），只训练CRISPR图像编码器
            #   向其对齐，直接补 CRISPR→药物L1000 短板
            loss_crispr_i2e = zero
            if crispr_i2e_samples:
                ci_batch = random.sample(crispr_i2e_samples,
                                         min(CRISPR_I2E_PER_BATCH, len(crispr_i2e_samples)))
                ci_imgs, ci_gids, ci_genes = fetch_crispr_i2e_batch(
                    ci_batch, crispr_a549, gene_to_pos_drugs)
                if ci_imgs is not None:
                    ci_img_norm   = gpu_normalize(ci_imgs.to(DEVICE))
                    ci_crispr_emb = head_img_i2i(img_bb(ci_img_norm, augment=True))
                    ci_drug_l1k   = torch.from_numpy(
                        l1000_combined[np.array(ci_gids)]).to(DEVICE)
                    ci_drug_emb   = head_expr_e2e(expr_bb(ci_drug_l1k))
                    K = len(ci_genes)
                    ci_pos = torch.zeros(K, K, dtype=torch.bool, device=DEVICE)
                    for i, g in enumerate(ci_genes):
                        posset = gene_to_pos_drugs[g]
                        for j, gid in enumerate(ci_gids):
                            if gid in posset: ci_pos[i, j] = True
                    if ci_pos.any():
                        loss_crispr_i2e = cross_supcon(ci_crispr_emb, ci_drug_emb.detach(),
                                                       ci_pos, quad_temp.log_temp_moa_cross)

            loss = (ALPHA_CPD        * loss_i2e_cpd
                  + ALPHA_MOA_I      * loss_i2i
                  + ALPHA_MOA_E      * loss_e2e
                  + ALPHA_MOA_CROSS  * loss_moa_cross
                  + ALPHA_GENE_I2I   * loss_gene_i2i
                  + ALPHA_CRISPR_I2E * loss_crispr_i2e)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        all_params = (list(img_bb.parameters()) + list(expr_bb.parameters()) +
                      list(head_img_cpd.parameters()) + list(head_expr_cpd.parameters()) +
                      list(head_img_i2i.parameters()) + list(head_expr_e2e.parameters()) +
                      list(quad_temp.parameters()))
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        scaler.step(optimizer); scaler.update()

        losses["total"]     += loss.item()
        losses["i2e_cpd"]   += loss_i2e_cpd.item()
        losses["i2i_moa"]   += loss_i2i.item()
        losses["e2e_moa"]   += loss_e2e.item()
        losses["moa_cross"] += loss_moa_cross.item()
        losses["gene_i2i"]  += loss_gene_i2i.item()
        losses["crispr_i2e"]+= loss_crispr_i2e.item()

    for k in losses: losses[k] /= K_PER_EPOCH
    return losses


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    global L1000_AVAILABLE_IDS
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda": print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Meta & MOA split ──────────────────────────────────────────────────────
    metas = {cell: pd.read_csv(META[cell]) for cell in ["U2OS","A549","A375","MCF7","HT29"]}

    def moa_split(df, label_col, seed=42):
        rng = np.random.default_rng(seed)
        eval_set = set()
        for moa, grp in df[df['moa'].notna()].groupby('moa'):
            idxs = grp[label_col].tolist()
            if len(idxs) >= 4:
                eval_set.update(rng.choice(idxs, max(1, len(idxs)//2), replace=False).tolist())
        return sorted([l for l in df[label_col].tolist() if l not in eval_set]), sorted(eval_set)

    # U2OS: 排除 Tocris
    prof = pd.read_csv(PROFILES_MERGED, low_memory=False)
    tocris_names = set(prof[prof['Metadata_source_name'] == 'Tocris Bioscience']
                       ['Metadata_cpd_name'].str.lower())
    metas["U2OS"]['is_tocris'] = metas["U2OS"]['cmap_name'].str.lower().isin(tocris_names)
    rng42 = np.random.default_rng(42)
    u2os_remain = [i for i in range(690) if not metas["U2OS"].iloc[i]['is_tocris']]
    u2os_eval_global = set()
    remain_df = metas["U2OS"].iloc[sorted(u2os_remain)].copy()
    remain_df['global_label'] = sorted(u2os_remain)
    for moa, grp in remain_df[remain_df['moa'].notna()].groupby('moa'):
        idxs = grp['global_label'].tolist()
        if len(idxs) >= 4:
            u2os_eval_global.update(rng42.choice(idxs, max(1, len(idxs)//2), replace=False).tolist())
    u2os_train_global = [i for i in u2os_remain if i not in u2os_eval_global]

    # A549
    meta_a = metas["A549"].rename(columns={"new_int_label": "local_label"})
    a549_train_local, a549_eval_local = moa_split(meta_a, "local_label")
    a549_train_global = [l + OFFSET["A549"] for l in a549_train_local]
    a549_eval_global  = [l + OFFSET["A549"] for l in a549_eval_local]

    # A375 / MCF7
    a375_train_local, a375_eval_local = moa_split(metas["A375"], "new_int_label")
    mcf7_train_local,  mcf7_eval_local  = moa_split(metas["MCF7"],  "new_int_label")
    a375_train_global = [l + OFFSET["A375"] for l in a375_train_local]
    a375_eval_global  = [l + OFFSET["A375"] for l in a375_eval_local]
    mcf7_train_global = [l + OFFSET["MCF7"] for l in mcf7_train_local]
    mcf7_eval_global  = [l + OFFSET["MCF7"] for l in mcf7_eval_local]

    ht29_eval_global = [r + OFFSET["HT29"] for r in metas["HT29"]["new_int_label"].tolist()]

    # cpg0030
    train30_df = pd.read_csv(META_D / "cpg0030_train_split.csv")
    train30_intlbls = set(int(x) for x in train30_df["int_label"].tolist() if int(x) >= 0)
    train30_sorted  = sorted(train30_intlbls)
    cpg30_intlbl_to_global = {ilbl: OFFSET["cpg0030"] + i for i, ilbl in enumerate(train30_sorted)}
    cpg0030_train_global   = list(cpg30_intlbl_to_global.values())

    # cpg0000 compound A549/U2OS（全量参与训练，无L1000）
    cpd_a549_meta = pd.read_csv(CPD_DIR / "meta_cpg0000_cpd_A549.csv")
    cpd_u2os_meta = pd.read_csv(CPD_DIR / "meta_cpg0000_cpd_U2OS.csv")
    cpd_a549_train_global = [OFFSET["cpd_A549"] + i for i in range(N_CPD_NEW["cpd_A549"])]
    cpd_u2os_train_global = [OFFSET["cpd_U2OS"] + i for i in range(N_CPD_NEW["cpd_U2OS"])]

    historical_total = OFFSET["cpd_U2OS"] + N_CPD_NEW["cpd_U2OS"]
    expansion_metas = {}
    expansion_train_global = []
    N_TOTAL = historical_total
    if TRAINING_EXPANSION:
        expected_offset = historical_total
        for key, cfg in TRAINING_EXPANSION.items():
            if int(cfg["offset"]) != expected_offset:
                raise RuntimeError(
                    f"Expansion {key} offset={cfg['offset']}, expected {expected_offset}"
                )
            frame = pd.read_csv(cfg["meta"])
            label_col = cfg.get("label_col", "int_label")
            labels = frame[label_col].astype(int).tolist()
            n = int(cfg["n"])
            if labels != list(range(n)):
                raise RuntimeError(f"Expansion {key} labels are not contiguous 0..{n-1}")
            expansion_metas[key] = frame
            expansion_train_global.extend(range(expected_offset, expected_offset + n))
            expected_offset += n
        N_TOTAL = expected_offset
    print(f"N_TOTAL = {N_TOTAL}")

    # ── MOA sets ──────────────────────────────────────────────────────────────
    moa_sets = [set() for _ in range(N_TOTAL)]

    for cell in ["U2OS","A549","A375","MCF7","HT29"]:
        off = OFFSET[cell]; df = metas[cell]
        for i, row in df.iterrows():
            g = i if cell == "U2OS" else int(row["new_int_label"]) + off
            moa_str = str(row["moa"]) if pd.notna(row.get("moa","")) else ""
            moa_sets[g] = set(m.strip() for m in moa_str.split("|") if m.strip())

    for _, row in train30_df.iterrows():
        ilbl = int(row["int_label"])
        if ilbl not in cpg30_intlbl_to_global: continue
        g = cpg30_intlbl_to_global[ilbl]
        moa_str = str(row["moa"]) if pd.notna(row.get("moa","")) else ""
        moa_sets[g] = set(m.strip() for m in moa_str.split("|") if m.strip())

    # cpg0000 化合物 MOA（按名称匹配 compoundinfo_beta）
    # cpg0000 化合物 MOA：按基因靶点查，保证同基因两个化合物 MOA 一致
    cpd_beta = pd.read_csv(META_D / "compoundinfo_beta.txt", sep="\t", low_memory=False)
    cpd_beta["target_up"] = cpd_beta["target"].fillna("").str.upper()
    gene_to_moa = {}
    for gene in set(cpd_a549_meta["gene"].tolist() + cpd_u2os_meta["gene"].tolist()):
        hits = cpd_beta[cpd_beta["target_up"].str.contains(rf"\b{gene.upper()}\b", regex=True, na=False)]
        moas = hits["moa"].dropna()
        if len(moas) > 0:
            gene_to_moa[gene] = moas.value_counts().index[0]
    print(f"cpg0000 化合物 gene→MOA 覆盖: {len(gene_to_moa)}/130 基因")

    for meta_df, off_key in [(cpd_a549_meta, "cpd_A549"), (cpd_u2os_meta, "cpd_U2OS")]:
        for _, row in meta_df.iterrows():
            g = OFFSET[off_key] + int(row["int_label"])
            moa_str = gene_to_moa.get(str(row["gene"]), "")
            if moa_str:
                moa_sets[g] = set(m.strip() for m in moa_str.split("|") if m.strip())

    for key, frame in expansion_metas.items():
        cfg = TRAINING_EXPANSION[key]
        label_col = cfg.get("label_col", "int_label")
        for _, row in frame.iterrows():
            g = int(cfg["offset"]) + int(row[label_col])
            moa_str = str(row["moa"]) if pd.notna(row.get("moa", "")) else ""
            moa_sets[g] = set(m.strip() for m in moa_str.split("|") if m.strip())

    all_train_drugs = sorted(
        u2os_train_global + a549_train_global +
        a375_train_global + mcf7_train_global +
        cpg0030_train_global +
        cpd_a549_train_global + cpd_u2os_train_global +
        expansion_train_global
    )
    print(f"Train: U2OS={len(u2os_train_global)} A549={len(a549_train_global)} "
          f"A375={len(a375_train_global)} MCF7={len(mcf7_train_global)} "
          f"cpg0030={len(cpg0030_train_global)} "
          f"cpd_A549={len(cpd_a549_train_global)} cpd_U2OS={len(cpd_u2os_train_global)} "
          f"expansion={len(expansion_train_global)} "
          f"total={len(all_train_drugs)}")

    # ── L1000 ────────────────────────────────────────────────────────────────
    l1k = {cell: np.load(L1K[cell]).astype(np.float32) for cell in ["U2OS","A549","A375","MCF7","HT29"]}
    IN_DIM = l1k["U2OS"].shape[1]
    l1000_combined = np.zeros((N_TOTAL, IN_DIM), dtype=np.float32)
    for cell in ["U2OS","A549","A375","MCF7","HT29"]:
        off = OFFSET[cell]; n = N_CPD[cell]
        l1000_combined[off:off+n] = l1k[cell]
    for key, cfg in TRAINING_EXPANSION.items():
        expression = np.load(cfg["l1k"]).astype(np.float32)
        expected_shape = (int(cfg["n"]), IN_DIM)
        if expression.shape != expected_shape:
            raise RuntimeError(
                f"Expansion {key} L1000 shape={expression.shape}, expected={expected_shape}"
            )
        off = int(cfg["offset"])
        l1000_combined[off:off + len(expression)] = expression
    if TRAINING_EXPANSION:
        L1000_AVAILABLE_IDS = set(
            np.flatnonzero(
                np.isfinite(l1000_combined).all(axis=1)
                & np.any(l1000_combined != 0, axis=1)
            ).tolist()
        )
    print(f"L1000: {l1000_combined.shape}")

    # ── H5 ───────────────────────────────────────────────────────────────────
    multi_h5 = MultiH5Reader()
    multi_h5.load_cell("U2OS",  H5["U2OS"],  max_crops_per_label=150)
    multi_h5.load_cell("A549",  H5["A549"],  max_crops_per_label=150)
    multi_h5.load_cell("A375",  H5["A375"],  max_crops_per_label=150)
    multi_h5.load_cell("MCF7",  H5["MCF7"],  max_crops_per_label=150)
    multi_h5.load_cpg0030(H5["cpg0030"],
                          PROC / "meta_cpg0030_u2os.csv",
                          train30_intlbls, cpg30_intlbl_to_global, max_crops=100)
    # cpg0000 化合物（global_id = label + offset）
    multi_h5.load_with_global_offset("cpd_A549", H5["cpd_A549"], "labels", OFFSET["cpd_A549"], max_crops_per_label=150)
    multi_h5.load_with_global_offset("cpd_U2OS", H5["cpd_U2OS"], "labels", OFFSET["cpd_U2OS"], max_crops_per_label=150)
    for key, cfg in TRAINING_EXPANSION.items():
        multi_h5.load_with_global_offset(
            key, cfg["h5"], "labels", int(cfg["offset"]),
            max_crops_per_label=150,
        )

    # ── drug→rows 构建 ────────────────────────────────────────────────────────
    a549_new_to_h5 = dict(zip(metas["A549"]["new_int_label"].astype(int),
                               metas["A549"]["int_label"].astype(int)))
    inv_a549 = {v: k for k, v in a549_new_to_h5.items()}

    all_rows_raw = {}
    for h5_lbl, rows in multi_h5.get_label_to_rows("U2OS").items():
        all_rows_raw[h5_lbl] = ("U2OS", rows)
    for h5_lbl, rows in multi_h5.get_label_to_rows("A549").items():
        new_lbl = inv_a549.get(h5_lbl)
        if new_lbl is not None:
            all_rows_raw[new_lbl + OFFSET["A549"]] = ("A549", rows)
    for cell in ["A375", "MCF7"]:
        for h5_lbl, rows in multi_h5.get_label_to_rows(cell).items():
            all_rows_raw[h5_lbl + OFFSET[cell]] = (cell, rows)
    for g_id, rows in multi_h5.get_label_to_rows("cpg0030").items():
        all_rows_raw[g_id] = ("cpg0030", rows)
    for cell_key in ["cpd_A549", "cpd_U2OS"]:
        for g_id, rows in multi_h5.get_label_to_rows(cell_key).items():
            all_rows_raw[g_id] = (cell_key, rows)
    for key in TRAINING_EXPANSION:
        for g_id, rows in multi_h5.get_label_to_rows(key).items():
            all_rows_raw[g_id] = (key, rows)

    train_rows, eval_rows_split = split_drug_crops(
        {d: all_rows_raw[d] for d in all_train_drugs if d in all_rows_raw},
        TRAIN_CROP_RATIO, RANDOM_SEED)

    eval_rows_all = dict(train_rows)
    for d in (list(u2os_eval_global) + list(a549_eval_global) +
              list(a375_eval_global) + list(mcf7_eval_global)):
        if d in all_rows_raw:
            eval_rows_all[d] = all_rows_raw[d]

    rng_i2i = np.random.default_rng(RANDOM_SEED + 1)
    i2i_query_rows = {}; i2i_gallery_rows = {}
    eval_drugs_all = (list(u2os_eval_global) + list(a549_eval_global) +
                      list(a375_eval_global) + list(mcf7_eval_global))
    for d in eval_drugs_all:
        if d not in all_rows_raw: continue
        cell, rows = all_rows_raw[d]
        perm = rng_i2i.permutation(len(rows))
        n_half = max(1, len(rows) // 2)
        i2i_query_rows[d]   = (cell, rows[perm[:n_half]])
        i2i_gallery_rows[d] = (cell, rows[perm[n_half:]] if len(rows) > 1 else rows[perm[:1]])
    i2i_query_rows.update(eval_rows_split)
    i2i_gallery_rows.update(train_rows)

    valid_train     = [d for d in all_train_drugs if d in train_rows]
    valid_eval_u2os = [d for d in u2os_eval_global if d in all_rows_raw and moa_sets[d]]
    valid_eval_a549 = [d for d in a549_eval_global  if d in all_rows_raw and moa_sets[d]]

    u2os_gallery = list(range(OFFSET["U2OS"], OFFSET["U2OS"] + N_CPD["U2OS"]))
    a549_gallery = list(range(OFFSET["A549"], OFFSET["A549"] + N_CPD["A549"]))

    # ── CRISPR 加载（从 H5，held-out A549=10 / U2OS=8）────────────────────────
    print("\n加载 CRISPR crops...")
    crispr_a549_meta_df = pd.read_csv(CRISPR_DIR / "meta_cpg0000_A549.csv")
    crispr_u2os_meta_df = pd.read_csv(CRISPR_DIR / "meta_cpg0000_U2OS.csv")

    crispr_a549, a549_gene_names = load_crispr_h5(
        CRISPR_DIR / "crops_cpg0000_A549.h5",
        CRISPR_DIR / "meta_cpg0000_A549.csv",
        held_out_genes=HELD_OUT_GENES_A549,
        n_per_gene=N_CRISPR_CACHE)
    crispr_u2os, u2os_gene_names = load_crispr_h5(
        CRISPR_DIR / "crops_cpg0000_U2OS.h5",
        CRISPR_DIR / "meta_cpg0000_U2OS.csv",
        held_out_genes=HELD_OUT_GENES_U2OS,
        n_per_gene=N_CRISPR_CACHE)

    # ── 基因-化合物配对映射 ────────────────────────────────────────────────────
    gene_to_a549_crispr_idx = {row['gene']: row['int_label']
                                for _, row in crispr_a549_meta_df.iterrows()}
    gene_to_u2os_crispr_idx = {row['gene']: row['int_label']
                                for _, row in crispr_u2os_meta_df.iterrows()}

    cpd_a549_rows_by_label = multi_h5.get_label_to_rows("cpd_A549")  # global_id → rows
    cpd_u2os_rows_by_label = multi_h5.get_label_to_rows("cpd_U2OS")

    # gene_to_cpd_rows: crispr_gene_idx → [(cpd_cell, rows), ...]
    gene_to_cpd_rows_a549 = {}
    for _, row in cpd_a549_meta.iterrows():
        gene = row['gene']
        gi   = gene_to_a549_crispr_idx.get(gene)
        if gi is None or gene in HELD_OUT_GENES_A549: continue
        g_id = OFFSET["cpd_A549"] + int(row["int_label"])
        if g_id not in cpd_a549_rows_by_label: continue
        gene_to_cpd_rows_a549.setdefault(gi, []).append(
            ("cpd_A549", cpd_a549_rows_by_label[g_id]))

    gene_to_cpd_rows_u2os = {}
    for _, row in cpd_u2os_meta.iterrows():
        gene = row['gene']
        gi   = gene_to_u2os_crispr_idx.get(gene)
        u2os_held = (HELD_OUT_GENES_U2OS == "ALL") or (gene in HELD_OUT_GENES_U2OS)
        if gi is None or u2os_held: continue
        g_id = OFFSET["cpd_U2OS"] + int(row["int_label"])
        if g_id not in cpd_u2os_rows_by_label: continue
        gene_to_cpd_rows_u2os.setdefault(gi, []).append(
            ("cpd_U2OS", cpd_u2os_rows_by_label[g_id]))

    valid_gene_i2i_indices = (
        [("A549", gi) for gi in gene_to_cpd_rows_a549 if gi in crispr_a549] +
        [("U2OS", gi) for gi in gene_to_cpd_rows_u2os if gi in crispr_u2os]
    )
    print(f"gene i2i 有效基因: A549={sum(1 for c,_ in valid_gene_i2i_indices if c=='A549')} "
          f"U2OS={sum(1 for c,_ in valid_gene_i2i_indices if c=='U2OS')} "
          f"total={len(valid_gene_i2i_indices)}")

    # ── crispr i2e: CRISPR基因 → 正样本A549药物L1000 (靶向+同MOA) ──────────────
    #   正样本定义与评估脚本 build_gene_pos 一致：靶向同基因的药物 ∪ 同MOA药物
    a549_name2gid, a549_gid2moa = {}, {}
    for _, row in metas["A549"].iterrows():
        gid = int(row["new_int_label"]) + OFFSET["A549"]
        a549_name2gid.setdefault(str(row["cmap_name"]).lower().strip(), gid)
        a549_gid2moa[gid] = moa_sets[gid]
    # 药物名 → 靶点基因(大写)
    beta_name2tgts = {}
    for _, r in cpd_beta.iterrows():
        if r["target_up"]:
            beta_name2tgts.setdefault(str(r["cmap_name"]).lower().strip(), set()).update(
                g.strip() for g in r["target_up"].split("|") if g.strip())
    # 基因(大写) → 直接靶向的A549药物gid
    gene_to_direct_gid = {}
    for name, gid in a549_name2gid.items():
        for t in beta_name2tgts.get(name, set()):
            gene_to_direct_gid.setdefault(t, set()).add(gid)
    # CRISPR训练基因 → 正样本药物(靶向 ∪ 同MOA)
    gene_to_pos_drugs = {}
    for gene, gi in gene_to_a549_crispr_idx.items():
        if gene in HELD_OUT_GENES_A549 or gi not in crispr_a549: continue
        direct = gene_to_direct_gid.get(gene.upper(), set())
        moas = set()
        for gid in direct: moas |= a549_gid2moa.get(gid, set())
        allpos = set(direct)
        if moas:
            allpos |= {gid for gid, ms in a549_gid2moa.items() if ms & moas}
        if allpos:
            gene_to_pos_drugs[gene] = allpos
    crispr_i2e_samples = [(gene_to_a549_crispr_idx[g], g) for g in gene_to_pos_drugs]
    if crispr_i2e_samples:
        avg = np.mean([len(v) for v in gene_to_pos_drugs.values()])
        print(f"crispr i2e: {len(crispr_i2e_samples)}个训练基因有正样本药物, 平均{avg:.0f}个药物/基因")

    # ── 模型 ──────────────────────────────────────────────────────────────────
    img_bb        = ImageBackbone(IMG_CHANNELS, VIT_PATCH, 128, VIT_DIM, VIT_DEPTH, VIT_HEADS, VIT_MLP_RATIO).to(DEVICE)
    expr_bb       = ExprBackbone(IN_DIM, EXPR_TRUNK).to(DEVICE)
    head_img_cpd  = ProjHead(VIT_DIM,    EMBED_DIM).to(DEVICE)
    head_expr_cpd = ProjHead(EXPR_TRUNK, EMBED_DIM).to(DEVICE)
    head_img_i2i  = ProjHead(VIT_DIM,    EMBED_DIM).to(DEVICE)
    head_expr_e2e = ProjHead(EXPR_TRUNK, EMBED_DIM).to(DEVICE)
    quad_temp     = QuadTemp(INIT_TEMP).to(DEVICE)

    # ── 优化器 ────────────────────────────────────────────────────────────────
    all_params = (list(img_bb.parameters()) + list(expr_bb.parameters()) +
                  list(head_img_cpd.parameters()) + list(head_expr_cpd.parameters()) +
                  list(head_img_i2i.parameters()) + list(head_expr_e2e.parameters()) +
                  list(quad_temp.parameters()))
    optimizer = torch.optim.AdamW(all_params, lr=LR, weight_decay=WEIGHT_DECAY)
    scaler    = GradScaler(enabled=USE_AMP)
    def lr_lambda(ep):
        if ep < WARMUP: return (ep + 1) / WARMUP
        prog = (ep - WARMUP) / max(1, EPOCHS - WARMUP)
        return max(1e-6 / LR, 0.5 * (1 + math.cos(math.pi * prog)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if USE_MOA_BALANCED_SAMPLER:
        sampler = MoaBalancedDrugSampler(
            valid_train, BATCH_DRUGS, K_PER_EPOCH, moa_sets,
            min_per_moa=MIN_DRUGS_PER_MOA,
            balanced_fraction=MOA_BALANCED_FRACTION,
        )
    else:
        sampler = RandomDrugSampler(valid_train, BATCH_DRUGS, K_PER_EPOCH)

    fn_img_i2i  = lambda x: head_img_i2i(img_bb(x))
    fn_expr_e2e = lambda x: head_expr_e2e(expr_bb(x))

    best = dict(i2i=0., e2e=0., i2e_moa=0., combined=0.)
    history = []
    start_epoch = 1

    # ── Resume ────────────────────────────────────────────────────────────────
    resume_ckpt = LOG_DIR / "checkpoint_ep020.pt"
    if resume_ckpt.exists():
        print(f"Resuming from {resume_ckpt} ...")
        ckpt = torch.load(resume_ckpt, map_location=DEVICE)
        for m, k in [(img_bb,"img_bb"),(expr_bb,"expr_bb"),
                     (head_img_cpd,"head_img_cpd"),(head_expr_cpd,"head_expr_cpd"),
                     (head_img_i2i,"head_img_i2i"),(head_expr_e2e,"head_expr_e2e"),
                     (quad_temp,"quad_temp")]:
            m.load_state_dict(ckpt[k])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        hist_csv = LOG_DIR / "train_history.csv"
        if hist_csv.exists():
            hist_df = pd.read_csv(hist_csv)
            history = hist_df.to_dict("records")
            best["i2i"]      = max((r["u2os_i2i"] + r["a549_i2i"]) / 2 for r in history)
            best["e2e"]      = max((r["u2os_e2e"] + r["a549_e2e"]) / 2 for r in history)
            best["i2e_moa"]  = max((r["u2os_i2e"] + r["a549_i2e"]) / 2 for r in history)
            best["combined"] = max(r["combined"] for r in history)
        print(f"  Resumed ep={ckpt['epoch']}, start={start_epoch}")
        print(f"  best: i2i={best['i2i']*100:.1f}% e2e={best['e2e']*100:.1f}%")
        del ckpt

    def save_ckpt(tag, epoch, metrics_dict):
        path = LOG_DIR / f"best_{tag}.pt"
        torch.save({"epoch": epoch, "tag": tag, **metrics_dict,
                    "img_bb": img_bb.state_dict(), "expr_bb": expr_bb.state_dict(),
                    "head_img_cpd": head_img_cpd.state_dict(), "head_expr_cpd": head_expr_cpd.state_dict(),
                    "head_img_i2i": head_img_i2i.state_dict(), "head_expr_e2e": head_expr_e2e.state_dict(),
                    "quad_temp": quad_temp.state_dict()}, path)
        vals = "  ".join(f"{k}={v*100:.1f}%" for k, v in metrics_dict.items())
        print(f"  -> best_{tag}  ep={epoch}  {vals}")

    def do_eval(epoch):
        for m in [img_bb, expr_bb, head_img_cpd, head_expr_cpd, head_img_i2i, head_expr_e2e]: m.eval()

        i2i_u = evaluate_i2i(fn_img_i2i, valid_eval_u2os, u2os_gallery,
                              i2i_query_rows, multi_h5, N_CROPS_EVAL, moa_sets,
                              gallery_rows_override=i2i_gallery_rows)
        i2i_a = evaluate_i2i(fn_img_i2i, valid_eval_a549, a549_gallery,
                              i2i_query_rows, multi_h5, N_CROPS_EVAL, moa_sets,
                              gallery_rows_override=i2i_gallery_rows)
        e2e_u = evaluate_e2e(fn_expr_e2e, valid_eval_u2os, u2os_gallery, l1000_combined, moa_sets)
        e2e_a = evaluate_e2e(fn_expr_e2e, valid_eval_a549, a549_gallery, l1000_combined, moa_sets)
        i2e_u = evaluate_i2e(fn_img_i2i, fn_expr_e2e, valid_eval_u2os, i2i_query_rows,
                              u2os_gallery, l1000_combined, multi_h5, N_CROPS_EVAL, moa_sets)
        i2e_a = evaluate_i2e(fn_img_i2i, fn_expr_e2e, valid_eval_a549, i2i_query_rows,
                              a549_gallery, l1000_combined, multi_h5, N_CROPS_EVAL, moa_sets)
        e2i_u = evaluate_e2i(fn_expr_e2e, fn_img_i2i, valid_eval_u2os, u2os_gallery,
                              i2i_gallery_rows, l1000_combined, multi_h5, N_CROPS_EVAL, moa_sets)
        e2i_a = evaluate_e2i(fn_expr_e2e, fn_img_i2i, valid_eval_a549, a549_gallery,
                              i2i_gallery_rows, l1000_combined, multi_h5, N_CROPS_EVAL, moa_sets)

        i2i_u5 = i2i_u.get("i2i_m_top5",0); i2i_a5 = i2i_a.get("i2i_m_top5",0)
        e2e_u5 = e2e_u.get("e2e_m_top5",0); e2e_a5 = e2e_a.get("e2e_m_top5",0)
        i2e_u5 = i2e_u.get("i2e_m_top5",0); i2e_a5 = i2e_a.get("i2e_m_top5",0)
        e2i_u5 = e2i_u.get("e2i_m_top5",0); e2i_a5 = e2i_a.get("e2i_m_top5",0)

        print(f"  [E{epoch}] U2OS i2i={i2i_u5*100:.1f}% e2e={e2e_u5*100:.1f}% i2e={i2e_u5*100:.1f}% e2i={e2i_u5*100:.1f}%")
        print(f"  [E{epoch}] A549 i2i={i2i_a5*100:.1f}% e2e={e2e_a5*100:.1f}% i2e={i2e_a5*100:.1f}% e2i={e2i_a5*100:.1f}%")

        combined = (i2i_u5 + i2i_a5 + e2e_u5 + e2e_a5 + i2e_u5 + i2e_a5 + e2i_u5 + e2i_a5) / 8
        print(f"  [E{epoch}] combined={combined*100:.1f}%  "
              f"temp_cpd={quad_temp.temp_cpd:.4f} temp_i2i={quad_temp.temp_i2i:.4f} "
              f"temp_e2e={quad_temp.temp_e2e:.4f} temp_cross={quad_temp.temp_moa_cross:.4f}")
        return i2i_u5, i2i_a5, e2e_u5, e2e_a5, i2e_u5, i2e_a5, e2i_u5, e2i_a5, combined

    # ── 训练循环 ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"moax_v7: {len(valid_train)} train drugs, {EPOCHS} epochs")
    print(f"  gene i2i pairs: {len(valid_gene_i2i_indices)} 基因")
    print(f"  qualified checkpoint 条件: e2e≥{THRESH_E2E_U2OS*100:.1f}%/{THRESH_E2E_A549*100:.1f}% "
          f"AND i2i≥{THRESH_I2I_U2OS*100:.1f}%/{THRESH_I2I_A549*100:.1f}%")
    print(f"{'='*60}")

    for epoch in range(start_epoch, min(EPOCHS, globals().get("STOP_AFTER_EPOCH", EPOCHS)) + 1):
        t0 = time.time()
        losses = run_epoch(img_bb, expr_bb, head_img_cpd, head_expr_cpd,
                           head_img_i2i, head_expr_e2e, quad_temp, optimizer, scaler, sampler,
                           train_rows, multi_h5, l1000_combined, moa_sets,
                           valid_gene_i2i_indices, crispr_a549, crispr_u2os,
                           gene_to_cpd_rows_a549, gene_to_cpd_rows_u2os,
                           crispr_i2e_samples, gene_to_pos_drugs)
        scheduler.step()
        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]
        print(f"\n[E{epoch:3d}/{EPOCHS}] "
              f"loss={losses['total']:.4f} "
              f"(cpd={losses['i2e_cpd']:.3f} i2i={losses['i2i_moa']:.3f} "
              f"e2e={losses['e2e_moa']:.3f} cross={losses['moa_cross']:.3f} "
              f"gene={losses['gene_i2i']:.3f} cr_i2e={losses['crispr_i2e']:.3f})"
              f"  lr={lr_now:.1e}  ({elapsed:.0f}s)")

        i2i_u5, i2i_a5, e2e_u5, e2e_a5, i2e_u5, i2e_a5, e2i_u5, e2i_a5, combined = do_eval(epoch)

        row = {"epoch": epoch, **{f"loss_{k}": v for k, v in losses.items()}, "lr": lr_now,
               "temp_cpd": quad_temp.temp_cpd, "temp_i2i": quad_temp.temp_i2i,
               "temp_e2e": quad_temp.temp_e2e, "temp_moa_cross": quad_temp.temp_moa_cross,
               "u2os_i2i": i2i_u5, "a549_i2i": i2i_a5,
               "u2os_e2e": e2e_u5, "a549_e2e": e2e_a5,
               "u2os_i2e": i2e_u5, "a549_i2e": i2e_a5,
               "u2os_e2i": e2i_u5, "a549_e2i": e2i_a5, "combined": combined}
        history.append(row)
        pd.DataFrame(history).to_csv(LOG_DIR / "train_history.csv", index=False)

        base_m = {"u2os_i2i": i2i_u5, "a549_i2i": i2i_a5,
                  "u2os_e2e": e2e_u5, "a549_e2e": e2e_a5,
                  "u2os_i2e": i2e_u5, "a549_i2e": i2e_a5,
                  "u2os_e2i": e2i_u5, "a549_e2i": e2i_a5}
        mean_i2i = (i2i_u5 + i2i_a5) / 2
        mean_e2e = (e2e_u5 + e2e_a5) / 2
        mean_i2e = (i2e_u5 + i2e_a5) / 2

        if combined  > best["combined"]:  best["combined"]  = combined;  save_ckpt("combined",  epoch, base_m)
        if mean_i2i  > best["i2i"]:       best["i2i"]       = mean_i2i;  save_ckpt("best_i2i",  epoch, base_m)
        if mean_e2e  > best["e2e"]:       best["e2e"]       = mean_e2e;  save_ckpt("best_e2e",  epoch, base_m)
        if mean_i2e  > best["i2e_moa"]:   best["i2e_moa"]   = mean_i2e;  save_ckpt("best_i2e",  epoch, base_m)

        # ── qualified checkpoint：e2e持平 + i2i超baseline，每epoch都保存 ──────
        qual_ok = (e2e_u5 >= THRESH_E2E_U2OS and e2e_a5 >= THRESH_E2E_A549 and
                   i2i_u5 >= THRESH_I2I_U2OS  and i2i_a5 >= THRESH_I2I_A549)
        if qual_ok:
            path = LOG_DIR / f"qualified_ep{epoch:03d}.pt"
            torch.save({"epoch": epoch, **base_m,
                        "img_bb": img_bb.state_dict(), "expr_bb": expr_bb.state_dict(),
                        "head_img_cpd": head_img_cpd.state_dict(), "head_expr_cpd": head_expr_cpd.state_dict(),
                        "head_img_i2i": head_img_i2i.state_dict(), "head_expr_e2e": head_expr_e2e.state_dict(),
                        "quad_temp": quad_temp.state_dict()}, path)
            print(f"  *** qualified ep{epoch}: "
                  f"e2e({e2e_u5*100:.1f}%/{e2e_a5*100:.1f}%) "
                  f"i2i({i2i_u5*100:.1f}%/{i2i_a5*100:.1f}%)")

        if epoch % 20 == 0:
            torch.save({"epoch": epoch,
                        "img_bb": img_bb.state_dict(), "expr_bb": expr_bb.state_dict(),
                        "head_img_cpd": head_img_cpd.state_dict(), "head_expr_cpd": head_expr_cpd.state_dict(),
                        "head_img_i2i": head_img_i2i.state_dict(), "head_expr_e2e": head_expr_e2e.state_dict(),
                        "quad_temp": quad_temp.state_dict(),
                        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
                       LOG_DIR / f"checkpoint_ep{epoch:03d}.pt")

        if epoch % ZEROSHOT_FREQ == 0 or epoch == EPOCHS:
            print(f"  [E{epoch}] HT29 zero-shot eval...")
            multi_h5.load_cell("HT29", H5["HT29"])
            ht29_lbl_map = multi_h5.get_label_to_rows("HT29")
            ht29_rows = {g: ("HT29", ht29_lbl_map[g - OFFSET["HT29"]])
                         for g in ht29_eval_global if (g - OFFSET["HT29"]) in ht29_lbl_map}
            ht29_moa = [g for g in ht29_eval_global if moa_sets[g]]
            ht29_i2e = evaluate_i2e(fn_img_i2i, fn_expr_e2e, ht29_moa, ht29_rows,
                                    u2os_gallery, l1000_combined, multi_h5, N_CROPS_EVAL, moa_sets)
            ht29_i2i = evaluate_i2i(fn_img_i2i, ht29_moa, ht29_eval_global,
                                    ht29_rows, multi_h5, N_CROPS_EVAL, moa_sets)
            print(f"  [E{epoch}] HT29  i2e(→U2OS)={ht29_i2e.get('i2e_m_top5',0)*100:.1f}%  "
                  f"i2i={ht29_i2i.get('i2i_m_top5',0)*100:.1f}%")
            multi_h5.unload_cell("HT29")

    torch.save({"epoch": EPOCHS,
                "img_bb": img_bb.state_dict(), "expr_bb": expr_bb.state_dict(),
                "head_img_cpd": head_img_cpd.state_dict(), "head_expr_cpd": head_expr_cpd.state_dict(),
                "head_img_i2i": head_img_i2i.state_dict(), "head_expr_e2e": head_expr_e2e.state_dict(),
                "quad_temp": quad_temp.state_dict()},
               LOG_DIR / "final_checkpoint.pt")
    print(f"\nDone. combined={best['combined']*100:.1f}% i2i={best['i2i']*100:.1f}% "
          f"e2e={best['e2e']*100:.1f}% i2e={best['i2e_moa']*100:.1f}%")


if __name__ == "__main__":
    main()
