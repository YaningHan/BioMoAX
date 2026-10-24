"""Definitions extracted unchanged from the paper training archive."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt
IMG_CHANNELS = 5

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

def gpu_normalize(imgs):
    B, C, H, W = imgs.shape; flat = imgs.reshape(B * C, H * W); n = H * W
    k_lo = max(1, int(0.01 * n)); k_hi = min(n, int(0.99 * n))
    lo = torch.kthvalue(flat, k_lo, dim=1).values.view(B, C, 1, 1)
    hi = torch.kthvalue(flat, k_hi, dim=1).values.view(B, C, 1, 1)
    return torch.clamp((imgs - lo) / (hi - lo + 1e-8), 0, 1)
