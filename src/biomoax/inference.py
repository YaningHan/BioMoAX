"""Load historical state dictionaries and encode one condition at a time."""
import numpy as np
import torch
from torch.nn import functional as F
from .model import ImageBackbone, ExprBackbone, ProjHead, gpu_normalize


def load_models(path, device="cpu", head="moa"):
    if head not in {"moa", "compound"}:
        raise ValueError("head must be moa or compound")
    # Historical checkpoints include RNG tuples. Load only checkpoints from trusted sources.
    state = torch.load(path, map_location="cpu", weights_only=False)
    in_dim = state["expr_bb"]["net.0.weight"].shape[1]
    image, expression = ImageBackbone(), ExprBackbone(in_dim)
    ih, eh = ProjHead(384), ProjHead(512)
    for module, key in [(image,"img_bb"),(expression,"expr_bb"),
                        (ih,"head_img_i2i" if head=="moa" else "head_img_cpd"),
                        (eh,"head_expr_e2e" if head=="moa" else "head_expr_cpd")]:
        module.load_state_dict(state[key], strict=True)
        module.to(device).eval()
    return image, expression, ih, eh


@torch.no_grad()
def encode_condition(models, values, modality, device="cpu", batch_size=16, per_crop=False):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if per_crop and modality != "image":
        raise ValueError("per_crop is only supported for image inputs")
    x = np.asarray(values)
    if len(x) == 0 or not np.isfinite(x).all():
        raise ValueError("Empty or nonfinite input")
    image, expression, ih, eh = models
    parts = []
    for start in range(0, len(x), batch_size):
        v = torch.as_tensor(x[start:start+batch_size].astype(np.float32), device=device)
        if modality == "image":
            if v.ndim != 4 or tuple(v.shape[1:]) != (5,128,128):
                raise ValueError("Images must have shape [crops,5,128,128]")
            # Scale matches the historical uint16 reader before percentile normalization.
            parts.append(ih(image(gpu_normalize(v / 65535.0))))
        elif modality == "expression":
            if v.ndim != 2:
                raise ValueError("Expressions must have shape [records,features]")
            parts.append(eh(expression(v)))
        else:
            raise ValueError("modality must be image or expression")
    z = torch.cat(parts)
    if modality == "image" and not per_crop:
        z = F.normalize(z.mean(0, keepdim=True), dim=-1)
    return z.cpu().numpy()
