"""Shared validation for the portable crop processing routes."""
import hashlib
import json
from pathlib import Path
import numpy as np

CHANNELS = ['DNA', 'ER', 'RNA', 'AGP', 'Mito']

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def new_output(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path

def validate_crops(handle):
    ds = handle['crops']
    if ds.ndim != 4 or ds.shape[1:] != (5, 128, 128) or ds.dtype != np.uint16 or not len(ds):
        raise ValueError('Expected nonempty uint16 crops [N,5,128,128]')
    if json.loads(handle.attrs.get('channel_order', 'null')) != CHANNELS:
        raise ValueError('Explicit channel_order attribute must be DNA, ER, RNA, AGP, Mito')
    return ds

def dp_normalize(crops):
    x = crops.astype(np.float32)[:, [0, 2, 1, 3, 4]].transpose(0, 2, 3, 1)
    lo = x.min(axis=(1, 2), keepdims=True)
    hi = x.max(axis=(1, 2), keepdims=True)
    return (x - lo) / (hi - lo + 1e-7)

def write_record(output, **record):
    (output / 'run.json').write_text(json.dumps(record, indent=2) + '\n')
