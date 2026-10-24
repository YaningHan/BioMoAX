"""Cellpose explicit-manifest reconstruction of the chemical/genetic crop route."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from common import CHANNELS, new_output, sha256, write_record

def correct(image, illum, kind):
    if image.ndim != 2 or image.shape != illum.shape:
        raise ValueError('Image and illumination must have the same 2D shape')
    if not np.isfinite(image).all() or not np.isfinite(illum).all() or np.any(illum <= 0):
        raise ValueError('Nonfinite input or nonpositive illumination')
    image = image.astype(np.float32)
    if kind == 'crispr':
        return image / np.maximum(illum, 1e-6)
    if kind == 'compound':
        return image / (illum + 1e-6) * np.median(illum)
    raise ValueError('kind must be crispr or compound')

def crop_mask(images, masks):
    from scipy import ndimage
    ids = np.unique(masks); ids = ids[ids > 0]
    centers = ndimage.center_of_mass(np.ones_like(masks), masks, ids)
    crops, origins = [], []
    for label, (y, x) in zip(ids, centers):
        y, x = int(y), int(x)
        if y < 64 or x < 64 or y+64 > images.shape[1] or x+64 > images.shape[2]:
            continue
        crops.append(np.clip(images[:, y-64:y+64, x-64:x+64], 0, 65535).astype(np.uint16))
        origins.append(dict(mask_id=int(label), center_y=y, center_x=x))
    return (np.stack(crops) if crops else np.empty((0, 5, 128, 128), np.uint16)), origins

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--weights', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--gpu', action='store_true')
    a = p.parse_args()
    if not a.weights.is_file():
        raise FileNotFoundError(a.weights)
    import h5py
    import tifffile
    from PIL import Image
    from cellpose import models
    from importlib.metadata import version
    if version('cellpose') != '4.0.9':
        raise RuntimeError('This reconstruction requires cellpose==4.0.9')
    with a.manifest.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError('Empty manifest')
    keys = [(r['plate'], r['well'], r['site']) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate plate/well/site')
    model = models.CellposeModel(gpu=a.gpu, pretrained_model=str(a.weights.resolve()))
    if Path(model.pretrained_model).resolve() != a.weights.resolve():
        raise ValueError('Unexpected fallback segmentation model')
    output = new_output(a.output)
    records = []
    with h5py.File(output/'crops.h5', 'w') as h:
        ds = h.create_dataset('crops', shape=(0,5,128,128), maxshape=(None,5,128,128),
                              dtype='uint16', chunks=(16,5,128,128))
        labels = h.create_dataset('labels', shape=(0,), maxshape=(None,), dtype='int32')
        h.attrs['channel_order'] = json.dumps(CHANNELS)
        for row in rows:
            images, sources = [], []
            for channel in CHANNELS:
                def resolve(key):
                    path = Path(row[key])
                    return path if path.is_absolute() else a.manifest.resolve().parent/path
                path, ip = resolve(channel), resolve('illum_'+channel)
                if path.suffix.lower() in ('.tif', '.tiff'):
                    raw = tifffile.imread(path)
                else:
                    with Image.open(path) as image:
                        raw = np.array(image)
                images.append(correct(raw, np.load(ip, allow_pickle=False), row['kind']))
                sources.append(dict(channel=channel, image_sha256=sha256(path), illum_sha256=sha256(ip)))
            images = np.stack(images)
            lo, hi = np.percentile(images[0], [1,99])
            dna = np.clip((images[0]-lo)/max(float(hi-lo), 1e-6), 0, 1)
            result = model.eval(dna, diameter=40 if row['kind']=='crispr' else None, batch_size=8)
            crops, origins = crop_mask(images, result[0])
            start, end = len(ds), len(ds)+len(crops)
            ds.resize(end, axis=0); labels.resize(end, axis=0)
            ds[start:end] = crops; labels[start:end] = int(row['label'])
            records.append(dict(plate=row['plate'], well=row['well'], site=row['site'],
                                label=int(row['label']), kind=row['kind'], start=start, stop=end,
                                origins=origins, sources=sources))
    write_record(output, status='complete', route='chemical_genetic_reconstruction',
                 manifest_sha256=sha256(a.manifest), weights_sha256=sha256(a.weights),
                 cellpose=version('cellpose'), records=records)

if __name__ == '__main__':
    main()
