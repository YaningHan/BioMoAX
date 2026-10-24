"""Audited CellPainting-CNN crop features; no random-weight fallback."""
import argparse
from pathlib import Path
import numpy as np
from common import dp_normalize, new_output, sha256, validate_crops, write_record

def load_model(weights):
    import h5py
    import efficientnet
    import efficientnet.tfkeras as efficientnet_keras
    if efficientnet.__version__ != '1.1.1':
        raise RuntimeError('This route requires efficientnet==1.1.1')
    if not weights.is_file():
        raise FileNotFoundError(weights)
    model = efficientnet_keras.EfficientNetB0(
        include_top=False, weights=None, input_shape=(128, 128, 5), pooling='avg')
    loaded = 0
    with h5py.File(weights) as f:
        for layer in model.layers:
            if not layer.weights:
                continue
            values = {}
            def visit(key, value):
                if isinstance(value, h5py.Dataset):
                    name = key.split('/')[-1]
                    if name in values:
                        raise ValueError('Ambiguous weight name: ' + name)
                    values[name] = value[()]
            f[layer.name].visititems(visit)
            arrays = [values[w.name.split('/')[-1]] for w in layer.weights]
            for array, weight in zip(arrays, layer.weights):
                if array.shape != tuple(weight.shape) or not np.isfinite(array).all():
                    raise ValueError('Invalid tensor in ' + layer.name)
            layer.set_weights(arrays)
            for expected, actual in zip(arrays, layer.get_weights()):
                np.testing.assert_array_equal(expected, actual)
            loaded += len(arrays)
    if loaded != 309:
        raise ValueError(f'Expected 309 verified tensors; got {loaded}')
    return model

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--weights', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--batch-size', type=int, default=32)
    a = p.parse_args()
    if a.batch_size < 1:
        p.error('batch-size must be positive')
    import h5py
    with h5py.File(a.input) as f:
        crops = validate_crops(f)
        model = load_model(a.weights)
        output = new_output(a.output)
        z = np.lib.format.open_memmap(output / 'features.npy', mode='w+',
                                     dtype='float32', shape=(len(crops), 1280))
        for start in range(0, len(crops), a.batch_size):
            y = model(dp_normalize(crops[start:start+a.batch_size]), training=False).numpy()
            if not np.isfinite(y).all() or np.any(np.linalg.norm(y, axis=1) == 0):
                raise ValueError('Invalid feature vectors')
            z[start:start+len(y)] = y
        z.flush()
        np.save(output / 'source_rows.npy', np.arange(len(crops)))
        write_record(output, status='complete', route='audited_crop_minmax',
                     input_sha256=sha256(a.input), weights_sha256=sha256(a.weights),
                     verified_tensors=309, n_crops=len(crops),
                     feature_channels=['DNA', 'RNA', 'ER', 'AGP', 'Mito'])

if __name__ == '__main__':
    main()
