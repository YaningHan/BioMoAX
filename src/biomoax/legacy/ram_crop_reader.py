"""Use the legacy sampling rules with uint16 NumPy caches, not mmap files."""
import gc
import inspect
import json
from pathlib import Path
import numpy as np


def copy_selected(ds, idx, destination, channels=5, progress=None):
    idx = np.asarray(idx, dtype=np.int64)
    assert len(idx) == len(destination)
    assert not len(idx) or (idx[0] >= 0 and idx[-1] < len(ds))
    assert np.all(np.diff(idx) > 0)
    if not len(idx):
        return
    # Read each touched source block once instead of many tiny fancy-index reads.
    groups = idx // 256
    boundaries = np.r_[0, np.flatnonzero(np.diff(groups)) + 1, len(idx)]
    for k in range(len(boundaries)-1):
        a, b = boundaries[k:k+2]
        start, end = int(idx[a]), int(idx[b-1]) + 1
        block = ds[start:end, :channels]
        destination[a:b] = block[idx[a:b] - start]
        if progress and (k % 32 == 0 or b == len(idx)):
            progress(int(b), len(idx))


def make_reader(v8, base, log):
    log = Path(log)
    class RAMReader(v8.DiskBackedMultiH5Reader):
        def __init__(self):
            self._caches = {}; self._labels = {}; self._source_indices = {}; self._cache_paths = {}
            self._allocated = 0
            print('Resident CPU RAM crop cache (uint16); no mmap allocation', flush=True)

        def _allocate(self, cell, shape, dtype):
            assert np.dtype(dtype) == np.uint16
            size = int(np.prod(shape)) * np.dtype(dtype).itemsize
            self._allocated += size
            # ~56.5GiB drug crops expected; reserve RAM for CRISPR and training.
            assert self._allocated <= 64 * 2**30, 'Drug RAM cache exceeds 64GiB budget'
            self._active_cell = cell
            print(f'RAM allocate {cell}: {size/2**30:.2f} GiB; cumulative {self._allocated/2**30:.2f} GiB', flush=True)
            return np.empty(shape, dtype=dtype)

        def _copy_selected(self, ds, idx, raw):
            def report(done, total):
                p = log / 'ram_loading.json'
                tmp = p.with_suffix('.tmp')
                tmp.write_text(json.dumps({'cell':self._active_cell, 'copied_crops':done, 'selected_crops':total,
                                           'allocated_drug_cache_bytes':self._allocated, 'storage':'resident_uint16_ndarray'}))
                tmp.replace(p)
            copy_selected(ds, idx, raw, base.IMG_CHANNELS, report)

        @staticmethod
        def _finish(raw, source_h5):
            assert type(raw) is np.ndarray and raw.dtype == np.uint16 and raw.flags.c_contiguous
            v8.evict_file_cache(source_h5)

        def unload_cell(self, cell):
            array = self._caches.pop(cell, None)
            if array is not None: self._allocated -= array.nbytes
            self._labels.pop(cell, None); self._source_indices.pop(cell, None)
            del array; gc.collect()

    # Keep exact existing sampling/label remapping, with truthful storage messages.
    for name in ['load_cell', 'load_cpg0030', 'load_with_global_offset']:
        import textwrap
        original = getattr(v8.DiskBackedMultiH5Reader, name)
        source = textwrap.dedent(inspect.getsource(original)).replace('disk-backed', 'resident RAM').replace(' GB memmap', ' GB RAM')
        scope = dict(original.__globals__)
        exec(compile(source, '<ram-sampling-compatible>', 'exec'), scope)
        setattr(RAMReader, name, scope[name])
    return RAMReader
