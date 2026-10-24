"""Crop-level Nuclei median features, not native whole-site profiles."""
import argparse
from pathlib import Path
import subprocess
import numpy as np
from common import CHANNELS, new_output, sha256, validate_crops, write_record

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--executable', default='cellprofiler')
    p.add_argument('--pipeline', type=Path, default=Path(__file__).parent/'assets/crop.cppipe')
    p.add_argument('--columns', type=Path, default=Path(__file__).parent/'assets/cp_columns.txt')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--timeout', type=int, default=900)
    a = p.parse_args()
    if min(a.batch_size, a.timeout) < 1:
        p.error('batch-size and timeout must be positive')
    import h5py
    import pandas as pd
    import tifffile
    columns = [s.strip() for s in a.columns.read_text().splitlines() if s.strip()]
    if not columns or len(set(columns)) != len(columns):
        raise ValueError('Feature names must be nonempty and unique')
    if not a.pipeline.is_file():
        raise FileNotFoundError(a.pipeline)
    with h5py.File(a.input) as f:
        ds = validate_crops(f)
        output = new_output(a.output)
        z = np.lib.format.open_memmap(output/'features.npy', mode='w+',
                                     dtype='float32', shape=(len(ds), len(columns)))
        z[:] = np.nan
        counts = np.zeros(len(ds), dtype=np.int32)
        for start in range(0, len(ds), a.batch_size):
            batch = output / f'batch_{start:08d}'
            images = batch/'images'; images.mkdir(parents=True)
            result = batch/'output'; result.mkdir()
            records = []
            crops = ds[start:start+a.batch_size]
            for i, crop in enumerate(crops):
                row = start+i
                rec = dict(Metadata_Plate='input', Metadata_Well=f'R{row:07d}',
                           Metadata_Site=1, Metadata_H5Row=row)
                for c, name in enumerate(CHANNELS):
                    fn = f'r{row:07d}_{name}.tif'
                    tifffile.imwrite(images/fn, crop[c], photometric='minisblack')
                    rec['FileName_'+name] = fn
                    rec['PathName_'+name] = str(images)
                for name in ('Brightfield', 'BFHigh', 'BFLow'):
                    rec['FileName_'+name] = rec['FileName_AGP']
                    rec['PathName_'+name] = str(images)
                records.append(rec)
            csv = batch/'load_data.csv'
            pd.DataFrame(records).to_csv(csv, index=False)
            with (batch/'cellprofiler.log').open('w') as log:
                subprocess.run([a.executable, '-c', '-r', '-p', str(a.pipeline.resolve()),
                                '--data-file', str(csv), '-o', str(result)],
                               stdout=log, stderr=subprocess.STDOUT, check=True, timeout=a.timeout)
            table = pd.read_csv(result/'Nuclei.csv', low_memory=False)
            missing = set(columns) - set(table.columns)
            if missing:
                raise ValueError('Missing Nuclei features: ' + str(sorted(missing)[:5]))
            for image, group in table.groupby('ImageNumber', sort=False):
                pos = int(image)-1
                if image != int(image) or not 0 <= pos < len(crops):
                    raise ValueError('Unexpected ImageNumber')
                z[start+pos] = group[columns].replace([np.inf, -np.inf], np.nan).median().to_numpy()
                counts[start+pos] = len(group)
            z.flush()
        np.save(output/'source_rows.npy', np.arange(len(ds)))
        np.save(output/'nuclei_counts.npy', counts)
        np.save(output/'valid_rows.npy', (counts > 0) & np.isfinite(z).all(axis=1))
        (output/'feature_columns.txt').write_text('\n'.join(columns)+'\n')
        write_record(output, status='complete', route='crop_nuclei_median',
                     input_sha256=sha256(a.input), pipeline_sha256=sha256(a.pipeline),
                     columns_sha256=sha256(a.columns), n_crops=len(ds))

if __name__ == '__main__':
    main()
