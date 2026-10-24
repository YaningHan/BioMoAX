"""Extract same-cell compound means from a Level-3 L1000 GCTX."""
import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from common import new_output, sha256, write_record


def normalize_broad(value):
    parts = str(value).strip().split('-')
    if len(parts) < 2 or parts[0] != 'BRD' or not parts[1]:
        raise ValueError(f'Expected Broad compound ID, got {value!r}')
    return '-'.join(parts[:2])


def strings(dataset):
    return [v.decode() if isinstance(v, bytes) else str(v) for v in dataset[:]]


def extract(records, instinfo, gctx, cell, output):
    records, instinfo, gctx = map(Path, (records, instinfo, gctx))
    rec = pd.read_csv(records, dtype=str, keep_default_na=False)
    if not {'record_id', 'broad_id'} <= set(rec.columns) or rec.empty:
        raise ValueError('Records must contain nonempty record_id,broad_id columns')
    if rec.record_id.str.strip().eq('').any() or rec.record_id.duplicated().any():
        raise ValueError('record_id must be nonempty and unique')
    if 'cell' in rec and not rec.cell.eq(cell).all():
        raise ValueError('Every record must belong to the requested cell')
    keys = rec.broad_id.map(normalize_broad)
    columns = ['sample_id', 'pert_id', 'pert_type', 'cell_iname']
    pieces = []
    for chunk in pd.read_csv(instinfo, sep='\t', usecols=columns, dtype=str,
                             keep_default_na=False, chunksize=100000):
        chunk = chunk[(chunk.pert_type == 'trt_cp') & (chunk.cell_iname == cell)].copy()
        chunk['broad_id'] = chunk.pert_id.str.split('-').str[:2].str.join('-')
        pieces.append(chunk[chunk.broad_id.isin(set(keys))])
    instances = pd.concat(pieces, ignore_index=True)
    if instances.sample_id.duplicated().any():
        raise ValueError('Duplicate sample IDs in selected instance metadata')
    out = new_output(output)
    with h5py.File(gctx, 'r') as source:
        sample_ids = strings(source['0/META/COL/id'])
        gene_ids = strings(source['0/META/ROW/id'])
        matrix = source['0/DATA/0/matrix']
        if matrix.shape != (len(sample_ids), len(gene_ids)):
            raise ValueError('Expected GCTX matrix ordered as samples × genes')
        if len(set(sample_ids)) != len(sample_ids) or len(set(gene_ids)) != len(gene_ids):
            raise ValueError('GCTX identifiers must be unique')
        lookup = {s: i for i, s in enumerate(sample_ids)}
        features = np.full((len(rec), len(gene_ids)), np.nan, dtype=np.float32)
        counts, evidence, absent = [], [], []
        for row, (record_id, key) in enumerate(zip(rec.record_id, keys)):
            samples = instances.loc[instances.broad_id == key, 'sample_id'].tolist()
            missing = [s for s in samples if s not in lookup]
            absent.extend({'record_id': record_id, 'sample_id': s} for s in missing)
            indices = sorted(lookup[s] for s in samples if s in lookup)
            counts.append(len(indices))
            if indices:
                values = np.concatenate([matrix[indices[start:start+256], :]
                                         for start in range(0, len(indices), 256)]).astype(np.float32)
                if not np.isfinite(values).all():
                    raise ValueError(f'Nonfinite expression for {record_id}')
                features[row] = values.mean(axis=0)
                evidence.extend({'record_id': record_id, 'sample_id': sample_ids[i],
                                 'matrix_row': i} for i in indices)
    rec['n_samples'] = counts
    rec['available'] = np.array(counts) > 0
    rec.to_csv(out / 'records.csv', index=False)
    np.save(out / 'expression.npy', features)
    np.save(out / 'valid_rows.npy', np.array(counts) > 0)
    pd.DataFrame({'gene_id': gene_ids}).to_csv(out / 'genes.csv', index=False)
    pd.DataFrame(evidence, columns=['record_id', 'sample_id', 'matrix_row']).to_csv(out / 'samples.csv', index=False)
    pd.DataFrame(absent, columns=['record_id', 'sample_id']).to_csv(out / 'unmatched_samples.csv', index=False)
    write_record(out, status='complete', cell=cell, aggregation='same-cell trt_cp float32 instance mean; no QC/dose/time filter',
                 n_records=len(rec), n_available=int(sum(np.array(counts) > 0)),
                 inputs={str(p.resolve()): {'bytes': p.stat().st_size} for p in (records, instinfo, gctx)},
                 records_sha256=sha256(records), instinfo_sha256=sha256(instinfo))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('records', 'instinfo', 'gctx', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--cell', required=True)
    args = parser.parse_args()
    extract(**vars(args))


if __name__ == '__main__':
    main()
