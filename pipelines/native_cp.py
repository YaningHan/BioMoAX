"""Align native CellProfiler profile tables with ordered condition records."""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from common import new_output, sha256, write_record


def align(profiles, records, columns, label_column, mapping=None):
    if not columns or len(set(columns)) != len(columns):
        raise ValueError('Feature columns must be nonempty and unique')
    if not {'record_id', 'label'} <= set(records) or records.empty:
        raise ValueError('Records require record_id,label')
    if records.record_id.isna().any() or records.record_id.astype(str).str.strip().eq('').any() or records.record_id.duplicated().any():
        raise ValueError('Record IDs must be nonempty and unique')
    def integers(values):
        x = pd.to_numeric(values, errors='raise').to_numpy()
        if not np.isfinite(x).all() or not np.equal(x, np.floor(x)).all():
            raise ValueError('Labels must be finite integers')
        return x.astype(np.int64)
    target = integers(records.label)
    source = integers(profiles[label_column])
    if len(set(target)) != len(target) or len(set(source)) != len(source):
        raise ValueError('Duplicate profile or record labels')
    if mapping is not None:
        old, new = integers(mapping.source_label), integers(mapping.label)
        pairs = pd.DataFrame({'old': old, 'new': new}).drop_duplicates()
        if pairs.old.duplicated().any() or pairs.new.duplicated().any():
            raise ValueError('Label mapping must be one-to-one')
        lookup = dict(zip(pairs.old, pairs.new))
        source = [lookup.get(i) for i in source]
    values = profiles[columns].to_numpy(dtype=np.float32)
    result = np.zeros((len(records), len(columns)), dtype=np.float32)
    present = np.zeros(len(records), dtype=bool)
    index = {v: i for i, v in enumerate(target)}
    for label, vector in zip(source, values):
        if label not in index:
            continue
        if not np.isfinite(vector).all():
            raise ValueError('Nonfinite native profile')
        i = index[label]
        result[i] = vector
        present[i] = True
    return result, present


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('profiles', 'records', 'columns', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--label-column', required=True)
    p.add_argument('--mapping', type=Path, help='CSV: source_label,label')
    p.add_argument('--reference', type=Path, help='Optional robust-scaled matrix to compare')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    columns = [c.strip() for c in args.columns.read_text().splitlines() if c.strip()]
    records = pd.read_csv(args.records, dtype={'record_id': str})
    raw, present = align(pd.read_csv(args.profiles), records, columns, args.label_column,
                         pd.read_csv(args.mapping) if args.mapping else None)
    # Native stored matrices use all supplied rows, including zero placeholders.
    # Ridge uses raw.npy and a separate training-only scaler instead.
    scaler = RobustScaler().fit(raw)
    scaled = scaler.transform(raw).astype(np.float32)
    error = None
    if args.reference:
        ref = np.load(args.reference, allow_pickle=False)
        if ref.shape != scaled.shape or not np.isfinite(ref).all():
            raise ValueError('Reference shape or values are invalid')
        error = float(np.max(np.abs(ref-scaled)))
        if not np.allclose(ref, scaled, rtol=3e-3, atol=3e-3):
            raise ValueError(f'Reference mismatch: max absolute error {error}')
    out = new_output(args.output)
    np.save(out/'raw.npy', raw)
    np.save(out/'features.npy', scaled)
    np.save(out/'valid_rows.npy', present & (np.linalg.norm(raw, axis=1) > 0))
    records[['record_id']].to_csv(out/'feature_ids.csv', index=False)
    records.assign(profile_present=present).to_csv(out/'coverage.csv', index=False)
    joblib.dump(scaler, out/'scaler.joblib')
    write_record(out, status='complete', n_records=len(records), n_missing=int((~present).sum()),
                 scaling='RobustScaler fitted to all supplied records, including zero placeholders',
                 reference_max_abs=error,
                 inputs={key: sha256(getattr(args,key)) for key in
                         ('profiles','records','columns','mapping') if getattr(args,key)})


if __name__ == '__main__':
    main()
