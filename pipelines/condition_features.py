"""Build ID-indexed condition features from one crop collection."""
import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from common import new_output, validate_crops, write_record, sha256


def groups(handle, records):
    frame = pd.read_csv(records, dtype={'record_id': str}, keep_default_na=False)
    if not {'record_id', 'label'} <= set(frame) or frame.empty:
        raise ValueError('Provide a nonempty CSV with record_id,label columns')
    if frame.record_id.str.strip().eq('').any() or frame.record_id.duplicated().any():
        raise ValueError('record_id must be nonempty and unique')
    numeric = pd.to_numeric(frame.label, errors='raise').to_numpy()
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError('Labels must be finite integers')
    frame['label'] = numeric.astype(np.int64)
    if frame.label.duplicated().any():
        raise ValueError('Each crop label must map to exactly one record_id')
    labels = handle['labels'][:]
    if labels.shape != (len(handle['crops']),) or labels.dtype.kind not in 'iu':
        raise ValueError('Expected one integer label per crop')
    return frame, labels


def aggregate(values, mode):
    values = np.asarray(values, dtype=np.float64)
    if mode == 'normalized-mean':
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError('Cannot normalize a zero crop vector')
        values = values / norms
    result = values.mean(axis=0)
    norm = np.linalg.norm(result)
    if not np.isfinite(result).all() or norm == 0:
        raise ValueError('Condition mean is nonfinite or zero')
    if mode == 'normalized-mean':
        result /= norm
    return result.astype(np.float32)


def build(args, encoder=None):
    if args.max_crops < 0:
        raise ValueError('max-crops must be zero (all) or positive')
    # Fail before loading checkpoints if an output directory already exists.
    if args.output.exists():
        raise FileExistsError(args.output)
    with h5py.File(args.crops, 'r') as handle:
        frame, labels = groups(handle, args.records)
        if args.mode == 'image':
            validate_crops(handle)
            if args.batch_size < 1:
                raise ValueError('batch-size must be positive')
            if encoder is None:
                from biomoax.inference import load_models, encode_condition
                models = load_models(args.checkpoint, device=args.device, head=args.head)
                encoder = lambda x: encode_condition(models, x, 'image', args.device, args.batch_size)[0]
            source_rows = np.arange(len(labels))
            valid = np.ones(len(labels), dtype=bool)
            values = None
        else:
            values = np.load(args.features, mmap_mode='r', allow_pickle=False)
            source_rows = np.load(args.source_rows, allow_pickle=False)
            if values.ndim != 2 or values.shape[1] == 0:
                raise ValueError('Expected a crop × feature matrix')
            if source_rows.shape != (len(values),) or source_rows.dtype.kind not in 'iu':
                raise ValueError('source-rows must be an integer vector aligned with features')
            if len(np.unique(source_rows)) != len(source_rows) or np.any(source_rows < 0) or np.any(source_rows >= len(labels)):
                raise ValueError('Duplicate or out-of-range crop source rows')
            valid = np.isfinite(values).all(axis=1)
            if args.valid_rows:
                mask = np.load(args.valid_rows, allow_pickle=False)
                if mask.dtype != np.bool_ or mask.shape != (len(values),):
                    raise ValueError('valid-rows must be a boolean vector aligned with features')
                valid &= mask
            if args.aggregation == 'normalized-mean':
                valid &= np.linalg.norm(values, axis=1) > 0
        vectors, coverage, selected = [], [], []
        for row in frame.itertuples():
            candidates = np.flatnonzero(labels[source_rows] == row.label)
            usable = candidates[valid[candidates]]
            usable = usable[np.argsort(source_rows[usable])]
            if args.max_crops:
                usable = usable[:args.max_crops]
            if not len(usable):
                raise ValueError(f'No usable crops for {row.record_id}; resolve coverage explicitly')
            actual_rows = source_rows[usable]
            if args.mode == 'image':
                # Model batches are bounded; selected raw crops are loaded per condition.
                # Invoke once per condition to avoid averaging normalized batch means.
                vector = encoder(handle['crops'][actual_rows.tolist()])
            else:
                vector = aggregate(values[usable], args.aggregation)
            if not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
                raise ValueError(f'Invalid condition embedding: {row.record_id}')
            vectors.append(vector)
            coverage.append({'record_id': row.record_id, 'label': row.label,
                             'n_crops_total': int(np.sum(labels == row.label)),
                             'n_feature_rows': len(candidates),
                             'n_valid': int(valid[candidates].sum()), 'n_used': len(usable)})
            selected.extend({'record_id': row.record_id, 'source_row': int(i)} for i in actual_rows)
    out = new_output(args.output)
    np.save(out/'features.npy', np.stack(vectors).astype(np.float32))
    frame[['record_id']].to_csv(out/'feature_ids.csv', index=False)
    pd.DataFrame(coverage).to_csv(out/'coverage.csv', index=False)
    pd.DataFrame(selected).to_csv(out/'selected_crops.csv', index=False)
    record = dict(status='complete', mode=args.mode, records_sha256=sha256(args.records),
                  crop_selection='source row ascending; first max_crops, or all if zero',
                  max_crops=args.max_crops, n_records=len(frame))
    if args.mode == 'image':
        record.update(head=args.head, checkpoint_sha256=sha256(args.checkpoint),
                      aggregation='normalized mean of crop embeddings')
    else:
        record.update(aggregation=args.aggregation, features_sha256=sha256(args.features),
                      source_rows_sha256=sha256(args.source_rows),
                      valid_rows_sha256=sha256(args.valid_rows) if args.valid_rows else None)
    write_record(out, **record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='mode', required=True)
    for mode in ('image', 'baseline'):
        p = commands.add_parser(mode)
        for name in ('crops', 'records', 'output'):
            p.add_argument('--'+name, type=Path, required=True)
        p.add_argument('--max-crops', type=int, default=100 if mode == 'image' else 0)
        if mode == 'image':
            p.add_argument('--checkpoint', type=Path, required=True)
            p.add_argument('--head', choices=['moa', 'compound'], default='moa')
            p.add_argument('--device', default='cpu')
            p.add_argument('--batch-size', type=int, default=16)
        else:
            for name in ('features', 'source-rows'):
                p.add_argument('--'+name, type=Path, required=True)
            p.add_argument('--valid-rows', type=Path)
            p.add_argument('--aggregation', choices=['mean', 'normalized-mean'], default='mean')
    build(parser.parse_args())


if __name__ == '__main__':
    main()
