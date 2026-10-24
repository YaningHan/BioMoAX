"""Join explicit record IDs to feature rows and build evaluation inputs."""
import argparse
import csv
from pathlib import Path
import numpy as np

def read_ids(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        if 'record_id' not in (reader.fieldnames or []):
            raise ValueError('ID CSV requires record_id')
        ids = [row['record_id'] for row in reader]
    if not ids or any(not x.strip() for x in ids) or len(set(ids)) != len(ids):
        raise ValueError('Record IDs must be nonempty and unique')
    return ids

def links(path, query_ids, gallery_ids):
    result = np.zeros((len(query_ids),len(gallery_ids)), dtype=bool)
    if path is None:
        return result
    qi = {x:i for i,x in enumerate(query_ids)}
    gi = {x:i for i,x in enumerate(gallery_ids)}
    with open(path) as f:
        reader = csv.DictReader(f)
        if not {'query_id','gallery_id'} <= set(reader.fieldnames or []):
            raise ValueError('Pair CSV requires query_id,gallery_id')
        for row in reader:
            q,g = row['query_id'],row['gallery_id']
            if q not in qi or g not in gi:
                raise ValueError(f'Unknown pair IDs: {q}, {g}')
            result[qi[q],gi[g]] = True
    return result

def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('query','gallery','query-ids','gallery-ids','positive-pairs','output'):
        p.add_argument('--'+name, required=True, type=Path)
    p.add_argument('--excluded-pairs', type=Path)
    a = p.parse_args()
    q,g = np.load(a.query, allow_pickle=False), np.load(a.gallery, allow_pickle=False)
    qi,gi = read_ids(a.query_ids),read_ids(a.gallery_ids)
    if q.ndim != 2 or g.ndim != 2 or q.shape[1] != g.shape[1] or len(q)!=len(qi) or len(g)!=len(gi):
        raise ValueError('Feature dimensions or ID row counts do not match')
    if any(not np.isfinite(x).all() or np.any(np.linalg.norm(x,axis=1)==0) for x in (q,g)):
        raise ValueError('Features must be finite and nonzero')
    positive = links(a.positive_pairs,qi,gi)
    excluded = links(a.excluded_pairs,qi,gi)
    if not np.any(positive & ~excluded):
        raise ValueError('No eligible positive pairs after exclusions')
    with a.output.open('xb') as f:
        np.savez_compressed(f,query=q,gallery=g,positive=positive,excluded=excluded,
                            query_ids=np.array(qi),gallery_ids=np.array(gi))

if __name__ == '__main__':
    main()
