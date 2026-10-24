"""Select task embeddings by record IDs, never by assumed row positions."""
import argparse
from pathlib import Path
import numpy as np
from build_retrieval import read_ids

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('features','feature-ids','task-ids','output'):
        p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args()
    features=np.load(a.features,allow_pickle=False)
    all_ids=read_ids(a.feature_ids);selected=read_ids(a.task_ids)
    if features.ndim!=2 or len(features)!=len(all_ids):raise ValueError('Feature/ID count mismatch')
    index={v:i for i,v in enumerate(all_ids)}
    missing=set(selected)-set(index)
    if missing:raise ValueError('Missing task feature IDs: '+str(sorted(missing)[:5]))
    result=features[[index[x] for x in selected]]
    if not np.isfinite(result).all() or np.any(np.linalg.norm(result,axis=1)==0):
        raise ValueError('Nonfinite or zero task features; resolve coverage explicitly')
    with a.output.open('xb') as f:np.save(f,result)

if __name__=='__main__':main()
