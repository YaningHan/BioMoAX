"""Repeated disjoint crop splits for same-drug identity retrieval."""
import argparse
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from common import new_output, sha256, write_record


def evaluate(features, rows, method, seeds):
    if features.ndim != 2 or not len(features) or not features.shape[1] or len(rows) != len(features):
        raise ValueError('Expected crop features and one metadata row per feature row')
    if not {'record_id','label','source_row'} <= set(rows):
        raise ValueError('Metadata requires record_id,label,source_row')
    if rows.record_id.isna().any() or rows.record_id.astype(str).str.strip().eq('').any():
        raise ValueError('Record IDs must be nonempty')
    for column in ('label','source_row'):
        x=pd.to_numeric(rows[column],errors='raise').to_numpy()
        if not np.isfinite(x).all() or (x<0).any() or not np.equal(x,np.floor(x)).all():
            raise ValueError('Labels and source rows must be nonnegative integers')
        rows=rows.copy(); rows[column]=x.astype(np.int64)
    if rows.source_row.duplicated().any():
        raise ValueError('Duplicate crop source rows')
    records=rows[['record_id','label']].drop_duplicates().sort_values('label')
    if records.record_id.duplicated().any() or records.label.duplicated().any():
        raise ValueError('Record IDs and labels must map one-to-one')
    if len(records)<2:
        raise ValueError('Identity retrieval requires at least two drugs')
    if not seeds or len(set(seeds))!=len(seeds) or min(seeds)<0:
        raise ValueError('Split seeds must be unique nonnegative integers')
    groups=[]
    for label in records.label:
        pos=np.flatnonzero(rows.label.to_numpy()==label)
        pos=pos[np.argsort(rows.source_row.to_numpy()[pos])]
        if len(pos)<2:
            raise ValueError('Each drug requires at least two crops')
        groups.append(pos)
    metrics=[]; ranks=[]; splits=[]
    for seed in seeds:
        profiles=[]
        for record,pos in zip(records.itertuples(),groups):
            rng=np.random.default_rng(np.random.SeedSequence([seed,int(record.label)]))
            shuffled=rng.permutation(pos)
            nq=max(1,min(len(pos)-1,int(np.floor(.2*len(pos)))))
            query=set(shuffled[:nq]); pair=[]
            for side in ('query','gallery'):
                chosen=np.array([i for i in pos if (i in query)==(side=='query')])
                z=np.asarray(features[chosen],dtype=np.float64).copy()
                if method=='cellprofiler':
                    z[~np.isfinite(z)]=np.nan
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore',RuntimeWarning)
                        v=np.nanmedian(z,axis=0)
                else:
                    if not np.isfinite(z).all(): raise ValueError('Mean inputs must be finite')
                    v=z.mean(axis=0)
                pair.append(v)
                splits.extend(dict(seed=seed,record_id=record.record_id,source_row=int(rows.iloc[i].source_row),side=side) for i in chosen)
            profiles.append(pair)
        profiles=np.asarray(profiles)
        if method=='cellprofiler':
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                med=np.nan_to_num(np.nanmedian(profiles[:,1],axis=0),nan=0.,posinf=0.,neginf=0.)
            bad=~np.isfinite(profiles)
            profiles[bad]=np.broadcast_to(med,profiles.shape)[bad]
        profiles/=np.maximum(np.linalg.norm(profiles,axis=-1,keepdims=True),1e-12)
        similarity=profiles[:,0]@profiles[:,1].T
        order=np.argsort(-similarity,axis=1,kind='stable')
        rank=(order==np.arange(len(records))[:,None]).argmax(axis=1)+1
        ranks.extend(dict(seed=seed,record_id=rid,rank=int(v)) for rid,v in zip(records.record_id,rank))
        metrics.extend(dict(seed=seed,k=k,n=len(records),hit_rate=float(100*np.mean(rank<=k))) for k in range(1,6))
    return pd.DataFrame(metrics),pd.DataFrame(ranks),pd.DataFrame(splits)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('features','crop-ids','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--method',choices=['cellprofiler','mean'],required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[42,142,242,342,442])
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    metrics,ranks,splits=evaluate(np.load(a.features,allow_pickle=False),
        pd.read_csv(a.crop_ids,dtype={'record_id':str}),a.method,a.seeds)
    out=new_output(a.output)
    metrics.to_csv(out/'metrics.csv',index=False); ranks.to_csv(out/'ranks.csv',index=False)
    splits.to_csv(out/'splits.csv',index=False)
    metrics.groupby('k').hit_rate.agg(['mean','std']).to_csv(out/'summary.csv')
    write_record(out,status='complete',method=a.method,seeds=a.seeds,
        features_sha256=sha256(a.features),crop_ids_sha256=sha256(a.crop_ids))


if __name__=='__main__': main()
