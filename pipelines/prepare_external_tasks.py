"""Export cross-batch and A549 CRISPR task definitions from recorded inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import pandas as pd
from prepare_moa_tasks import moa_set

def save_task(out,query,gallery,positive,extra=None):
    out.mkdir(parents=True)
    query.to_csv(out/'query_ids.csv',index=False)
    gallery.to_csv(out/'gallery_ids.csv',index=False)
    pd.DataFrame(positive,columns=['query_id','gallery_id']).to_csv(out/'positive_pairs.csv',index=False)
    pd.DataFrame(columns=['query_id','gallery_id']).to_csv(out/'excluded_pairs.csv',index=False)
    if extra is not None:pd.DataFrame(extra).to_csv(out/'positive_evidence.csv',index=False)
    return dict(queries=len(query),gallery=len(gallery),positive_pairs=len(positive))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',required=True,type=Path)
    p.add_argument('--records',required=True,type=Path)
    p.add_argument('--crispr-gallery-keys',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    proc=a.project_root/'data/processed'
    records=pd.read_csv(a.records,keep_default_na=False)
    meta_path=proc/'meta_cpg0030_u2os.csv'
    train_path=proc/'metadata/cpg0030_train_split.csv'
    held_path=proc/'metadata/cpg0030_eval_split.csv'
    meta=pd.read_csv(meta_path)
    with h5py.File(proc/'crops_cpg0030_u2os.h5') as f:
        if len(f['crops'])!=len(meta):raise ValueError('Cross-batch crop/metadata row mismatch')
    train=set(pd.read_csv(train_path).cpd_name.astype(str))
    held=set(pd.read_csv(held_path).cpd_name.astype(str))
    gallery=records[records.cell.eq('U2OS')].copy()
    gmoa={r.record_id:moa_set(r.moa) for r in gallery.itertuples()}
    tasks={}; crop_rows=[]
    for name,match,expected in [('cross_batch_moa','moa',115),('cross_batch_shared','name',32)]:
        query=[]; positive=[]
        for compound,group in meta[meta.match_type.eq(match)].groupby('cpd_name',sort=True):
            if match=='name' and compound not in held:continue
            if match=='moa' and compound in train:continue
            annotations=moa_set(group.iloc[0].moa)
            hits=[gid for gid,m in gmoa.items() if m & annotations]
            if not hits:continue
            # IDs follow the frozen alphabetical query ordering, not a model ranking.
            qid=f'cpg0030:U2OS:{match}:{len(query)}'
            query.append(dict(record_id=qid,name=compound,moa='|'.join(sorted(annotations)),
                              match_type=match,n_crops=len(group)))
            positive.extend((qid,gid) for gid in hits)
            crop_rows.extend(dict(record_id=qid,h5_row=int(i)) for i in group.index)
        if len(query)!=expected:raise ValueError(f'{name}: unexpected query count {len(query)}')
        tasks[name]=save_task(a.output/name,pd.DataFrame(query),gallery,positive)
    pd.DataFrame(crop_rows).to_csv(a.output/'cross_batch_crop_rows.csv',index=False)

    # Freeze baseline coverage separately from the biological positive definition.
    keys=json.loads(a.crispr_gallery_keys.read_text())
    if len(keys)!=1038 or len(set(keys))!=1038 or any(not k.startswith('full:') for k in keys):
        raise ValueError('Expected 1038 unique full-gallery keys')
    selected=[int(k.split(':')[1]) for k in keys]
    full=records[records.cell.eq('A549')].sort_values('expression_row').reset_index(drop=True)
    if full.expression_row.tolist()!=list(range(1039)):raise ValueError('A549 row alignment mismatch')
    gallery=full.iloc[selected].copy()
    beta_path=proc/'metadata/compoundinfo_beta.txt'
    beta=pd.read_csv(beta_path,sep='\t',low_memory=False)
    targets={}
    for row in beta.itertuples():
        annotations=set() if pd.isna(row.target) else {x.strip() for x in str(row.target).upper().split('|') if x.strip()}
        targets.setdefault(str(row.cmap_name).lower().strip(),set()).update(annotations)
    gene_path=proc/'retraining_candidate_v1/cpg0000_crispr/meta_cpg0000_A549.csv'
    split_path=proc/'metadata/crispr_gene_split_v8_90_40_seed42.csv'
    genes=pd.read_csv(gene_path).sort_values('int_label')
    splits=pd.read_csv(split_path)
    held_genes=set(splits.loc[splits.split.eq('test'),'gene'])
    query=[];positive=[];evidence=[];coverage=[]
    for gene in genes.itertuples():
        if gene.gene not in held_genes:continue
        anchors=[r for r in full.itertuples() if gene.gene in targets.get(r.name.lower().strip(),set())]
        anchor_ids={r.record_id for r in anchors}
        anchor_moas=set().union(*(moa_set(r.moa) for r in anchors)) if anchors else set()
        hits=[r for r in gallery.itertuples() if r.record_id in anchor_ids or moa_set(r.moa)&anchor_moas]
        coverage.append(dict(gene=gene.gene,n_positive=len(hits),eligible=bool(hits)))
        if not hits:continue
        qid=f'cpg0000:A549:gene:{gene.gene}'
        query.append(dict(record_id=qid,gene=gene.gene,h5_label=int(gene.int_label)))
        for hit in hits:
            shared=moa_set(hit.moa)&anchor_moas
            positive.append((qid,hit.record_id))
            evidence.append(dict(query_id=qid,gallery_id=hit.record_id,
                direct_target=hit.record_id in anchor_ids,shared_moa='|'.join(sorted(shared)),
                anchor_record_ids='|'.join(r.record_id for r in anchors
                    if r.record_id==hit.record_id or moa_set(r.moa)&shared)))
    if len(query)!=29:raise ValueError(f'Expected 29 A549 queries, got {len(query)}')
    tasks['crispr_a549']=save_task(a.output/'crispr_a549',pd.DataFrame(query),gallery,positive,evidence)
    pd.DataFrame(coverage).to_csv(a.output/'crispr_a549/heldout_coverage.csv',index=False)
    sources=[meta_path,train_path,held_path,beta_path,gene_path,split_path]
    report=dict(tasks=tasks,source_hashes={str(f.relative_to(proc)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources},
                gallery_keys_sha256=hashlib.sha256(a.crispr_gallery_keys.read_bytes()).hexdigest(),
                rule='CRISPR anchors on full gallery, one-hop MOA expansion, then frozen common-gallery filtering')
    (a.output/'external_task_provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(tasks,indent=2))

if __name__=='__main__':main()
