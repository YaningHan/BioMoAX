"""Build core MOA task lists from fixed drug records, without model-dependent selection."""
import argparse
import json
from pathlib import Path
import pandas as pd

def moa_set(value):
    return {x.strip() for x in str(value).split('|') if x.strip()} if pd.notna(value) else set()

def build_task(records, query_cell, gallery_cells):
    query=records[records.cell.eq(query_cell)&records.partition.eq('held_out')].copy()
    gallery=records[records.cell.isin(gallery_cells)].copy()
    gs={row.record_id:moa_set(row.moa) for row in gallery.itertuples()}
    positives=[]; exclusions=[]; eligible=[]
    for row in query.itertuples():
        annotation=moa_set(row.moa)
        hits=[gid for gid,labels in gs.items() if annotation & labels and gid!=row.record_id]
        if not hits:
            continue
        eligible.append(row.record_id)
        positives.extend((row.record_id,gid) for gid in hits)
        if row.record_id in gs:
            exclusions.append((row.record_id,row.record_id))
    return query[query.record_id.isin(eligible)],gallery,positives,exclusions

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    records=pd.read_csv(a.records,keep_default_na=False)
    if records.record_id.duplicated().any():raise ValueError('Duplicate record IDs')
    tasks=[('within_u2os','U2OS',['U2OS']),('within_a549','A549',['A549']),
           ('u2os_to_a549','U2OS',['A549']),('a549_to_u2os','A549',['U2OS']),
           ('ht29_to_reference','HT29',['U2OS','A549'])]
    a.output.mkdir(parents=True)
    counts={}
    for name,qcell,gcells in tasks:
        q,g,pos,exc=build_task(records,qcell,gcells)
        out=a.output/name;out.mkdir()
        q.to_csv(out/'query_ids.csv',index=False);g.to_csv(out/'gallery_ids.csv',index=False)
        for filename,rows in [('positive_pairs.csv',pos),('excluded_pairs.csv',exc)]:
            pd.DataFrame(rows,columns=['query_id','gallery_id']).to_csv(out/filename,index=False)
        counts[name]=dict(eligible_queries=len(q),gallery_records=len(g),positive_pairs=len(pos),excluded_pairs=len(exc))
    (a.output/'task_summary.json').write_text(json.dumps(dict(tasks=counts,
        scope='Core MOA annotation-defined tasks; method-specific missing-feature masks are not applied',
        excluded_tasks=['Crop-split identity retrieval','Cross-batch','CRISPR target/MOA expansion']),indent=2)+'\n')
    print(json.dumps(counts,indent=2))

if __name__=='__main__':main()
