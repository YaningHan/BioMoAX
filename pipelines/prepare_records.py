"""Export fixed core drug partitions and processed image/expression row mappings."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
import pandas as pd

CELLS = {
    'U2OS': ('cpg0012', 'L1000_JumpCP_meta.csv', 'crops_cpg0012_690.h5', 690),
    'A549': ('cpg0004', 'l1000_a549_1039_meta.csv', 'crops_cpg0004_a549x1055.h5', 1039),
    'A375': ('local_cpg0016', 'cpg0016_A375_meta.csv', 'crops_cpg0016_A375.h5', 86),
    'MCF7': ('local_cpg0016', 'cpg0016_MCF7_meta.csv', 'crops_cpg0016_MCF7.h5', 85),
    'HT29': ('local_cpg0016', 'cpg0016_HT29_meta.csv', 'crops_cpg0016_HT29.h5', 78),
}

def split_function():
    # Execute only the pure split function from the shipped training source.
    path = Path(__file__).resolve().parents[1]/'src/biomoax/legacy/train_moax_v7.py'
    tree = ast.parse(path.read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name=='moa_split')
    scope = {'np':np}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), scope)
    return scope['moa_split']

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    proc = a.project_root/'data/processed'
    source_files = []
    prof_path = proc/'metadata/cpg0012_profiles_merged.csv'
    prof = pd.read_csv(prof_path, usecols=['Metadata_source_name','Metadata_cpd_name'])
    tocris = set(prof.loc[prof.Metadata_source_name.eq('Tocris Bioscience'),'Metadata_cpd_name'].str.lower())
    source_files.append(prof_path)
    split = split_function()
    records, counts = [], {}
    for cell,(source,meta_name,crop_name,n) in CELLS.items():
        path = proc/'metadata'/meta_name
        meta = pd.read_csv(path)
        source_files.append(path)
        if len(meta)!=n:
            raise ValueError(f'{cell}: expected {n} metadata rows')
        meta['row_index'] = np.arange(n)
        if cell!='U2OS' and not np.array_equal(meta.new_int_label.to_numpy(),np.arange(n)):
            raise ValueError('Expression row order differs from local labels: '+cell)
        selected = meta
        if cell=='U2OS':
            selected = meta[~meta.cmap_name.str.lower().isin(tocris)]
        train, held = split(selected, 'row_index') if cell!='HT29' else ([],list(range(n)))
        train,held = set(train),set(held)
        if cell=='U2OS' and (len(train),len(held))!=(473,90):
            raise ValueError('Unexpected U2OS partition')
        if cell=='A549' and (len(train),len(held))!=(738,301):
            raise ValueError('Unexpected A549 partition')
        crop_rel = 'crops/'+crop_name
        expr_rel = f'features/l1000_{cell.lower()}_{n}.npy'
        expr = np.load(proc/expr_rel,mmap_mode='r',allow_pickle=False)
        if expr.shape!=(n,12328) or not np.isfinite(expr).all():
            raise ValueError('Invalid expression array: '+cell)
        with h5py.File(proc/crop_rel) as h:
            labels = h['labels'][:]
            unique,frequency = np.unique(labels,return_counts=True)
            count_map = dict(zip(unique,frequency))
        counts[cell] = dict(train=len(train),held_out=len(held),gallery_only=n-len(train)-len(held))
        for i,row in meta.iterrows():
            # U2OS and the auxiliary cells use local row labels; A549 retains original HDF5 labels.
            label = int(row.int_label) if cell=='A549' else i
            def text(key):
                value=row.get(key,'')
                return '' if pd.isna(value) else str(value)
            records.append(dict(record_id=f'{source}:{cell}:{i}',source=source,cell=cell,
                expression_row=i,h5_label=label,crop_file=crop_rel,expression_file=expr_rel,
                source_label=text('int_label'),broad_id=text('broad_id'),jcp2022=text('jcp2022'),
                name=text('cmap_name') or text('cpd_name'),moa=text('moa'),
                partition='train' if i in train else 'held_out' if i in held else 'gallery_only',
                n_crops=int(count_map.get(label,0)),expression_nonzero=bool(np.any(expr[i]!=0))))
    # Independent cross-check against the existing training-site partition.
    sites=pd.read_csv(proc/'metadata/u2os_cpg0012_train473_site_split_seed42.csv')
    utrain={r['expression_row'] for r in records if r['cell']=='U2OS' and r['partition']=='train'}
    if set(sites.new_int_label)!=utrain:
        raise ValueError('U2OS drug partition disagrees with saved site split')
    a.output.mkdir(parents=True)
    pd.DataFrame(records).to_csv(a.output/'drug_records.csv',index=False)
    provenance=dict(counts=counts, scope='Processed core drug mappings; not raw L1000 signature matching',
        source_hashes={str(f.relative_to(proc)):hashlib.sha256(f.read_bytes()).hexdigest() for f in source_files})
    (a.output/'record_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(counts,indent=2))

if __name__=='__main__':
    main()
