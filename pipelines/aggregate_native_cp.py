"""Aggregate U2OS native CellProfiler wells by compound using medians."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from common import new_output, sha256, write_record


def aggregate(wells, columns, compound_column, label_column):
    required = [compound_column, label_column, *columns]
    if not columns or len(set(columns)) != len(columns):
        raise ValueError('Feature list must be nonempty and unique')
    if wells.empty or not set(required) <= set(wells):
        raise ValueError('Missing well rows or required columns')
    if wells[[compound_column,label_column]].isna().any().any():
        raise ValueError('Missing compound identifiers or labels')
    identity = wells[[compound_column,label_column]].drop_duplicates()
    if identity[compound_column].duplicated().any() or identity[label_column].duplicated().any():
        raise ValueError('Compound-to-label mapping must be one-to-one')
    values = wells[columns].apply(pd.to_numeric, errors='raise')
    if np.isinf(values.to_numpy()).any():
        raise ValueError('Infinite feature values')
    numeric = pd.concat([wells[[compound_column]],values],axis=1)
    result = numeric.groupby(compound_column,sort=True)[columns].median()
    result = result.fillna(result.median(axis=0))
    if result.isna().any().any():
        raise ValueError('Feature columns without any finite compound median')
    return result.reset_index().merge(identity,on=compound_column,validate='one_to_one')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('wells','columns','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--compound-column',default='brd_core')
    p.add_argument('--label-column',default='int_label')
    p.add_argument('--reference',type=Path)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    columns=[c.strip() for c in args.columns.read_text().splitlines() if c.strip()]
    wells=pd.read_csv(args.wells,low_memory=False)
    result=aggregate(wells,columns,args.compound_column,args.label_column)
    error=None
    if args.reference:
        ref=pd.read_csv(args.reference).set_index(args.compound_column)
        actual=result.set_index(args.compound_column)
        if ref.index.duplicated().any() or set(ref.index)!=set(actual.index):
            raise ValueError('Reference compound coverage differs')
        ref=ref.loc[actual.index]
        if not np.array_equal(actual[args.label_column],ref[args.label_column]):
            raise ValueError('Reference labels differ')
        x,y=actual[columns].to_numpy(),ref[columns].to_numpy()
        error=float(np.max(np.abs(x-y)))
        if not np.allclose(x,y,rtol=1e-6,atol=1e-7):
            raise ValueError(f'Reference mismatch: {error}')
    out=new_output(args.output)
    result.to_csv(out/'profiles.csv',index=False)
    write_record(out,status='complete',n_wells=len(wells),n_compounds=len(result),
                 aggregation='Median across wells; missing compound values filled by column median',
                 reference_max_abs=error,wells_sha256=sha256(args.wells),columns_sha256=sha256(args.columns))


if __name__=='__main__':
    main()
