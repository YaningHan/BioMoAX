"""Isolated v12 1-vs-4 crop experiment; historical defaults stay unchanged."""

import os
import argparse
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
import train_moax_v7 as base
import train_moax_v8_crispr130 as v8
import train_moax_v12_a549_crispr_only as v12

ROOT=Path(os.environ['BIOMOAX_PROJECT_ROOT'])
OUT=ROOT/'logs/crop_ablation_20260909'

def encode_drug_images(backbone,images):
    """Mean PRE-projection trunk vectors; all cells retain gradients.

    Use identical chunk/checkpoint machinery for both arms. CPU/CUDA torch RNG
    is preserved on recomputation, including stochastic augmentation/dropout.
    This is not gradient accumulation: all 1000 drugs share the same loss.
    """
    b,k,c,h,w=images.shape
    flat=images.reshape(b*k,c,h,w)
    parts=[]
    for chunk in flat.split(64):
        parts.append(checkpoint(lambda x:backbone(x,augment=True),chunk,
                                use_reentrant=False,preserve_rng_state=True))
    return torch.cat(parts).reshape(b,k,-1).mean(1)

def configure(crops,seed,pilot):
    label=f'{"pilot" if pilot else "formal"}_k{crops}_seed{seed}'
    log=OUT/label
    if (log/'run_config.json').exists():
        raise RuntimeError(f'Refusing to overwrite or implicitly resume {log}')
    split=v8.build_or_validate_split()
    test=set(split.loc[split.split.eq('test'),'gene'].astype(str))
    base.HELD_OUT_GENES_A549=test;base.HELD_OUT_GENES_U2OS='ALL'
    base.N_CRISPR_CACHE=500
    excluded=v12.heldout_u2os_source_rows()
    base.EXCLUDED_SOURCE_ROWS_BY_CELL={'U2OS':excluded}
    base.TRAINING_EXPANSION={};base.L1000_AVAILABLE_IDS=None
    base.BATCH_DRUGS=1000;base.K_PER_EPOCH=300;base.N_CROPS_BATCH=crops
    base.USE_MOA_BALANCED_SAMPLER=True;base.MIN_DRUGS_PER_MOA=2;base.MOA_BALANCED_FRACTION=.5
    base.GENE_I2I_PER_BATCH=24;base.CRISPR_I2E_PER_BATCH=48
    base.EPOCHS=60;base.STOP_AFTER_EPOCH=2 if pilot else 60
    base.RANDOM_SEED=seed;base.LOG_DIR=log
    base.DRUG_IMAGE_ENCODER=encode_drug_images
    # Sequential jobs reuse only this experiment's regenerable memmap space.
    v8.MEMMAP_DIR=ROOT/'data/tmp/crop_ablation_20260909'
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    log.mkdir(parents=True,exist_ok=True)
    config=dict(crops=crops,seed=seed,pilot=pilot,batch_drugs=1000,steps_per_epoch=300,
        scheduler_epochs=60,run_epochs=base.STOP_AFTER_EPOCH,aggregation='mean trunk before heads',
        gradient_checkpoint_chunk=64,crispr_aggregation='unchanged single crop',
        checkpoint_selection='qualified four thresholds, highest combined validation score; earliest tie; none if no qualified',
        external_metrics_used_for_selection=False,
        gene_split_sha256=hashlib.sha256(v8.SPLIT_PATH.read_bytes()).hexdigest(),
        site_split_sha256=hashlib.sha256(v12.SITE_SPLIT.read_bytes()).hexdigest(),
        heldout_crops=len(excluded))
    (log/'run_config.json').write_text(json.dumps(config,indent=2)+'\n')
    # Archive the code actually used, as the shared base may evolve later.
    for path in [Path(__file__),Path(base.__file__),Path(v8.__file__),Path(v12.__file__)]:
        (log/path.name).write_bytes(path.read_bytes())
    print(json.dumps(config,indent=2),flush=True)
    return log

def main():
    p=argparse.ArgumentParser();p.add_argument('--crops',type=int,choices=[1,4],required=True)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--pilot',action='store_true');a=p.parse_args()
    log=configure(a.crops,a.seed,a.pilot)
    v8.install_disk_backed_reader();base.main()
    import pandas as pd
    h=pd.read_csv(log/'train_history.csv')
    assert len(h)==base.STOP_AFTER_EPOCH
    assert np.isfinite(h.select_dtypes('number').to_numpy()).all()
    good=h[(h.u2os_i2i>=base.THRESH_I2I_U2OS)&(h.a549_i2i>=base.THRESH_I2I_A549)&
           (h.u2os_e2e>=base.THRESH_E2E_U2OS)&(h.a549_e2e>=base.THRESH_E2E_A549)]
    chosen=None if good.empty else int(good.sort_values(['combined','epoch'],ascending=[False,True]).iloc[0].epoch)
    (log/'completion.json').write_text(json.dumps(dict(epochs=len(h),selected_epoch=chosen,pilot=a.pilot))+'\n')

if __name__=='__main__':main()
