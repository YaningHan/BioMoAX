"""1-crop independent training seeds with fixed data and validation RNG."""
import functools,json,shutil
from pathlib import Path
import numpy as np
import train_crop_ablation_20260909 as experiment

ROOT=experiment.ROOT
OUT=ROOT/'logs/k1_multiseed_fixedval_20260911'
DATA_SEED=42
VALIDATION_SEEDS={'evaluate_i2i':42001,'evaluate_e2e':42002,'evaluate_i2e':42003,'evaluate_e2i':42004}

def fixed_numpy_rng(fn,seed):
    @functools.wraps(fn)
    def wrapped(*args,**kwargs):
        state=np.random.get_state()
        try:
            np.random.seed(seed)
            return fn(*args,**kwargs)
        finally:
            np.random.set_state(state)
    return wrapped

original_configure=experiment.configure
def configure(crops,seed,pilot):
    if crops!=1:raise ValueError('This queue is 1-crop only')
    experiment.OUT=OUT
    log=original_configure(crops,seed,pilot)
    # RANDOM_SEED controls source caching and crop partitions, not initialized
    # torch/NumPy/Python training RNG states (seeded by original_configure).
    experiment.base.RANDOM_SEED=DATA_SEED
    experiment.v8.MEMMAP_DIR=ROOT/f'data/tmp/k1_multiseed_fixedval_20260911/seed{seed}'
    for name,value in VALIDATION_SEEDS.items():
        setattr(experiment.base,name,fixed_numpy_rng(getattr(experiment.base,name),value))
    cfg=json.loads((log/'run_config.json').read_text())
    cfg.update(data_seed=DATA_SEED,training_seed=seed,validation_numpy_seeds=VALIDATION_SEEDS,
               validation_rng_restores_training_state=True,validation_crops=50,
               source_crop_cap=150,memmap_directory=str(experiment.v8.MEMMAP_DIR),
               protocol='fixed data42, original training query/gallery partitions, deterministic per-direction validation',
               comparison_note='Historical seed42 used stochastic validation; re-evaluate saved candidates before pooling.')
    (log/'run_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    shutil.copy2(__file__,log/Path(__file__).name)
    print('FIXED VALIDATION CONFIG',json.dumps(cfg),flush=True)
    return log

if __name__=='__main__':
    experiment.configure=configure
    experiment.main()
