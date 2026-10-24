"""Full five-loss training on the user-authorized, QC-flagged candidate bundle."""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '2'
from pathlib import Path
import argparse
import functools
import inspect
import json
import hashlib
import random
import traceback
import numpy as np
import pandas as pd
import h5py
import sys

R = Path(os.environ['BIOMOAX_PROJECT_ROOT'])
OUT = Path(os.environ['BIOMOAX_TRAIN_OUT'])
BUNDLE = R / 'data/processed/retraining_candidate_v1'
ARCHIVE = Path(__file__).resolve().parent
sys.path.insert(0, str(ARCHIVE))
import train_k1_multiseed_fixedval_20260911 as fixed
import torch
from crispr_i2e_policy import configure as loss_configure


class PreflightDone(Exception):
    pass


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def main():
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.set_num_threads(2)
    fixed.OUT = OUT
    log = fixed.configure(1, args.seed, args.preflight)
    base = fixed.experiment.base; v8 = fixed.experiment.v8
    state = log / 'status.json'
    write(state, {'state': 'LOADING', 'seed': args.seed, 'preflight': args.preflight})
    try:
        policy = loss_configure(base, log, 'off')
        split = pd.read_csv(v8.SPLIT_PATH)
        held = set(split.loc[split.split.eq('test'), 'gene'].astype(str))
        assert len(held) == 40
        base.HELD_OUT_GENES_A549 = held; base.HELD_OUT_GENES_U2OS = held
        base.CRISPR_DIR = BUNDLE / 'cpg0000_crispr'
        base.CPD_DIR = BUNDLE / 'cpg0000_compound'
        for cell in ['A549','U2OS']:
            base.H5['cpd_'+cell] = base.CPD_DIR / f'crops_cpg0000_cpd_{cell}.h5'
        base.EPOCHS = 60; base.STOP_AFTER_EPOCH = 60
        base.K_PER_EPOCH = 2 if args.preflight else 300
        base.BATCH_DRUGS = 1000
        base.N_CRISPR_CACHE = 2 if args.preflight else 500
        # Shared serial cache belongs exclusively to this queue; no old cache is touched.
        v8.MEMMAP_DIR = R / 'data/tmp/rebuilt_five_loss_queue_cache'
        base.autocast = lambda device_type, enabled=True: torch.autocast(device_type, enabled=enabled, dtype=torch.bfloat16)
        class CheckedScaler(torch.cuda.amp.GradScaler):
            def __init__(self, **unused):
                super().__init__(enabled=False)
            def unscale_(self, optimizer):
                result = super().unscale_(optimizer)
                # Fail before optimizer or scheduler step if any gradient is nonfinite.
                flags = [torch.isfinite(p.grad).all() for g in optimizer.param_groups for p in g['params'] if p.grad is not None]
                assert flags and bool(torch.stack(flags).all()), 'Nonfinite or missing gradients'
                return result
        base.GradScaler = CheckedScaler
        from ram_crop_reader import make_reader, copy_selected
        limit_path = Path('/sys/fs/cgroup/memory.max')
        limit = limit_path.read_text().strip() if limit_path.exists() else 'max'
        assert limit == 'max' or int(limit) >= 100 * 2**30, 'RAM mode requires >=100GiB cgroup limit'
        reader = make_reader(v8, base, log)
        if args.preflight:
            for name, arg in [('load_cell','max_crops_per_label'), ('load_with_global_offset','max_crops_per_label'), ('load_cpg0030','max_crops')]:
                fn = getattr(reader, name)
                def capped(self, *values, _fn=fn, _arg=arg, **kwargs):
                    bound = inspect.signature(_fn).bind(self, *values, **kwargs)
                    bound.apply_defaults(); bound.arguments[_arg] = 2
                    return _fn(*bound.args, **bound.kwargs)
                setattr(reader, name, capped)
        audits = {}
        def load_crispr(path, meta_csv, held_out_genes=None, n_per_gene=500):
            assert set(held_out_genes) == held
            meta = pd.read_csv(meta_csv)
            assert meta.int_label.tolist() == list(range(len(meta)))
            rng = np.random.default_rng(base.RANDOM_SEED)
            crops = {}
            with h5py.File(path, 'r') as f:
                labels = f['labels'][:]
                for row in meta.itertuples():
                    if row.gene in held: continue
                    ix = np.flatnonzero(labels == row.int_label)
                    if len(ix) > n_per_gene: ix = np.sort(rng.choice(ix, n_per_gene, replace=False))
                    if len(ix):
                        x = np.empty((len(ix), 5, 128, 128), np.uint16)
                        copy_selected(f['crops'], ix, x)
                        crops[row.int_label] = x
            assert len(crops) == 90
            v8.evict_file_cache(path)
            audits[str(path)] = {'train_genes': len(crops), 'heldout_overlap': 0, 'cached_crops': sum(len(x) for x in crops.values())}
            write(log / 'supervision_audit.json', audits)
            return crops, meta.gene.tolist()
        base.load_crispr_h5 = load_crispr
        # Preserve every qualified epoch. Latest checkpoint includes optimizer/scheduler.
        source = inspect.getsource(base.main)
        assert source.count('if epoch % 20 == 0:') == 1
        source = source.replace('if epoch % 20 == 0:', 'if True:')
        source = source.replace('LOG_DIR / f"checkpoint_ep{epoch:03d}.pt"', 'LOG_DIR / "latest_checkpoint.pt"')
        source = source.replace('if epoch % ZEROSHOT_FREQ == 0 or epoch == EPOCHS:', 'if False:  # no test-set evaluation during training')
        code_path = log / 'effective_main.py'; code_path.write_text(source)
        exec(compile(source, str(code_path), 'exec'), base.main.__globals__)
        real = base.run_epoch; epoch_count = [0]
        @functools.wraps(real)
        def checked_epoch(*values, **kwargs):
            bound = inspect.signature(real).bind(*values, **kwargs)
            pairs = bound.arguments['valid_gene_i2i_indices']
            assert all(sum(c == cell for c, _ in pairs) == 90 for cell in ['A549', 'U2OS'])
            updates = [0]
            handle = bound.arguments['optimizer'].register_step_post_hook(lambda *unused: updates.__setitem__(0, updates[0]+1))
            torch.cuda.reset_peak_memory_stats()
            write(state, {'state':'TRAINING' if not args.preflight else 'PREFLIGHT', 'seed':args.seed, 'epoch':epoch_count[0]+1})
            try: losses = real(*bound.args, **bound.kwargs)
            finally: handle.remove()
            assert all(np.isfinite(v) for v in losses.values()) and losses['crispr_i2e'] == 0
            assert updates[0] == base.K_PER_EPOCH
            epoch_count[0] += 1
            result = {'epoch':epoch_count[0], 'actual_updates':updates[0], 'losses':losses,
                      'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(), 'batch_drugs':base.BATCH_DRUGS}
            write(log / 'last_epoch_numerics.json', result)
            if args.preflight:
                write(state, dict(result, state='PREFLIGHT_PASS', cache_cap=2, formal_training_started=False))
                raise PreflightDone()
            return losses
        base.run_epoch = checked_epoch
        save = torch.save
        def atomic_save(obj, path, *values, **kwargs):
            assert not args.preflight, 'Preflight must not save checkpoints'
            if isinstance(obj, dict):
                obj.update(python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), loss_policy=policy)
            path = Path(path); tmp = path.with_suffix(path.suffix + '.partial')
            save(obj, tmp, *values, **kwargs); tmp.replace(path)
        torch.save = atomic_save
        config = json.loads((log/'run_config.json').read_text())
        config.update(policy, precision='BF16', run_epochs=60, steps_per_epoch=base.K_PER_EPOCH,
                      source_crop_cap=2 if args.preflight else 150,
                      checkpoint_selection='SAVE ALL epochs meeting four thresholds; no final selection',
                      data_policy='User explicitly authorized current unfiltered candidate data with QC flags retained',
                      quality_certified=False, bundle_sha256=hashlib.sha256((BUNDLE/'complete.json').read_bytes()).hexdigest(),
                      paths={k:str(v) for k,v in base.H5.items()}, crispr_directory=str(base.CRISPR_DIR),
                      memmap_directory=None, crop_storage='resident CPU RAM uint16',
                      drug_cache_budget_gib=64, estimated_crispr_cache_upper_gib=13.74,
                      limitations=['staining QC unresolved; flags retained', 'A549 per-crop site provenance unavailable', 'CP/DP repairs proceed independently', 'historical drug validation splits retained; new cpg0000 uses shared 90/40 CRISPR gene split'])
        write(log/'run_config.json', config)
        for p in [Path(__file__), Path(__file__).with_name('crispr_i2e_policy.py'), Path(__file__).with_name('ram_crop_reader.py')]:
            (log/p.name).write_bytes(p.read_bytes())
        base.MultiH5Reader = reader
        base.main()
        history = pd.read_csv(log/'train_history.csv')
        assert len(history) == 60 and np.isfinite(history.select_dtypes('number').to_numpy()).all()
        good = history[(history.u2os_i2i >= base.THRESH_I2I_U2OS) & (history.a549_i2i >= base.THRESH_I2I_A549) & (history.u2os_e2e >= base.THRESH_E2E_U2OS) & (history.a549_e2e >= base.THRESH_E2E_A549)]
        assert all((log/f'qualified_ep{int(ep):03d}.pt').exists() for ep in good.epoch)
        good.to_csv(log/'qualified_epochs.csv', index=False)
        write(state, {'state':'COMPLETE','epochs':60,'qualified_epochs':good.epoch.astype(int).tolist(), 'selected_epoch':None})
    except PreflightDone:
        print('PREFLIGHT PASS', flush=True)
    except BaseException as exc:
        write(state, {'state':'FAILED', 'error':str(exc), 'seed':args.seed})
        traceback.print_exc(); raise


if __name__ == '__main__':
    main()
