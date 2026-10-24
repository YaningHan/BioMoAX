"""Candidate default: five losses; optional dual-cell CRISPR I2E ablation."""
import inspect
from pathlib import Path


def install_off_gate(base, output):
    if getattr(base, '_crispr_i2e_off_gate_installed', False):
        return
    source = inspect.getsource(base.run_epoch)
    old = '            if crispr_i2e_samples:'
    assert source.count(old) == 1, 'Unexpected loss-loop source; refuse unsafe patch'
    source = source.replace(old, '            if ALPHA_CRISPR_I2E > 0 and crispr_i2e_samples:')
    path = Path(output) / 'five_loss_run_epoch.py'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    scope = base.run_epoch.__globals__
    exec(compile(source, str(path), 'exec'), scope)
    base.run_epoch = scope['run_epoch']
    base._crispr_i2e_off_gate_installed = True
    # Fail loudly if a future edit accidentally enters the disabled branch.
    def forbidden_fetch(*args, **kwargs):
        raise AssertionError('Disabled CRISPR I2E branch executed')
    base.fetch_crispr_i2e_batch = forbidden_fetch


def configure(base, output, mode='off'):
    if mode == 'off':
        install_off_gate(base, output)
        base.ALPHA_CRISPR_I2E = 0.0
    elif mode == 'dualcell':
        if getattr(base, '_crispr_i2e_off_gate_installed', False):
            raise RuntimeError('Use a fresh process when switching loss policies')
        if not getattr(base, '_dualcell_crispr_i2e_installed', False):
            from dualcell_crispr_i2e import install
            install(base, output)
        base.ALPHA_CRISPR_I2E = 1.5
    else:
        raise ValueError(f'Unsupported CRISPR I2E mode: {mode}')
    return {'crispr_i2e_mode': mode, 'alpha_crispr_i2e': base.ALPHA_CRISPR_I2E,
            'crispr_i2e_sampling_and_forward': mode != 'off',
            'active_loss_count': 5 if mode == 'off' else 6}
