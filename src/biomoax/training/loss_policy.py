"""Fixed five-loss policy used by the selected paper checkpoints."""
import os

def configure(base, output, mode='off'):
    if mode != 'off':
        raise ValueError('Only the paper five-loss policy is included')
    base.ALPHA_CRISPR_I2E = 0.0

    def forbidden_fetch(*args, **kwargs):
        raise AssertionError('Disabled CRISPR I2E branch executed')
    base.fetch_crispr_i2e_batch = forbidden_fetch
    return {'crispr_i2e_mode': 'off', 'alpha_crispr_i2e': 0.0, 'crispr_i2e_sampling_and_forward': False, 'active_loss_count': 5}
