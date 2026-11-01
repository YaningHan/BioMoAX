"""Fixed MOA-stratified record partition used by training and preparation."""
import numpy as np

def moa_split(df, label_col, seed=42):
    rng = np.random.default_rng(seed)
    eval_set = set()
    for moa, grp in df[df['moa'].notna()].groupby('moa'):
        idxs = grp[label_col].tolist()
        if len(idxs) >= 4:
            eval_set.update(rng.choice(idxs, max(1, len(idxs) // 2), replace=False).tolist())
    return (sorted([l for l in df[label_col].tolist() if l not in eval_set]), sorted(eval_set))
