"""Per-split-optimal baseline score: the refit procedure exp22 confirmed, made safe to use.

exp22 established on two independent backtest folds (+0.0020, +0.0022) that refitting the
FM on train+validation beats training on train alone. But the procedure has an asymmetry
that is easy to get wrong and silently self-deceiving: a model refit on validation has SEEN
validation, so its validation score is meaningless and any blend weight fitted against it
is fitted against a lie.

So this primitive resolves the asymmetry per row rather than handing the caller a footgun:

    train / valid rows -> prediction from a model trained on TRAIN ONLY
    test rows          -> prediction from a model trained on TRAIN + VALIDATION

A caller therefore fits blend weights on validation against honest train-only predictions,
while the test predictions it finally emits carry the refit benefit - which is exactly the
procedure exp22 validated, with no way to accidentally invert it. No test label is ever
used; the refit model trains on train+valid only.

Epoch count for the refit comes from the train-only run's early-stopping choice, because
early stopping is unavailable once validation has become training data.
"""
import os
import numpy as np
from kairos.kernel.dataset import variant_path

CACHE_DIR = variant_path('runs/refit_cache')


def build_refit_signal(data, fold, seeds=(0, 1, 2), recency_tau=14, cache_dir=CACHE_DIR,
                       force=False):
    """Returns float32 (n,) - see the module docstring for the per-split semantics."""
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f'{fold.name}_tau{recency_tau}.npy')
    if os.path.exists(cache) and not force:
        return np.load(cache)

    from kairos.kernel.features import Encoder
    from kairos.models.train import train_fm, predict

    tr, va, te = fold.idx['train'], fold.idx['valid'], fold.idx['test']
    enc = Encoder(data).fit(tr)
    out = np.zeros(data.n, dtype=np.float32)

    # pass 1: train-only models -> honest scores for train and validation rows
    epochs, tr_preds, va_preds = [], [], []
    for sd in seeds:
        r = train_fm(fold, enc, loss='bce', seed=sd, recency_tau=recency_tau)
        epochs.append(r['best_epoch'])
        tr_preds.append(predict(r['model'], enc, tr))
        va_preds.append(predict(r['model'], enc, va))
    out[tr] = np.mean(tr_preds, 0).astype(np.float32)
    out[va] = np.mean(va_preds, 0).astype(np.float32)
    n_ep = int(np.median(epochs))

    # pass 2: refit on train+valid for that many epochs -> scores for test rows only
    te_preds = []
    for sd in seeds:
        r = train_fm(fold, enc, loss='bce', seed=sd, recency_tau=recency_tau,
                     epochs=n_ep, patience=n_ep + 1, train_parts=('train', 'valid'))
        te_preds.append(predict(r['model'], enc, te))
    out[te] = np.mean(te_preds, 0).astype(np.float32)

    np.save(cache, out)
    return out
