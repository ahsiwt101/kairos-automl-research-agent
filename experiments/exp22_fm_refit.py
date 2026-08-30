"""Should the FM ensemble members be refit on train+validation for the final submission?

exp16 tested this for the GBDT and the backtests said +0.0000, so we declined it. But the
FM is a different model on a different feature space, and one datapoint hints the other
way: baseline_signal.py trains its test-window FM on everything <= the horizon (i.e.
train+valid, 1.27M rows) and scored 0.5953, against the official train-only FM's 0.5946.

Same discipline as exp16: you cannot check this on validation (training on it destroys the
signal), so validate the PROCEDURE on backtest folds where the test window is unsealed,
using the epoch count the train-only run selected. Decide on the backtests only.
"""
import sys, json; sys.path.insert(0,'.')
import numpy as np
from kairos.kernel.dataset import Data, FOLDS
from kairos.kernel.features import Encoder
from kairos.kernel.fastmetrics import fast_evaluate, factorize
from kairos.models.train import train_fm, predict

d = Data()
SEEDS = (0, 1, 2)
rows = []
for fname in ('backtest_a', 'backtest_c', 'official'):
    fold = d.fold(fname)
    tr, va, te = fold.idx['train'], fold.idx['valid'], fold.idx['test']
    enc = Encoder(d).fit(tr)
    gte,_ = factorize(d.user_id[te])

    def wrank(s, users):
        s = np.asarray(s, float)
        o = np.lexsort((np.arange(len(s)), -s, users)); u = users[o]
        st = np.flatnonzero(np.r_[True, u[1:]!=u[:-1]]); sz = np.diff(np.r_[st, len(u)])
        seg = np.repeat(np.arange(len(st)), sz)
        p = 1.0 - (np.arange(len(u)) - st[seg]) / np.maximum(sz[seg]-1, 1)
        r = np.empty(len(s)); r[o] = p; return r

    # pass 1: train-only, and record the epoch each seed selected on validation
    epochs, te_only = [], []
    for sd in SEEDS:
        r = train_fm(fold, enc, loss='bce', seed=sd, recency_tau=14)
        epochs.append(r['best_epoch'])
        te_only.append(wrank(predict(r['model'], enc, te), d.user_id[te]))
    n_ep = int(np.median(epochs))

    # pass 2: refit on train+valid for the SAME number of epochs (no early stopping
    # available - validation is now training data, which is exactly the difficulty)
    te_refit = []
    for sd in SEEDS:
        r = train_fm(fold, enc, loss='bce', seed=sd, recency_tau=14,
                     epochs=n_ep, patience=n_ep + 1, train_parts=('train', 'valid'))
        te_refit.append(wrank(predict(r['model'], enc, te), d.user_id[te]))

    def sc(pred):
        if fname == 'official':
            return fold.scorers['test'].score(pred, reason='exp22 fm refit')['primary']
        return fast_evaluate(gte, d.y_raw[te], pred)['primary']
    a, b = sc(np.mean(te_only,0)), sc(np.mean(te_refit,0))
    rows.append({'fold': fname, 'epochs': n_ep, 'train_only': a, 'refit': b, 'delta': b-a,
                 'extra_rows': int(len(va))})
    print(f"{fname:12s} epochs={n_ep:2d}  train-only {a:.4f}  "
          f"refit(+{len(va):,}) {b:.4f}  delta {b-a:+.4f}", flush=True)

bt = [r['delta'] for r in rows if r['fold'].startswith('backtest')]
print(f"\nbacktest verdict: refitting the FM is worth {np.mean(bt):+.4f} on average "
      f"({', '.join(f'{x:+.4f}' for x in bt)})")
print("official row is reference only - the DECISION is made on the backtests.")
json.dump(rows, open('runs/exp22_fm_refit.json','w'), indent=2, default=float)
