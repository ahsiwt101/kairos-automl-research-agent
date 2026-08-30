"""Generate FM predictions under the backtest-confirmed refit procedure.

exp22 confirmed on BOTH backtest folds (+0.0020, +0.0022) that refitting the FM on
train+validation beats train-only. Apply it to the official fold.

The asymmetry this creates is deliberate and is the whole reason exp22 had to validate the
PROCEDURE rather than the model: weights must still be fitted using TRAIN-ONLY models
scored on validation (a refit model has seen validation, so its validation score is
meaningless), and only the final TEST predictions come from the refit models. Epoch count
is taken from the train-only run, since early stopping is unavailable once validation is
training data.
"""
import sys, json; sys.path.insert(0,'.')
import numpy as np
from kairos.kernel.dataset import Data
from kairos.kernel.features import Encoder
from kairos.models.train import train_fm, predict

d = Data(); fold = d.fold('official')
tr, va, te = fold.idx['train'], fold.idx['valid'], fold.idx['test']
enc = Encoder(d).fit(tr)

epochs = []
for sd in range(3):
    r = train_fm(fold, enc, loss='bce', seed=sd, recency_tau=14)
    epochs.append(r['best_epoch'])
    np.save(f'runs/fmT_s{sd}_va.npy', predict(r['model'], enc, va))   # weight fitting
    print(f"  train-only seed{sd}: valid {r['valid']['primary']:.4f} "
          f"(best epoch {r['best_epoch']})", flush=True)
n_ep = int(np.median(epochs))
print(f"epoch count carried to refit: {n_ep}")

for sd in range(3):
    r = train_fm(fold, enc, loss='bce', seed=sd, recency_tau=14,
                 epochs=n_ep, patience=n_ep+1, train_parts=('train','valid'))
    np.save(f'runs/fmR_s{sd}_te.npy', predict(r['model'], enc, te))   # final test preds
    print(f"  refit seed{sd}: trained on train+valid for {n_ep} epochs", flush=True)
print("saved: fmT_*_va.npy (train-only, for weights) and fmR_*_te.npy (refit, for test)")
