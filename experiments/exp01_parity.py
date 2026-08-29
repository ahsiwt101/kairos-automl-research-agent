"""Parity check: torch FM + pointwise BCE must reproduce the official numpy baseline.
Without this, nothing measured against 'the baseline' later means anything."""
import sys, time; sys.path.insert(0,'.')
import numpy as np
from kairos.kernel.dataset import Data
from kairos.kernel.features import Encoder
from kairos.models.train import train_fm, predict
from kairos.kernel.fastmetrics import fast_evaluate, factorize

d = Data(); fold = d.fold('official')
enc = Encoder(d).fit(fold.idx['train'])
print(f"encoder: fields={enc.names()} dim={enc.dim:,}")
t0 = time.time()
r = train_fm(fold, enc, loss='bce', seed=0, verbose=True)
print(f"\ntrain time {time.time()-t0:.0f}s, best epoch {r['best_epoch']}")
print(f"VALID  GAUC {r['valid']['GAUC']:.4f} nDCG@5 {r['valid']['nDCG@5']:.4f} "
      f"primary {r['valid']['primary']:.4f}   (official numpy FM: 0.6674 / 0.5357 / 0.6016)")
st = predict(r['model'], enc, fold.idx['test'])
m = fold.scorers['test'].score(st, reason='exp01 parity check')
print(f"TEST   GAUC {m['GAUC']:.4f} nDCG@5 {m['nDCG@5']:.4f} "
      f"primary {m['primary']:.4f}   (official numpy FM: 0.6610 / 0.5282 / 0.5946)")
