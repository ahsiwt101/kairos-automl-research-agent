"""Item D: close the DIN train/serve history mismatch.

Measured: training rows average 7.3 history items (32% empty), validation and test average
~17 (4% empty), because 78% of training rows sit in the early window before histories
accumulate. Late-window training rows already have the right distribution (mean 17.2 vs
test 17.5) - there are just far fewer of them.

So this is a bias/variance trade: matching the serving distribution costs training data.
Three ways to spend it, against unweighted DIN's 0.6023, which is the number to beat.
"""
import sys, json; sys.path.insert(0,'.')
import numpy as np
from kairos.kernel.dataset import Data, FOLDS
from kairos.kernel.causal import window_horizons
from kairos.kernel.frozenfeat import windows_for_fold
from kairos.kernel.din_signal import build_din_signal
from kairos.kernel.fastmetrics import fast_evaluate, factorize

d = Data(); fold = d.fold('official')
va = fold.idx['valid']; g,_ = factorize(d.user_id[va]); y = d.y_raw[va]
hz = window_horizons(d.date.astype(np.int64), windows_for_fold(FOLDS['official']))

print(f"{'weighting':<14} {'valid':>8} {'vs 0.6023':>10}")
print("-"*36)
res = {}
for mode in (None, 'recency', 'hist_match', 'late_only'):
    s = build_din_signal(d, fold, hz, weight_mode=mode, force=True)
    p = fast_evaluate(g, y, s[va].astype(np.float64))['primary']
    res[str(mode)] = p
    print(f"{str(mode):<14} {p:>8.4f} {p-0.6023:>+10.4f}", flush=True)

best = max(res, key=res.get)
print(f"\nbest: {best} ({res[best]:.4f})")
print("adopt" if res[best] > 0.6023 + 0.0005 else
      "DO NOT ADOPT - no mode clears the unweighted baseline by more than noise")
json.dump(res, open('runs/exp28_din_history.json','w'), indent=2)
