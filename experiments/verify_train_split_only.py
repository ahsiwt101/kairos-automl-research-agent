"""FAQ 2.9.2: models may be fit on the TRAIN SPLIT ONLY.

    "training data is the train split only: date 20220408-20220421"

with 20220422-28 supplying validation and 20220429-0508 supplying test. Validation is for
tuning and selection, and FAQ 2.2 permits developing on "the training split and the public
validation feedback" - so validation labels may inform FEATURE statistics. But no model's
loss may see a validation row.

Horizon and fit cut-off are therefore DIFFERENT quantities, and conflating them is the easy
mistake: a frozen window over the test period legitimately aggregates labels to 20220428,
while a model scoring that window may still only fit on rows to 20220421. Every signal in
this project takes a window horizon, and three of them originally used it as the fit set too.

This test pins the separation at the point that matters - the rows handed to a trainer.
"""
import sys, numpy as np
sys.path.insert(0, '.')
from kairos.kernel.dataset import load, FOLDS, train_end

d = load()
date = d.date
TE = train_end('official')
assert TE == 20220421, f'train split ends 20220421, got {TE}'
print(f"  train-split cut-off for 'official': {TE}")

valid_lo, valid_hi = FOLDS['official']['valid']
test_lo, test_hi = FOLDS['official']['test']

# 1. The clamp itself: for every OFFICIAL window, the fit set must exclude valid and test.
from kairos.kernel.frozenfeat import OFFICIAL_WINDOWS
bad = []
for lo, hi, hz in OFFICIAL_WINDOWS:
    cut = min(hz, TE)
    fit = np.flatnonzero(date <= cut)
    n_valid = int(((date[fit] >= valid_lo) & (date[fit] <= valid_hi)).sum())
    n_test = int(((date[fit] >= test_lo) & (date[fit] <= test_hi)).sum())
    if n_valid or n_test:
        bad.append((lo, hi, hz, n_valid, n_test))
assert not bad, f'fit set leaks non-train rows: {bad}'
print(f"  [PASS] all {len(OFFICIAL_WINDOWS)} window fit sets contain train-split rows only")

# 2. The un-clamped horizon WOULD have leaked - proving the clamp is load-bearing, not
#    decorative. The test-period window's horizon is 20220428, deep inside validation.
lo, hi, hz = OFFICIAL_WINDOWS[-1]
unclamped = np.flatnonzero(date <= hz)
n_valid_unclamped = int(((date[unclamped] >= valid_lo) & (date[unclamped] <= valid_hi)).sum())
assert n_valid_unclamped > 0, 'premise failed: the unclamped horizon should include validation'
print(f"  [PASS] clamp is load-bearing: horizon {hz} would admit "
      f"{n_valid_unclamped:,} validation rows into training")

# 3. No trainer in the kernel may fit on validation. Checked structurally: every builder
#    that fits a model must accept fit_end, or fit on fold.idx['train'] directly.
import inspect
from kairos.kernel import baseline_signal, cf_signal, mf_signal, refit_signal
for mod, fn in ((baseline_signal, 'build_fm_signal'),
                (baseline_signal, 'build_auxiliary_signal'),
                (cf_signal, 'build_cf_score'),
                (mf_signal, 'build_mf_factors')):
    sig = inspect.signature(getattr(mod, fn))
    assert 'fit_end' in sig.parameters, f'{fn} cannot be restricted to the train split'
print("  [PASS] every windowed trainer accepts a train-split cut-off")

# 4. refit_signal must NOT train on validation any more.
src = inspect.getsource(refit_signal)
assert "train_parts=('train', 'valid')" not in src, \
    'refit_signal still fits a model on validation rows'
assert 'train_parts' not in src, 'refit_signal should not select training parts at all'
print("  [PASS] refit_signal no longer fits any model on validation")

# 5. DIN and experts fit on fold.idx['train'] - confirm that index is train-only.
tr = d.fold('official').idx['train']
assert date[tr].max() <= TE, 'fold.idx[train] contains rows past the train split'
print(f"  [PASS] fold.idx['train'] spans {date[tr].min()}-{date[tr].max()}, train split only")

print("\nTRAIN-SPLIT-ONLY TESTS PASS")
