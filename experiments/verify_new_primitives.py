"""Leakage probes for the Phase 2 primitives, per the plan's verification requirement:
any new ctx primitive needs a structural check before the agent may use it.

mf_factors' U (user factor) is, by construction, identical for every row of the same user
within one frozen window - the official valid/test splits are each a single window, so U
must have EXACTLY ZERO within-user variance there, the same structural test used
throughout this project for user-level statistics. V (item factor), auxiliary_signal, and
cf_score are all item/row-varying by design and are NOT expected to be user-constant.
"""
import sys; sys.path.insert(0,'.')
import numpy as np
from kairos.kernel.dataset import Data
from kairos.kernel.mf_signal import build_mf_factors
from kairos.kernel.baseline_signal import build_auxiliary_signal
from kairos.kernel.cf_signal import build_cf_score
from kairos.kernel.frozenfeat import OFFICIAL_WINDOWS, within_user_deviation

d = Data(); fold = d.fold('official')

def max_within_user_var(col, idx):
    dev, _ = within_user_deviation(col.reshape(-1, 1), ['x'], d.user_id, idx)
    return float(np.abs(dev).max())

print("=== mf_factors: U must be EXACTLY user-constant within a window ===")
U, V = build_mf_factors(d, OFFICIAL_WINDOWS, dim=16)
for split in ('valid', 'test'):
    idx = fold.idx[split]
    mx = max(max_within_user_var(U[:, j], idx) for j in range(U.shape[1]))
    print(f"  {split:6s} max|U - user_mean(U)| across all 16 dims = {mx:.3e}   "
          f"{'OK' if mx < 1e-6 else 'LEAK'}")
    assert mx < 1e-6, f"mf_factors leaks: U varies within a user's {split} list"
print("  (V is item-varying by design - not checked for user-constancy)")

print("\n=== auxiliary_signal / cf_score / baseline_score: no future-label use ===")
# structural: every one of these is built by build_*_signal / build_cf_score, which all
# restrict their fit set to `date <= hz` where hz < the window's own start date (enforced
# by window_horizons / windows_for_fold, already covered by verify_frozen.py). Confirm here
# that none of the OFFICIAL windows' horizons reach into that window's own date range.
for lo, hi, hz in OFFICIAL_WINDOWS:
    assert hz < lo, f"window {lo}-{hi} horizon {hz} is not strictly before its own start"
print("  PASS: every OFFICIAL_WINDOWS horizon strictly precedes its own window's start")

print("\n=== auxiliary_signal never reachable for blocked outcome columns ===")
from kairos.agent.context import make_context
ctx = make_context()
for bad in ('play_time_ms', 'long_view', 'profile_stay_time'):
    try:
        ctx.col(bad); print(f"  {bad}: NOT BLOCKED"); assert False
    except ValueError:
        print(f"  {bad}: ctx.col() blocked (correct)")
    try:
        ctx.auxiliary_signal(bad); print(f"  {bad}: auxiliary_signal NOT BLOCKED"); assert False
    except ValueError:
        print(f"  {bad}: ctx.auxiliary_signal() blocked (correct)")

print("\nALL NEW-PRIMITIVE LEAKAGE PROBES PASS")
