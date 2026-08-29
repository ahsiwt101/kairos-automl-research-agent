"""Contract tests for the causal layer. These are the tests that catch silent leakage."""
import sys; sys.path.insert(0,'.')
import numpy as np
from kairos.kernel.causal import causal_prefix

# --- 1. hand-checkable tiny case -----------------------------------------
keys   = np.array([1,1,1,2,2,1])
time   = np.array([10,20,30,10,20,40])
y      = np.array([1,0,1,1,1,1], dtype=float)
lab    = np.array([1,1,1,1,1,0], dtype=bool)   # last row is past the horizon
n,l,p = causal_prefix(keys, time, y, lab)
assert list(n) == [0,1,2,0,1,3], n     # exposures before, within key
assert list(l) == [0,1,2,0,1,3], l     # all earlier rows are labeled
assert list(p) == [0,1,1,0,1,2], p     # positives before: k1 -> [], [1], [1,0], [1,0,1]
print("PASS: prefix counts exclude self and respect key/time ordering")

# --- 2. horizon: a row past the horizon must contribute exposure, never a label ---
keys = np.array([1,1,1]); time = np.array([1,2,3])
y    = np.array([1,1,1], dtype=float); lab = np.array([True, False, True])
n,l,p = causal_prefix(keys,time,y,lab)
assert list(n)==[0,1,2] and list(l)==[0,1,1] and list(p)==[0,1,1]
print("PASS: post-horizon rows add exposure but never leak their label")

# --- 3. randomized: brute-force reference ---------------------------------
rng = np.random.default_rng(0)
N = 4000
keys = rng.integers(0, 50, N); time = rng.integers(0, 500, N)
y = (rng.random(N) < .3).astype(float); lab = rng.random(N) < .8
n,l,p = causal_prefix(keys, time, y, lab)
tb = np.arange(N)
bad = 0
for i in rng.choice(N, 300, replace=False):
    earlier = (keys == keys[i]) & ((time < time[i]) | ((time == time[i]) & (tb < i)))
    if n[i] != earlier.sum() or p[i] != (y[earlier]*lab[earlier]).sum() \
       or l[i] != lab[earlier].sum():
        bad += 1
assert bad == 0, f"{bad} mismatches vs brute force"
print("PASS: matches brute-force reference on 300 random probes")

# --- 4. THE test that matters: a causal feature must not predict its own label ----
# Build item-rate features on shuffled labels; a leak-free builder yields a feature that
# is uninformative about the row's own label.
from scipy.stats import pearsonr
N = 20000
keys = rng.integers(0, 200, N); time = np.arange(N); rng.shuffle(time)
y = (rng.random(N) < .3).astype(float); lab = np.ones(N, bool)
n,l,p = causal_prefix(keys, time, y, lab)
rate = (p + 20*.3) / (l + 20)
r, _ = pearsonr(rate, y)
assert abs(r) < 0.05, f"causal feature correlates with own label (r={r:.3f}) - LEAK"
print(f"PASS: no self-correlation on random labels (r={r:+.4f})")
