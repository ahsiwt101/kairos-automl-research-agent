"""Contract test: fast_evaluate must reproduce the official evaluate.evaluate() exactly.

Stress cases are chosen to hit every branch of the reference:
  ties (nDCG uses a STABLE sort, GAUC uses average ranks), all-positive and all-negative
  users (excluded from GAUC, pinned to nDCG 1.0 / 0.0), and singleton groups.
"""
import sys, time
import numpy as np
sys.path.insert(0, '.')
from evaluate import evaluate as ref_evaluate
from kairos.kernel.fastmetrics import fast_evaluate, factorize, per_group_metrics

rng = np.random.default_rng(0)
worst = 0.0
cases = []

def make(n_users, max_size, score_kind, pos_rate, name):
    uids, labels, scores = [], [], []
    for u in range(n_users):
        m = rng.integers(1, max_size + 1)
        for _ in range(m):
            uids.append(f"u{u}")
            labels.append(int(rng.random() < pos_rate))
        if score_kind == 'cont':
            scores.extend(rng.normal(size=m))
        elif score_kind == 'ties':                       # only 3 distinct values -> massive ties
            scores.extend(rng.choice([0.0, 1.0, 2.0], size=m))
        elif score_kind == 'const':                      # every score identical
            scores.extend(np.zeros(m))
    return name, np.array(uids), np.array(labels), np.array(scores, dtype=float)

cases.append(make(400, 8, 'cont',  0.35, 'continuous scores'))
cases.append(make(400, 8, 'ties',  0.35, 'heavy ties (3 distinct)'))
cases.append(make(400, 6, 'const', 0.35, 'all scores identical'))
cases.append(make(300, 3, 'cont',  0.05, 'sparse positives (mostly all-neg users)'))
cases.append(make(300, 3, 'cont',  0.95, 'dense positives (mostly all-pos users)'))
cases.append(make(500, 1, 'cont',  0.50, 'singleton groups only'))
cases.append(make(200, 12, 'ties', 0.50, 'large groups + ties'))

print(f"{'case':<42} {'ref primary':>12} {'fast primary':>13} {'max abs diff':>13}")
for name, uids, labels, scores in cases:
    ref = ref_evaluate(list(uids), list(labels), list(scores))
    gid, _ = factorize(uids)
    fast = fast_evaluate(gid, labels, scores)
    d = max(abs(ref[k] - fast[k]) for k in ('GAUC', 'nDCG@5', 'primary'))
    worst = max(worst, d)
    assert ref['users'] == fast['users'], f"{name}: user count mismatch"
    print(f"{name:<42} {ref['primary']:>12.9f} {fast['primary']:>13.9f} {d:>13.2e}")

print(f"\nworst deviation across all cases: {worst:.3e}")
assert worst < 1e-12, f"FAST PATH IS NOT EXACT (worst={worst:.3e}) - do not use it"
print("PASS: fast_evaluate is numerically identical to the official evaluate.py")

# --- per_group_metrics must aggregate back to the same scalars -------------
name, uids, labels, scores = cases[1]
gid, _ = factorize(uids)
pg = per_group_metrics(gid, labels, scores)
v = pg['valid_for_gauc']
gauc_re = (pg['npos'][v] * pg['auc'][v]).sum() / pg['npos'][v].sum()
ndcg_re = pg['ndcg'].mean()
fast = fast_evaluate(gid, labels, scores)
assert abs(gauc_re - fast['GAUC']) < 1e-12 and abs(ndcg_re - fast['nDCG@5']) < 1e-12
print("PASS: per_group_metrics re-aggregates to the same GAUC / nDCG")

# --- speed ---------------------------------------------------------------
N, U = 170_588, 23_875
uids = rng.integers(0, U, size=N)
labels = (rng.random(N) < 0.3).astype(int)
scores = rng.normal(size=N)
t0 = time.time(); ref_evaluate(list(uids), list(labels), list(scores)); t_ref = time.time() - t0
gid, _ = factorize(uids)
t0 = time.time()
for _ in range(20): fast_evaluate(gid, labels, scores)
t_fast = (time.time() - t0) / 20
print(f"\ntest-sized eval: reference {t_ref*1000:7.1f} ms | fast {t_fast*1000:6.1f} ms "
      f"| speedup {t_ref/t_fast:.0f}x")
