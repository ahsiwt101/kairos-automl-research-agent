"""GO/NO-GO for sub-space experts: does forcing disjoint feature families actually
decorrelate the models, or do they all just rediscover item quality?

Premise under test: rank fusion is rewarded by decorrelation, and our current members are
less decorrelated than their architectures suggest - FM and DIN sit at Spearman +0.848
despite being unrelated families. An expert that cannot SEE item identity should not be
able to rediscover item quality, so its errors ought to be differently distributed.

That is a hypothesis, not a fact. If the experts also land near +0.848 the premise fails
and there is no point building fusion on top of them. +0.848 is the number to beat.
"""
import sys, json; sys.path.insert(0,'.')
import numpy as np
from scipy.stats import spearmanr
from kairos.kernel.dataset import Data
from kairos.kernel.causal import window_horizons
from kairos.kernel.frozenfeat import windows_for_fold
from kairos.kernel.dataset import FOLDS
from kairos.kernel.expert_signal import build_expert_signal, SUBSPACES
from kairos.kernel.fastmetrics import fast_evaluate, factorize

d = Data(); fold = d.fold('official')
va = fold.idx['valid']
hz = window_horizons(d.date.astype(np.int64), windows_for_fold(FOLDS['official']))
g,_ = factorize(d.user_id[va]); y = d.y_raw[va]; u = d.user_id[va]

def wrank(s):
    s = np.asarray(s, float)
    o = np.lexsort((np.arange(len(s)), -s, u)); uu = u[o]
    st = np.flatnonzero(np.r_[True, uu[1:]!=uu[:-1]]); sz = np.diff(np.r_[st, len(uu)])
    seg = np.repeat(np.arange(len(st)), sz)
    p = 1.0 - (np.arange(len(uu)) - st[seg]) / np.maximum(sz[seg]-1, 1)
    r = np.empty(len(s)); r[o] = p; return r

print("=== individual expert strength (validation) ===")
sig = {}
for sub in SUBSPACES:
    s = build_expert_signal(d, fold, hz, sub)
    sig[f'x_{sub}'] = wrank(s[va])
    print(f"  expert[{sub}]  {fast_evaluate(g, y, s[va].astype(np.float64))['primary']:.4f}")

# existing members, for the comparison that decides go/no-go
sig['fm']  = wrank(np.load('runs/fm_signal.npy')[va])
sig['din'] = wrank(np.load('runs/din_cache/official_k32.npy')[va])
print(f"  (reference) fm   {fast_evaluate(g, y, np.load('runs/fm_signal.npy')[va].astype(np.float64))['primary']:.4f}")
print(f"  (reference) din  {fast_evaluate(g, y, np.load('runs/din_cache/official_k32.npy')[va].astype(np.float64))['primary']:.4f}")

print("\n=== pairwise Spearman (lower = better for fusion) ===")
ks = list(sig)
rows = []
for i in range(len(ks)):
    for j in range(i+1, len(ks)):
        r = spearmanr(sig[ks[i]], sig[ks[j]]).statistic
        rows.append((ks[i], ks[j], r))
for a, b, r in sorted(rows, key=lambda t: t[2]):
    mark = ''
    if a.startswith('x_') and b.startswith('x_'): mark = '  <- expert pair'
    if {a,b} == {'fm','din'}: mark = '  <- the number to beat'
    print(f"  {a:10s} vs {b:10s} {r:+.3f}{mark}")

exp_pairs = [r for a,b,r in rows if a.startswith('x_') and b.startswith('x_')]
fmdin = [r for a,b,r in rows if {a,b}=={'fm','din'}][0]
print(f"\nmean expert-pair correlation : {np.mean(exp_pairs):+.3f}")
print(f"FM/DIN correlation           : {fmdin:+.3f}")
verdict = "GO - experts are more decorrelated" if np.mean(exp_pairs) < fmdin else \
          "NO-GO - sub-spacing did not decorrelate; premise fails"
print(f"\nVERDICT: {verdict}")
json.dump({'expert_pairs': float(np.mean(exp_pairs)), 'fm_din': float(fmdin),
           'pairs': [[a,b,float(r)] for a,b,r in rows]},
          open('runs/exp26_subspace.json','w'), indent=2)
