"""Does the measured decorrelation actually convert to score?

exp26 confirmed the premise: expert pairs sit at mean Spearman +0.362 against FM/DIN's
+0.848. That is necessary but not sufficient - weak-and-independent only beats
strong-and-redundant if the independence carries signal rather than noise. Test it directly,
including the parameterised power-rank fusion (item C of the plan).
"""
import sys, json; sys.path.insert(0,'.')
import numpy as np
from kairos.kernel.dataset import Data
from kairos.kernel.fastmetrics import fast_evaluate, factorize

d = Data(); fold = d.fold('official')
va, te = fold.idx['valid'], fold.idx['test']
g,_ = factorize(d.user_id[va]); yva = d.y_raw[va]
uva, ute = d.user_id[va], d.user_id[te]
ev = lambda s: fast_evaluate(g, yva, s)['primary']

def wrank(s, users):
    s = np.asarray(s, float)
    o = np.lexsort((np.arange(len(s)), -s, users)); u = users[o]
    st = np.flatnonzero(np.r_[True, u[1:]!=u[:-1]]); sz = np.diff(np.r_[st, len(u)])
    seg = np.repeat(np.arange(len(st)), sz)
    p = 1.0 - (np.arange(len(u)) - st[seg]) / np.maximum(sz[seg]-1, 1)
    r = np.empty(len(s)); r[o] = p; return r

S = {}
def add(nm, arr):
    S[nm] = (wrank(arr[va], uva), wrank(arr[te], ute))
add('refit', np.load('runs/refit_cache/official_tau14.npy'))
add('din',   np.load('runs/din_cache/official_k32.npy'))
for sub in ('context','item','user'):
    add(f'x_{sub}', np.load(f'runs/expert_cache/official_{sub}.npy'))
add('cf', np.load('runs/cf_cache_official/score.npy'))

print(f"{'member':<12} {'valid':>8}")
for k,(v,_) in S.items(): print(f"  {k:<10} {ev(v):>8.4f}")

keys = list(S)
def blend(w, gam, which):
    tot = sum(w.values())
    if tot <= 0: return np.zeros(len(S[keys[0]][which]))
    out = 0.0
    for k in keys:
        if w[k] <= 0: continue
        r = np.clip(S[k][which], 1e-6, 1.0)
        out = out + w[k] * (r ** gam[k])
    return out / tot

def ascend(gam, rounds=40):
    w = {k: 0.0 for k in keys}
    w[max(keys, key=lambda k: ev(S[k][0]))] = 1.0
    best = ev(blend(w, gam, 0))
    for _ in range(rounds):
        moved = False
        for k in keys:
            for dw in (0.4, 0.2, 0.1, -0.1, -0.2):
                w2 = dict(w); w2[k] = max(0.0, w2[k] + dw)
                if sum(w2.values()) <= 0: continue
                p = ev(blend(w2, gam, 0))
                if p > best + 1e-6: best, w, moved = p, w2, True
        if not moved: break
    return w, best

print(f"\n{'fusion scheme':<34} {'valid':>8} {'test':>8} {'gap':>8}")
print("-"*62)
res = {}

# linear (gamma = 1 everywhere) - what we currently ship
gam1 = {k: 1.0 for k in keys}
w_lin, v_lin = ascend(gam1)
# power-rank: sweep a shared gamma, then coordinate-ascend weights at the best one
best_g, best_v, best_w = 1.0, v_lin, w_lin
for gval in (0.5, 0.75, 1.5, 2.0):
    gam = {k: gval for k in keys}
    w_g, v_g = ascend(gam, rounds=25)
    if v_g > best_v: best_g, best_v, best_w = gval, v_g, w_g
gam_best = {k: best_g for k in keys}

for nm, w, gam in (('linear rank fusion', w_lin, gam1),
                   (f'power-rank fusion (gamma={best_g:g})', best_w, gam_best)):
    sv, st = blend(w, gam, 0), blend(w, gam, 1)
    mv = ev(sv); mt = fold.scorers['test'].score(st, reason=f'exp27 {nm}')['primary']
    res[nm] = {'valid': mv, 'test': mt,
               'weights': {k: round(w[k]/sum(w.values()),3) for k in keys if w[k] > 0}}
    print(f"{nm:<34} {mv:>8.4f} {mt:>8.4f} {mv-mt:>+8.4f}")
    print(f"   {res[nm]['weights']}")

print(f"\nreference: baseline 0.5946 | agent submission 0.5988")
json.dump(res, open('runs/exp27_expert_fusion.json','w'), indent=2, default=float)
