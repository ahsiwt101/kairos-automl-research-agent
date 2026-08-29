"""Correct multi-signal ensemble.

Rank handling matters and is easy to get wrong (exp10 did): convert EACH SEED's raw
prediction to within-user percentile ranks, average those per model, and blend across
models.  Re-ranking an already-averaged rank vector throws away exactly the information
seed-averaging created, and quietly degrades the blend.

Weights are fitted on validation by coordinate ascent, then applied unchanged to test.
"""
import sys, json, glob; sys.path.insert(0,'.')
import numpy as np
from kairos.kernel.dataset import Data
from kairos.kernel.fastmetrics import fast_evaluate, factorize

d = Data(); fold = d.fold('official')
va, te = fold.idx['valid'], fold.idx['test']
gva,_ = factorize(d.user_id[va]); yva = d.y_raw[va]
uva, ute = d.user_id[va], d.user_id[te]
ev = lambda s: fast_evaluate(gva, yva, s)['primary']

def wrank(s, users):
    s = np.asarray(s, float)
    o = np.lexsort((np.arange(len(s)), -s, users)); u = users[o]
    st = np.flatnonzero(np.r_[True, u[1:] != u[:-1]]); sz = np.diff(np.r_[st, len(u)])
    seg = np.repeat(np.arange(len(st)), sz)
    p = 1.0 - (np.arange(len(u)) - st[seg]) / np.maximum(sz[seg]-1, 1)
    r = np.empty(len(s)); r[o] = p; return r

def seed_avg(pattern_va, pattern_te):
    fv = sorted(glob.glob(pattern_va)); ft = sorted(glob.glob(pattern_te))
    if not fv: return None
    return (np.mean([wrank(np.load(f), uva) for f in fv], 0),
            np.mean([wrank(np.load(f), ute) for f in ft], 0))

S = {}
for tau in ('None','14','7','5'):
    r = seed_avg(f'runs/fm_tau{tau}_s*_va.npy', f'runs/fm_tau{tau}_s*_te.npy')
    if r: S[f'fm_tau{tau}'] = r
r = seed_avg('runs/gb_s*_va.npy', 'runs/gb_s*_te.npy')
if r: S['gb_frozen'] = r
for nm in ('binary','regress','huber'):
    try:
        S[f'wt_{nm}'] = (wrank(np.load(f'runs/wt_{nm}_va.npy'), uva),
                         wrank(np.load(f'runs/wt_{nm}_te.npy'), ute))
    except Exception: pass
tr = fold.idx['train']
cnt = np.bincount(d.video_id[tr], minlength=int(d.video_id.max())+1)
pos = np.bincount(d.video_id[tr], weights=d.y_raw[tr].astype(float),
                  minlength=int(d.video_id.max())+1)
pr = (pos + 20*d.y_raw[tr].mean())/(cnt + 20)
S['item_pop'] = (wrank(pr[d.video_id[va]], uva), wrank(pr[d.video_id[te]], ute))

print(f"{'signal':<14} {'valid':>8}")
for k,(v,_) in sorted(S.items(), key=lambda kv: -ev(kv[1][0])):
    print(f"  {k:<12} {ev(v):>8.4f}")

keys = list(S)
w = np.zeros(len(keys)); w[int(np.argmax([ev(S[k][0]) for k in keys]))] = 1.0
def score(wv, which=0):
    t = wv.sum()
    return sum(wv[i]*S[k][which] for i,k in enumerate(keys)) / (t if t else 1.0)
best = ev(score(w))
print(f"\nstart (best single): {best:.4f}")
for it in range(60):
    improved = False
    for i in range(len(keys)):
        for dw in (0.4, 0.2, 0.1, -0.1, -0.2):
            w2 = w.copy(); w2[i] = max(0.0, w2[i] + dw)
            if w2.sum() <= 0: continue
            p = ev(score(w2))
            if p > best + 1e-6:
                best, w, improved = p, w2, True
    if not improved: break
w = w / w.sum()
print(f"coordinate ascent -> valid {best:.4f}")
print("weights:", {keys[i]: round(float(w[i]),3) for i in range(len(keys)) if w[i] > 1e-6})

sv, st = score(w,0), score(w,1)
mt = fold.scorers['test'].score(st, reason='exp13 ensemble')
mv = fast_evaluate(gva, yva, sv)
print(f"\nENSEMBLE  valid {mv['primary']:.4f} (GAUC {mv['GAUC']:.4f} nDCG {mv['nDCG@5']:.4f})")
print(f"          test  {mt['primary']:.4f} (GAUC {mt['GAUC']:.4f} nDCG {mt['nDCG@5']:.4f})")
print(f"          gap {mv['primary']-mt['primary']:+.4f} | vs baseline {mt['primary']-0.5946:+.4f}")
np.save('runs/final_va.npy', sv); np.save('runs/final_te.npy', st)
json.dump({'weights':{keys[i]:float(w[i]) for i in range(len(keys))},
           'valid':mv,'test':mt}, open('runs/exp13_ensemble.json','w'), indent=2, default=float)
