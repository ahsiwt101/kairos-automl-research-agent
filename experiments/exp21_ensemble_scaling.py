"""Ensembling is the only intervention that has ever worked here, so scale it properly.

Across ~30 tested interventions this session, every modelling change landed inside seed
noise. Two things did not: seed ensembling (+0.0017) and selection discipline. That is a
finding about the benchmark, and it has a direct consequence - the remaining score is in
variance reduction, not in a better model.

Also tests the WEIGHTING scheme, which matters given our own result that validation is an
unreliable selection signal: coordinate ascent fits weights hard against validation and
may simply be overfitting it, where uniform weights cannot.
"""
import sys, json, glob, time; sys.path.insert(0,'.')
import numpy as np
from kairos.kernel.dataset import Data
from kairos.kernel.features import Encoder
from kairos.kernel.fastmetrics import fast_evaluate, factorize
from kairos.models.train import train_fm, predict

d = Data(); fold = d.fold('official')
tr, va, te = fold.idx['train'], fold.idx['valid'], fold.idx['test']
gva,_ = factorize(d.user_id[va]); yva = d.y_raw[va]
uva, ute = d.user_id[va], d.user_id[te]
ev = lambda s: fast_evaluate(gva, yva, s)['primary']

def wrank(s, users):
    s = np.asarray(s, float)
    o = np.lexsort((np.arange(len(s)), -s, users)); u = users[o]
    st = np.flatnonzero(np.r_[True, u[1:]!=u[:-1]]); sz = np.diff(np.r_[st, len(u)])
    seg = np.repeat(np.arange(len(st)), sz)
    p = 1.0 - (np.arange(len(u)) - st[seg]) / np.maximum(sz[seg]-1, 1)
    r = np.empty(len(s)); r[o] = p; return r

# --- more FM seeds: pure variance reduction on the strongest single model -------
enc = Encoder(d).fit(tr)
N_SEEDS = 10
print(f"training {N_SEEDS} FM seeds (tau=14, the validation-selected variant)...")
fm_va, fm_te = [], []
for sd in range(N_SEEDS):
    f = f'runs/fm10_s{sd}_va.npy'
    if not glob.glob(f):
        r = train_fm(fold, enc, loss='bce', seed=sd, recency_tau=14)
        np.save(f, predict(r['model'], enc, va))
        np.save(f.replace('_va','_te'), predict(r['model'], enc, te))
    fm_va.append(wrank(np.load(f), uva)); fm_te.append(wrank(np.load(f.replace('_va','_te')), ute))

print(f"\n{'FM seeds averaged':<22} {'valid':>8}")
for k in (1, 2, 3, 5, 10):
    print(f"  {k:>2d} seed(s){'':<11} {ev(np.mean(fm_va[:k],0)):>8.4f}")

# --- assemble every available signal -------------------------------------------
S = {'fm10': (np.mean(fm_va,0), np.mean(fm_te,0))}
for nm, pat in (('gb_frozen','runs/gb_s*_va.npy'), ('wt_binary','runs/wt_binary_va.npy')):
    fv = sorted(glob.glob(pat)); ft = [f.replace('_va','_te') for f in fv]
    if fv: S[nm] = (np.mean([wrank(np.load(f),uva) for f in fv],0),
                    np.mean([wrank(np.load(f),ute) for f in ft],0))
cnt = np.bincount(d.video_id[tr], minlength=int(d.video_id.max())+1)
pos = np.bincount(d.video_id[tr], weights=d.y_raw[tr].astype(float),
                  minlength=int(d.video_id.max())+1)
pr = (pos + 20*d.y_raw[tr].mean())/(cnt + 20)
S['item_pop'] = (wrank(pr[d.video_id[va]], uva), wrank(pr[d.video_id[te]], ute))
try:
    U,V = np.load('runs/mf_cache_official/U_d16.npy'), np.load('runs/mf_cache_official/V_d16.npy')
    mf = (U*V).sum(1); S['mf'] = (wrank(mf[va],uva), wrank(mf[te],ute))
    cf = np.load('runs/cf_cache_official/score.npy'); S['cf'] = (wrank(cf[va],uva), wrank(cf[te],ute))
except Exception as e: print('  (mf/cf caches unavailable:', e, ')')

print(f"\n{'signal':<12} {'valid':>8}")
for k,(v,_) in sorted(S.items(), key=lambda kv:-ev(kv[1][0])): print(f"  {k:<10} {ev(v):>8.4f}")

keys = list(S)
def blend(w, i): 
    t = sum(w.values());  return sum(w[k]*S[k][i] for k in keys)/(t if t else 1)

# scheme 1: uniform over signals that individually beat item-popularity
good = [k for k in keys if ev(S[k][0]) > ev(S['item_pop'][0])]
w_uni = {k: (1.0 if k in good else 0.0) for k in keys}
# scheme 2: coordinate ascent on validation (what we shipped)
w_ca = {k: 0.0 for k in keys}; w_ca[max(keys, key=lambda k: ev(S[k][0]))] = 1.0
best = ev(blend(w_ca,0))
for _ in range(40):
    improved = False
    for k in keys:
        for dw in (0.4,0.2,0.1,-0.1,-0.2):
            w2 = dict(w_ca); w2[k] = max(0.0, w2[k]+dw)
            if sum(w2.values()) <= 0: continue
            p = ev(blend(w2,0))
            if p > best + 1e-6: best, w_ca, improved = p, w2, True
    if not improved: break

print(f"\n{'weighting scheme':<34} {'valid':>8} {'test':>8} {'gap':>8}")
print("-"*62)
res = {}
for nm, w in (('uniform over good signals', w_uni), ('coordinate ascent on valid', w_ca)):
    sv, st = blend(w,0), blend(w,1)
    mv = ev(sv); mt = fold.scorers['test'].score(st, reason=f'exp21 {nm}')['primary']
    res[nm] = {'valid': mv, 'test': mt,
               'weights': {k: round(w[k]/sum(w.values()),3) for k in keys if w[k]>0}}
    print(f"{nm:<34} {mv:>8.4f} {mt:>8.4f} {mv-mt:>+8.4f}")
    print(f"   weights: {res[nm]['weights']}")
print(f"\nreference: FM baseline test 0.5946 | previous best ensemble test 0.5976")
json.dump(res, open('runs/exp21_ensemble_scaling.json','w'), indent=2, default=float)
