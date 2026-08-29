"""D2Q-style duration-deconfounded target, then a multi-signal ensemble.

exp09 showed that regressing raw log play-time and subtracting a threshold ranks WORSE
than the binary label.  That is not a fair test of the watch-time direction: the published
approach (Kuaishou's duration-deconfounded quantile) predicts a row's watch-time QUANTILE
WITHIN ITS DURATION BUCKET, which is comparable across durations by construction and does
not ask the model to fit a heavy-tailed magnitude it does not need.

Then fuse every signal we have.  Ensemble weights are fitted on VALIDATION only.
"""
import sys, json, glob; sys.path.insert(0,'.')
import numpy as np
from kairos.kernel.dataset import Data
from kairos.kernel.frozenfeat import within_user_deviation
from kairos.kernel.fastmetrics import fast_evaluate, factorize
import lightgbm as lgb

d = Data(); fold = d.fold('official')
tr, va, te = fold.idx['train'], fold.idx['valid'], fold.idx['test']
ytr, yva = d.y_raw[tr], d.y_raw[va]
gva,_ = factorize(d.user_id[va])
X = np.load('runs/X_frozen.npy'); names = json.load(open('runs/X_frozen_names.json'))
hz = np.load('runs/X_frozen_hz.npy')
P = {}
for tag, idx in (('tr',tr),('va',va),('te',te)):
    dev,_ = within_user_deviation(X, names, d.user_id, idx, window_id=hz)
    P[tag] = np.concatenate([X[idx], dev],1)
ev = lambda s: fast_evaluate(gva, yva, s)

# ---- D2Q target: watch-time quantile within duration bucket (train only) ----
pt = d.col('play_time_ms').astype(np.float64)
du = d.col('duration_ms').astype(np.float64)
edges = np.quantile(du[tr], np.linspace(0,1,31)[1:-1])
db = np.searchsorted(edges, du)
q = np.zeros(len(tr))
for b in np.unique(db[tr]):
    m = db[tr]==b
    if m.sum()<2: continue
    v = pt[tr][m]
    r = np.argsort(np.argsort(v, kind='stable'), kind='stable')
    q[m] = r/(m.sum()-1)
print(f"D2Q target built over {len(np.unique(db[tr]))} duration buckets; "
      f"mean={q.mean():.3f} (uniform by construction)")

def fit(objective, ytg, rounds=350, seeds=(0,1,2)):
    vs, ts = [], []
    for sd in seeds:
        p = dict(objective=objective, learning_rate=0.05, num_leaves=63,
                 min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
                 bagging_freq=1, verbose=-1, seed=sd, num_threads=8,
                 metric=('auc' if objective=='binary' else 'l2'))
        box={'primary':-1}
        def cbe(env):
            if env.iteration % 25 and env.iteration != env.end_iteration-1: return
            m = ev(env.model.predict(P['va'], num_iteration=env.iteration+1))
            if m['primary']>box['primary']: box.update(m); box['iter']=env.iteration+1
        ds = lgb.Dataset(P['tr'], label=ytg)
        b = lgb.train(p, ds, num_boost_round=rounds, valid_sets=[ds],
                      callbacks=[cbe, lgb.log_evaluation(0)])
        vs.append(b.predict(P['va'], num_iteration=box['iter']))
        ts.append(b.predict(P['te'], num_iteration=box['iter']))
    return np.mean(vs,0), np.mean(ts,0)

print("\n=== D2Q target vs binary ===")
sig = {}
for tag, obj, ytg in (('gb_binary','binary', ytr.astype(float)),
                      ('gb_d2q','regression', q)):
    sv, st = fit(obj, ytg)
    sig[tag] = (sv, st)
    mv = ev(sv); mt = fold.scorers['test'].score(st, reason=f'exp10 {tag}')
    print(f"  {tag:12s} valid {mv['primary']:.4f} | test {mt['primary']:.4f} | "
          f"vs base {mt['primary']-0.5946:+.4f}")

# ---- assemble every signal ---------------------------------------------------
def wrank(s, users):
    s=np.asarray(s,dtype=np.float64)
    order = np.lexsort((np.arange(len(s)), -s, users)); u=users[order]
    st_=np.flatnonzero(np.r_[True,u[1:]!=u[:-1]]); sz=np.diff(np.r_[st_,len(u)])
    seg=np.repeat(np.arange(len(st_)),sz)
    pct=1.0-(np.arange(len(u))-st_[seg])/np.maximum(sz[seg]-1,1)
    o=np.empty(len(s)); o[order]=pct; return o
uva, ute = d.user_id[va], d.user_id[te]

fmk = json.load(open('runs/exp07a_fm.json'))
best_tau = max(set(k.split('_')[0] for k in fmk),
               key=lambda t: np.mean([fmk[k]['valid'] for k in fmk if k.startswith(t+'_')]))
sig['fm'] = (np.mean([wrank(np.load(f),uva) for f in sorted(glob.glob(f'runs/fm_{best_tau}_s*_va.npy'))],0),
             np.mean([wrank(np.load(f),ute) for f in sorted(glob.glob(f'runs/fm_{best_tau}_s*_te.npy'))],0))
cnt = np.bincount(d.video_id[tr], minlength=int(d.video_id.max())+1)
pos = np.bincount(d.video_id[tr], weights=d.y_raw[tr].astype(float), minlength=int(d.video_id.max())+1)
pr = (pos+20*d.y_raw[tr].mean())/(cnt+20)
sig['pop'] = (pr[d.video_id[va]], pr[d.video_id[te]])

R = {k: (wrank(v[0],uva), wrank(v[1],ute)) for k,v in sig.items()}
print("\n=== individual signals (validation) ===")
for k,(v,_) in R.items(): print(f"  {k:12s} {ev(v)['primary']:.4f}")

# greedy forward ensemble with weights on a 0.05 grid, selected on validation
keys = list(R); w = {k:0.0 for k in keys}
cur = None; best_p = -1
for _ in range(12):
    cand_best=None
    for k in keys:
        for dw in (0.05,0.1,0.2):
            w2 = dict(w); w2[k]+=dw
            tot=sum(w2.values())
            s = sum(w2[kk]*R[kk][0] for kk in keys)/tot
            p = ev(s)['primary']
            if p>best_p+1e-6 and (cand_best is None or p>cand_best[0]): cand_best=(p,k,dw)
    if cand_best is None: break
    best_p,k,dw = cand_best; w[k]+=dw
tot=sum(w.values()); w={k:v/tot for k,v in w.items()}
print(f"\nensemble weights (validation-selected): "
      f"{ {k:round(v,3) for k,v in w.items() if v>0} }")
sv_ens = sum(w[k]*R[k][0] for k in keys); st_ens = sum(w[k]*R[k][1] for k in keys)
mv=ev(sv_ens); mt=fold.scorers['test'].score(st_ens, reason='exp10 ensemble')
print(f"ENSEMBLE      valid {mv['primary']:.4f} | test {mt['primary']:.4f} | "
      f"gap {mv['primary']-mt['primary']:+.4f} | vs base {mt['primary']-0.5946:+.4f}")
json.dump({'weights':w,'valid':mv,'test':mt}, open('runs/exp10_ensemble.json','w'),
          indent=2, default=float)
np.save('runs/ens_va.npy', sv_ens); np.save('runs/ens_te.npy', st_ens)
