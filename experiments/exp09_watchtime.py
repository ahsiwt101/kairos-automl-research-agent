"""Does modelling WATCH TIME beat modelling the binary label?

long_view is (empirically, see below) a duration-dependent threshold on watch ratio, and a
perfect watch-ratio predictor would score 0.8023 primary against the label oracle's 0.8484.
The binary label therefore discards magnitude information on every single row.

Controlled: identical features, identical model family, identical splits. Only the TARGET
changes. play_time_ms is used ONLY as a training target on train rows; it is never a feature.

Scoring: ranking by predicted watch ratio directly would favour short videos, which have
systematically higher ratios. The rankable quantity is the MARGIN of predicted play time
against that video's own long_view threshold T(duration).
"""
import sys, json; sys.path.insert(0,'.')
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

pt = d.col('play_time_ms').astype(np.float64)
du = d.col('duration_ms').astype(np.float64)
edges = np.quantile(du[tr], np.linspace(0,1,21)[1:-1])
db = np.searchsorted(edges, du)

# --- estimate the long_view threshold T(duration bucket) from TRAIN only -----
T = np.zeros(len(edges)+1)
for b in range(len(edges)+1):
    m = (db[tr]==b)
    if m.sum() < 100: T[b] = np.median(pt[tr][m]) if m.sum() else 1.0; continue
    p0 = pt[tr][m & (ytr==0)]; p1 = pt[tr][m & (ytr==1)]
    if len(p0)<20 or len(p1)<20: T[b] = np.median(pt[tr][m]); continue
    T[b] = 0.5*(np.percentile(p0,95) + np.percentile(p1,5))
print("threshold T(dur) estimated on train, 20 duration buckets")

logT = np.log1p(np.maximum(T[db], 1.0))
target_reg = np.log1p(np.maximum(pt, 0.0))          # dense, continuous
ev = lambda s: fast_evaluate(gva, yva, s)

def fit(objective, y_train, tag, rounds=400):
    res_v, res_t = [], []
    for sd in (0,1,2):
        p = dict(objective=objective, learning_rate=0.05, num_leaves=63,
                 min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
                 bagging_freq=1, verbose=-1, seed=sd, num_threads=8)
        if objective=='binary': p['metric']='auc'
        else: p['metric']='l2'
        box={'primary':-1}
        def cbe(env):
            if env.iteration % 20 and env.iteration != env.end_iteration-1: return
            raw = env.model.predict(P['va'], num_iteration=env.iteration+1)
            s = raw - logT[va] if objective!='binary' else raw
            m = ev(s)
            if m['primary']>box['primary']: box.update(m); box['iter']=env.iteration+1
        ds = lgb.Dataset(P['tr'], label=y_train)
        bst = lgb.train(p, ds, num_boost_round=rounds, valid_sets=[ds],
                        callbacks=[cbe, lgb.log_evaluation(0)])
        rv = bst.predict(P['va'], num_iteration=box['iter'])
        rt = bst.predict(P['te'], num_iteration=box['iter'])
        res_v.append(rv - logT[va] if objective!='binary' else rv)
        res_t.append(rt - logT[te] if objective!='binary' else rt)
    return np.mean(res_v,0), np.mean(res_t,0)

print("\n=== target ablation (identical features, only the target changes) ===")
out={}
for tag, obj, ytg in (('binary long_view','binary', ytr.astype(float)),
                      ('regress log play_time','regression', target_reg[tr]),
                      ('huber log play_time','huber', target_reg[tr])):
    sv, st = fit(obj, ytg, tag)
    mv = ev(sv); mt = fold.scorers['test'].score(st, reason=f'exp09 {tag}')
    out[tag]={'valid':mv,'test':mt}
    print(f"  {tag:24s} valid {mv['primary']:.4f} | test {mt['primary']:.4f} | "
          f"gap {mv['primary']-mt['primary']:+.4f} | vs base {mt['primary']-0.5946:+.4f}")
    np.save(f"runs/wt_{tag.split()[0]}_va.npy", sv); np.save(f"runs/wt_{tag.split()[0]}_te.npy", st)
json.dump(out, open('runs/exp09_watchtime.json','w'), indent=2, default=float)
