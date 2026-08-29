"""Do causal behaviour features contain signal the FM's ID embeddings cannot reach?

Compares, on identical splits:
  FM (5 ID fields, official baseline)  vs  LightGBM on causal history + CF features,
with both a binary objective and lambdarank, so 'features' and 'objective' stay separable.
"""
import sys, time, json; sys.path.insert(0,'.')
import numpy as np
sys.path.insert(0,'.')
from kairos.kernel.dataset import Data
from kairos.kernel.featmat import build_matrix, add_cf_columns, within_user_deviation
from kairos.kernel.fastmetrics import fast_evaluate, factorize
import lightgbm as lgb

d = Data(); fold = d.fold('official')
t0 = time.time()
X, names, cb = build_matrix(d, fold)
print(f"base causal matrix {X.shape} in {time.time()-t0:.0f}s")
t0 = time.time()
X, names = add_cf_columns(X, names, d, fold)
print(f"+ item-item CF -> {X.shape} in {time.time()-t0:.0f}s")
dev, dnames = within_user_deviation(X, names, d.user_id)
X = np.concatenate([X, dev], 1); names = names + dnames
print(f"+ within-user deviations -> {X.shape}")
np.save('runs/featmat_official.npy', X)
json.dump(names, open('runs/featmat_official_names.json','w'))

def groups_for(idx, key='user'):
    u = d.user_id[idx]
    if key == 'user_day':
        g = u.astype(np.int64)*100000 + d.date[idx] % 100000
    else:
        g = u.astype(np.int64)
    o = np.argsort(g, kind='stable')
    gs = g[o]
    sizes = np.diff(np.r_[np.flatnonzero(np.r_[True, gs[1:]!=gs[:-1]]), len(gs)])
    return o, sizes

tr, va, te = fold.idx['train'], fold.idx['valid'], fold.idx['test']
ytr, yva = d.y_raw[tr], d.y_raw[va]
gva,_ = factorize(d.user_id[va])

results = {}
for obj, gkey in [('binary', None), ('lambdarank','user_day'), ('lambdarank','user')]:
    t0 = time.time()
    if obj == 'binary':
        dtr = lgb.Dataset(X[tr], label=ytr)
        dva = lgb.Dataset(X[va], label=yva, reference=dtr)
        params = dict(objective='binary', metric='auc', learning_rate=0.05,
                      num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
                      bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=0,
                      num_threads=8)
    else:
        otr, str_ = groups_for(tr, gkey)
        ova, sva = groups_for(va, 'user')
        dtr = lgb.Dataset(X[tr][otr], label=ytr[otr], group=str_)
        dva = lgb.Dataset(X[va][ova], label=yva[ova], group=sva, reference=dtr)
        params = dict(objective='lambdarank', metric='ndcg', ndcg_eval_at=[5],
                      lambdarank_truncation_level=15, learning_rate=0.05,
                      num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
                      bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=0,
                      num_threads=8)
    best = {'primary': -1}
    def cb_eval(env):
        global best
        if env.iteration % 25 != 0 and env.iteration != env.end_iteration-1: return
        s = env.model.predict(X[va], num_iteration=env.iteration+1)
        m = fast_evaluate(gva, yva, s)
        if m['primary'] > best['primary']:
            best = dict(m); best['iter'] = env.iteration+1
    bst = lgb.train(params, dtr, num_boost_round=600, valid_sets=[dva],
                    callbacks=[cb_eval, lgb.log_evaluation(0)])
    tag = f"lgb-{obj}" + (f"/{gkey}" if gkey else "")
    st = bst.predict(X[te], num_iteration=best['iter'])
    mt = fold.scorers['test'].score(st, reason=f'exp04 {tag}')
    results[tag] = {'valid': best, 'test': mt, 'sec': round(time.time()-t0,1)}
    print(f"{tag:24s} iter {best['iter']:4d} | valid GAUC {best['GAUC']:.4f} "
          f"nDCG {best['nDCG@5']:.4f} primary {best['primary']:.4f} | "
          f"test primary {mt['primary']:.4f} | {time.time()-t0:.0f}s")

json.dump(results, open('runs/exp04_features.json','w'), indent=2, default=float)
print("\nbaseline FM: valid 0.6016 / test 0.5946")
imp = sorted(zip(names, bst.feature_importance('gain')), key=lambda x:-x[1])[:20]
print("\ntop features by gain:")
for n,g in imp: print(f"  {n:26s} {g:12.0f}")
