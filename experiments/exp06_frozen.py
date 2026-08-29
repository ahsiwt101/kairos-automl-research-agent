"""The decisive comparison: naive causal features vs window-frozen features.

Includes the structural leakage assertion.  Within-user ranking is invariant to any
quantity that is constant across a user's list, so under a correct frozen construction a
USER-level statistic must have EXACTLY zero within-user variance.  If it does not, labels
from inside the evaluation window are bleeding into the features.
"""
import sys, time, json; sys.path.insert(0,'.')
import numpy as np
from kairos.kernel.dataset import Data
from kairos.kernel.frozenfeat import build_frozen_matrix, within_user_deviation
from kairos.kernel.fastmetrics import fast_evaluate, factorize
import lightgbm as lgb

d = Data(); fold = d.fold('official')
tr, va, te = fold.idx['train'], fold.idx['valid'], fold.idx['test']
ytr, yva = d.y_raw[tr], d.y_raw[va]
gva, _ = factorize(d.user_id[va])

t0 = time.time()
X, names, hz = build_frozen_matrix(d)
print(f"frozen matrix {X.shape} in {time.time()-t0:.0f}s")
np.save('runs/X_frozen.npy', X); json.dump(names, open('runs/X_frozen_names.json','w'))

# ---- structural leakage assertion --------------------------------------------
print("\n=== leakage probe: within-user variance of user-level statistics ===")
ui = names.index('user_rate')
for split, idx in (('valid', va), ('test', te)):
    dev, _ = within_user_deviation(X[:, [ui]], ['user_rate'], d.user_id, idx)
    mx = float(np.abs(dev).max())
    print(f"  {split:6s} max |user_rate - user_mean(user_rate)| = {mx:.3e}"
          f"   {'OK (no within-window feedback)' if mx < 1e-6 else 'LEAK'}")
    assert mx < 1e-6, "user-level statistic varies within a user's list -> label feedback"

dev_va, dn = within_user_deviation(X, names, d.user_id, va)
dev_te, _  = within_user_deviation(X, names, d.user_id, te)
dev_tr, _  = within_user_deviation(X, names, d.user_id, tr)
Xtr = np.concatenate([X[tr], dev_tr], 1)
Xva = np.concatenate([X[va], dev_va], 1)
Xte = np.concatenate([X[te], dev_te], 1)
allnames = names + dn
print(f"with deviations: {Xtr.shape[1]} features")

def run(obj, gkey=None, tag=''):
    params = dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  verbose=-1, seed=0, num_threads=8)
    if obj == 'binary':
        params.update(objective='binary', metric='auc')
        dtr = lgb.Dataset(Xtr, label=ytr)
    else:
        g = d.user_id[tr].astype(np.int64)
        if gkey == 'user_day': g = g*100000 + d.date[tr] % 100000
        o = np.argsort(g, kind='stable'); gs = g[o]
        sizes = np.diff(np.r_[np.flatnonzero(np.r_[True, gs[1:]!=gs[:-1]]), len(gs)])
        params.update(objective='lambdarank', metric='ndcg', ndcg_eval_at=[5],
                      lambdarank_truncation_level=15)
        dtr = lgb.Dataset(Xtr[o], label=ytr[o], group=sizes)
    box = {'primary': -1}
    def cbe(env):
        if env.iteration % 25 and env.iteration != env.end_iteration-1: return
        m = fast_evaluate(gva, yva, env.model.predict(Xva, num_iteration=env.iteration+1))
        if m['primary'] > box['primary']:
            box.update(m); box['iter'] = env.iteration+1
    bst = lgb.train(params, dtr, num_boost_round=700, valid_sets=[dtr],
                    callbacks=[cbe, lgb.log_evaluation(0)])
    st = bst.predict(Xte, num_iteration=box['iter'])
    mt = fold.scorers['test'].score(st, reason=f'exp06 {tag}')
    print(f"{tag:28s} iter {box['iter']:4d} | valid GAUC {box['GAUC']:.4f} nDCG "
          f"{box['nDCG@5']:.4f} primary {box['primary']:.4f} | test {mt['primary']:.4f} "
          f"| gap {box['primary']-mt['primary']:+.4f}")
    return bst, box, mt

print(f"\n{'model':<28} {'':>9} {'valid':>36} {'test':>8} {'gap':>9}")
b1,_,_ = run('binary', tag='frozen / binary')
b2,_,_ = run('lambdarank','user_day', tag='frozen / lambdarank user_day')
b3,_,_ = run('lambdarank','user', tag='frozen / lambdarank user')
print("\nreference   FM baseline        valid 0.6016 | test 0.5946 | gap +0.0070")
print("reference   NAIVE causal lgbrank valid 0.7158 | test 0.5749 | gap +0.1409")
print("\ntop features by gain (frozen / binary):")
for n,g in sorted(zip(allnames, b1.feature_importance('gain')), key=lambda x:-x[1])[:18]:
    print(f"  {n:26s} {g:12.0f}")
