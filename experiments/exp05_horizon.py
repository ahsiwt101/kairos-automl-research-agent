"""Is validation lying to us, and does horizon-matching fix it?

Official setup: test rows (Apr 29 - May 8) may only use labels through Apr 28, so their
history is 1-10 days stale.  Validation rows (Apr 22-28) at the same horizon have history
right up to themselves - staleness 0.  A model selected on that validation set is being
scored on a strictly easier feature distribution than it will face.

FIX: build validation features at horizon = end of TRAIN (Apr 21), giving validation rows
1-7 days of staleness, structurally matching test.  If the hypothesis is right, the
validation score should FALL toward the test score and the val->test gap should collapse.
"""
import sys, time, json; sys.path.insert(0,'.')
import numpy as np
from kairos.kernel.dataset import Data
from kairos.kernel.featmat import build_matrix, add_cf_columns, within_user_deviation
from kairos.kernel.fastmetrics import fast_evaluate, factorize
import lightgbm as lgb

d = Data(); fold = d.fold('official')
tr, va, te = fold.idx['train'], fold.idx['valid'], fold.idx['test']
ytr, yva, = d.y_raw[tr], d.y_raw[va]
gva,_ = factorize(d.user_id[va])

def make(horizon, tag):
    t0=time.time()
    X, names, _ = build_matrix(d, fold, horizon=horizon)
    X, names = add_cf_columns(X, names, d, fold, horizon=horizon)
    dev, dn = within_user_deviation(X, names, d.user_id)
    X = np.concatenate([X, dev],1); names = names+dn
    print(f"  built {tag} (horizon={horizon}) {X.shape} in {time.time()-t0:.0f}s")
    return X, names

# leaky-for-validation matrix (what a naive pipeline builds) and the horizon-matched one
X_fresh, names = make(fold.horizon, 'fresh   ')   # horizon 20220428
X_stale, _     = make(20220421,     'matched ')   # horizon = end of train
X_serve, _     = make(20220428,     'serving ')   # what test genuinely gets

def run(Xtrain, Xvalid, Xtest, tag):
    params = dict(objective='binary', metric='auc', learning_rate=0.05, num_leaves=63,
                  min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, verbose=-1, seed=0, num_threads=8)
    dtr = lgb.Dataset(Xtrain[tr], label=ytr)
    best={'primary':-1}
    def cbe(env):
        global best
        if env.iteration % 25 and env.iteration != env.end_iteration-1: return
        m = fast_evaluate(gva, yva, env.model.predict(Xvalid[va], num_iteration=env.iteration+1))
        if m['primary']>best['primary']: best=dict(m); best['iter']=env.iteration+1
    bst = lgb.train(params, dtr, num_boost_round=600, valid_sets=[dtr],
                    callbacks=[cbe, lgb.log_evaluation(0)])
    st = bst.predict(Xtest[te], num_iteration=best['iter'])
    mt = fold.scorers['test'].score(st, reason=f'exp05 {tag}')
    print(f"{tag:34s} iter {best['iter']:4d} | valid {best['primary']:.4f} | "
          f"test {mt['primary']:.4f} | gap {best['primary']-mt['primary']:+.4f}")
    return best, mt

print(f"\n{'configuration':<34} {'':>9} {'valid':>6} {'test':>10} {'gap':>8}")
print("-"*72)
run(X_fresh, X_fresh, X_serve, 'A naive (fresh valid features)')
run(X_stale, X_stale, X_serve, 'B horizon-matched validation')
print("\nreference: FM baseline valid 0.6016 test 0.5946 (gap +0.0070)")
np.save('runs/featmat_stale.npy', X_stale); np.save('runs/featmat_serve.npy', X_serve)
json.dump(names, open('runs/featmat_names.json','w'))
