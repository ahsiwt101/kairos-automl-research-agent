"""Stage B (lightgbm only): GBDT on window-frozen features, window-grouped deviations."""
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
parts = {}
for tag, idx in (('tr',tr),('va',va),('te',te)):
    dev,dn = within_user_deviation(X, names, d.user_id, idx, window_id=hz)
    parts[tag] = np.concatenate([X[idx], dev],1)
allnames = names + [f'dev_{n}' for n in names]
print(f"features: {parts['tr'].shape[1]}")

out={}
for seed in (0,1,2):
    params = dict(objective='binary', metric='auc', learning_rate=0.05, num_leaves=63,
                  min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, verbose=-1, seed=seed, num_threads=8)
    box={'primary':-1}
    def cbe(env):
        if env.iteration % 20 and env.iteration != env.end_iteration-1: return
        m = fast_evaluate(gva, yva, env.model.predict(parts['va'], num_iteration=env.iteration+1))
        if m['primary']>box['primary']: box.update(m); box['iter']=env.iteration+1
    bst = lgb.train(params, lgb.Dataset(parts['tr'], label=ytr), num_boost_round=400,
                    valid_sets=[lgb.Dataset(parts['tr'], label=ytr)],
                    callbacks=[cbe, lgb.log_evaluation(0)])
    np.save(f'runs/gb_s{seed}_va.npy', bst.predict(parts['va'], num_iteration=box['iter']))
    np.save(f'runs/gb_s{seed}_te.npy', bst.predict(parts['te'], num_iteration=box['iter']))
    out[f's{seed}']={'valid':box['primary'],'iter':box['iter']}
    print(f"  seed {seed}: valid {box['primary']:.4f} (iter {box['iter']})")
    if seed==0:
        json.dump(sorted(zip(allnames, bst.feature_importance('gain').tolist()),
                         key=lambda x:-x[1])[:25], open('runs/gb_importance.json','w'), indent=2)
json.dump(out, open('runs/exp07b_gbdt.json','w'), indent=2)
