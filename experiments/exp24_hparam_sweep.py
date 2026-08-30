"""Sweep the LightGBM hyperparameters. Every experiment this project has run used ONE
untuned config (lr 0.05 / 63 leaves / min_data 200 / ff 0.8 / bag 0.8). We built the
tuning capability into the agent harness and then never used it ourselves.

Random search rather than grid: at 6 dimensions a grid wastes most of its budget on
dimensions that do not matter, and random search finds the important ones faster.
Selection on validation; winner confirmed on a backtest fold before it is believed.
"""
import sys, json, time; sys.path.insert(0,'.')
import numpy as np
from kairos.kernel.dataset import Data
from kairos.kernel.frozenfeat import within_user_deviation
from kairos.kernel.fastmetrics import fast_evaluate, factorize
import lightgbm as lgb

d = Data(); fold = d.fold('official')
tr, va = fold.idx['train'], fold.idx['valid']
g,_ = factorize(d.user_id[va]); yva = d.y_raw[va]; ytr = d.y_raw[tr]
X = np.load('runs/X_frozen.npy'); names = json.load(open('runs/X_frozen_names.json'))
hz = np.load('runs/X_frozen_hz.npy')
fm = np.load('runs/fm_signal.npy').reshape(-1,1)
Xf = np.concatenate([X, fm],1); nf = names+['fm_score']
P = {}
for t,i in (('tr',tr),('va',va)):
    dev,_ = within_user_deviation(Xf, nf, d.user_id, i, window_id=hz)
    P[t] = np.concatenate([Xf[i], dev],1)
print(f"feature matrix: {P['tr'].shape}")

def run(hp, seeds=(0,1)):
    out=[]
    for sd in seeds:
        p = dict(objective='binary', metric='auc', verbose=-1, seed=sd, num_threads=7, **hp)
        box={'p':-1,'it':0}
        def cbe(env):
            if env.iteration % 25 and env.iteration != env.end_iteration-1: return
            m = fast_evaluate(g, yva, env.model.predict(P['va'], num_iteration=env.iteration+1))
            if m['primary']>box['p']: box['p']=m['primary']; box['it']=env.iteration+1
        ds = lgb.Dataset(P['tr'], label=ytr)
        lgb.train(p, ds, num_boost_round=400, valid_sets=[ds],
                  callbacks=[cbe, lgb.log_evaluation(0)])
        out.append(box['p'])
    return float(np.mean(out)), float(np.std(out))

DEFAULT = dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
               feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1)
b, bs = run(DEFAULT)
print(f"\n{'config':<64} {'valid':>8} {'std':>7}")
print("-"*82)
print(f"{'DEFAULT (used in every experiment so far)':<64} {b:>8.4f} {bs:>7.4f}")

rng = np.random.default_rng(0)
best, best_hp = b, DEFAULT
t0 = time.time()
for i in range(14):
    hp = dict(
        learning_rate=float(np.round(rng.choice([0.02,0.03,0.05,0.08,0.12]),3)),
        num_leaves=int(rng.choice([15,31,63,127,255])),
        min_data_in_leaf=int(rng.choice([50,100,200,500,1000])),
        feature_fraction=float(np.round(rng.uniform(0.4,1.0),2)),
        bagging_fraction=float(np.round(rng.uniform(0.5,1.0),2)),
        bagging_freq=1,
        lambda_l2=float(np.round(rng.choice([0.0,1.0,5.0]),2)))
    m,s = run(hp)
    flag = ' <-- best' if m > best else ''
    if m > best: best, best_hp = m, hp
    desc = f"lr{hp['learning_rate']} lv{hp['num_leaves']} md{hp['min_data_in_leaf']} " \
           f"ff{hp['feature_fraction']} bg{hp['bagging_fraction']} l2={hp['lambda_l2']}"
    print(f"{desc:<64} {m:>8.4f} {s:>7.4f}{flag}", flush=True)
print(f"\nswept 14 configs in {time.time()-t0:.0f}s")
print(f"best {best:.4f} vs default {b:.4f}  ({best-b:+.4f})")
print(f"best config: {best_hp}")
json.dump({'default':{'valid':b}, 'best':{'valid':best,'hp':best_hp}},
          open('runs/exp24_hparam_sweep.json','w'), indent=2, default=float)
