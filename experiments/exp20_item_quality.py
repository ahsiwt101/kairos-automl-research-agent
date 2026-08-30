"""The decomposition says item quality carries +0.0550 of the +0.0595 total reachable
signal, and personalisation adds ~0.000. So optimise the ITEM-QUALITY ESTIMATOR, which
every experiment so far has left at its first-guess form: unweighted counts inside the
frozen horizon, smoothed toward the GLOBAL mean with a fixed alpha=20.

Two principled upgrades, tested against that baseline on identical splits:
  decay   weight recent evidence more (item quality drifts; equal-weighting a 3-week-old
          impression with yesterday's is a modelling choice we never justified)
  hier    shrink a thin item toward its AUTHOR's rate rather than the global mean
          (for a 5-impression video, the author's rate over hundreds of impressions is a
          far better prior than 0.33)
"""
import sys, json, itertools; sys.path.insert(0,'.')
import numpy as np
from kairos.kernel.dataset import Data
from kairos.kernel.causal import (frozen_prefix, frozen_prefix_decayed, window_horizons,
                                  smoothed_rate, hierarchical_rate)
from kairos.kernel.frozenfeat import OFFICIAL_WINDOWS, within_user_deviation
from kairos.kernel.fastmetrics import fast_evaluate, factorize
import lightgbm as lgb

d = Data(); fold = d.fold('official')
tr, va = fold.idx['train'], fold.idx['valid']
g,_ = factorize(d.user_id[va]); yva = d.y_raw[va]
date = d.date.astype(np.int64); hz = window_horizons(date, OFFICIAL_WINDOWS)
yall = d.y_raw.astype(np.float64); ones = np.ones(d.n, bool)
prior = float(yall[date <= 20220421].mean())
vid = d.video_id.astype(np.int64)
vb = d.col('author_id','vb')
author = np.where(vid < len(vb), vb[np.minimum(vid,len(vb)-1)], -1).astype(np.int64)
tab = d.col('tab').astype(np.int64)
dur = d.col('duration_ms').astype(np.float64)

def item_cols(decay_hl, hier):
    """Return the item-quality block under a given estimator choice."""
    if decay_hl is None:
        li, pi = frozen_prefix(vid, date, yall, ones, hz)
        la, pa = frozen_prefix(author, date, yall, ones, hz)
    else:
        li, pi = frozen_prefix_decayed(vid, date, yall, ones, hz, decay_hl)
        la, pa = frozen_prefix_decayed(author, date, yall, ones, hz, decay_hl)
    a_rate = smoothed_rate(pa, la, prior, 20.0)
    i_rate = (hierarchical_rate(pi, li, a_rate, 20.0) if hier
              else smoothed_rate(pi, li, prior, 20.0))
    return [i_rate, np.log1p(li), a_rate, np.log1p(la)]

def fit(cols, seeds=(0,1,2)):
    X = np.stack(cols, 1).astype(np.float32)
    names = [f'f{i}' for i in range(X.shape[1])]
    dtr,_ = within_user_deviation(X, names, d.user_id, tr, window_id=hz)
    dva,_ = within_user_deviation(X, names, d.user_id, va, window_id=hz)
    Xtr = np.concatenate([X[tr],dtr],1); Xva = np.concatenate([X[va],dva],1)
    out=[]
    for sd in seeds:
        p = dict(objective='binary', metric='auc', learning_rate=0.05, num_leaves=63,
                 min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
                 bagging_freq=1, verbose=-1, seed=sd, num_threads=7)
        box={'p':-1}
        def cbe(env):
            if env.iteration % 25 and env.iteration != env.end_iteration-1: return
            m = fast_evaluate(g, yva, env.model.predict(Xva, num_iteration=env.iteration+1))
            if m['primary']>box['p']: box['p']=m['primary']
        ds = lgb.Dataset(Xtr, label=d.y_raw[tr])
        lgb.train(p, ds, num_boost_round=300, valid_sets=[ds],
                  callbacks=[cbe, lgb.log_evaluation(0)])
        out.append(box['p'])
    return float(np.mean(out)), float(np.std(out))

ctx = [smoothed_rate(*frozen_prefix(tab, date, yall, ones, hz)[::-1], prior, 20.0)
       if False else None]
lt, pt = frozen_prefix(tab, date, yall, ones, hz)
base_ctx = [smoothed_rate(pt, lt, prior, 20.0), np.log1p(dur), tab.astype(np.float64)]

print(f"{'item-quality estimator':<34} {'valid':>8} {'std':>7} {'vs base':>9}")
print("-"*62)
res = {}
base = None
for hl, hier in itertools.product([None, 14.0, 7.0, 3.0], [False, True]):
    nm = f"{'unweighted' if hl is None else f'decay H={hl:g}d':<16}{'+hier' if hier else ''}"
    m, s = fit(base_ctx + item_cols(hl, hier))
    if base is None: base = m
    res[nm.strip()] = {'valid': m, 'std': s}
    print(f"{nm:<34} {m:>8.4f} {s:>7.4f} {m-base:>+9.4f}")
json.dump(res, open('runs/exp20_item_quality.json','w'), indent=2)
best = max(res, key=lambda k: res[k]['valid'])
print(f"\nbest: {best} ({res[best]['valid']:.4f}, {res[best]['valid']-base:+.4f} vs unweighted+global)")
