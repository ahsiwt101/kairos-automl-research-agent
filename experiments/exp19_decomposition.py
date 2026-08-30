"""Where does the achievable score actually come from, and where does it stop?

Everyone reports a number against the baseline. Almost nobody reports WHY the number is
where it is. This decomposes the primary metric into nested information sources, each an
honest leakage-safe predictor, so the remaining headroom can be attributed rather than
guessed at. Every rung is evaluated on the same validation split.

The oracle rungs at the end are NOT models - they use the true label / true watch time and
exist purely to bound what any model could achieve.
"""
import sys, json; sys.path.insert(0,'.')
import numpy as np
from kairos.kernel.dataset import Data
from kairos.kernel.causal import frozen_prefix, window_horizons, smoothed_rate
from kairos.kernel.frozenfeat import OFFICIAL_WINDOWS
from kairos.kernel.fastmetrics import fast_evaluate, factorize

d = Data(); fold = d.fold('official')
tr, va = fold.idx['train'], fold.idx['valid']
g,_ = factorize(d.user_id[va]); y = d.y_raw[va]
date = d.date.astype(np.int64)
hz = window_horizons(date, OFFICIAL_WINDOWS)
yall = d.y_raw.astype(np.float64)
prior = float(yall[date <= 20220421].mean())
ones = np.ones(d.n, bool)

def frozen_rate(keys, alpha=20.0):
    l_, p_ = frozen_prefix(keys, date, yall, ones, hz)
    return smoothed_rate(p_, l_, prior, alpha)

def ev(s):
    return fast_evaluate(g, y, np.asarray(s, dtype=np.float64)[va])['primary']

uid = d.user_id.astype(np.int64); vid = d.video_id.astype(np.int64)
tab = d.col('tab').astype(np.int64)
vb = d.col('author_id','vb'); author = np.where(vid < len(vb), vb[np.minimum(vid,len(vb)-1)], -1).astype(np.int64)
dur = d.col('duration_ms').astype(np.float64)
edges = np.quantile(dur[tr], np.linspace(0,1,11)[1:-1]); durb = np.searchsorted(edges, dur)
def pair(a,b): return np.unique(np.stack([a,b],1), axis=0, return_inverse=True)[1].astype(np.int64)

# Two DIFFERENT questions, reported separately - conflating them produces a table that
# looks like a decomposition and is not one.
#   (1) standalone: what does this single signal rank at, by itself?
#   (2) marginal:   what does ADDING it to everything before it buy, in a fitted model?
# A sparse affinity rate scores ~random standalone (most pairs have no frozen history, so
# the smoothed rate collapses to a constant prior that cannot order anything within a
# user), yet can still contribute inside a model that has other features to gate it on.
import lightgbm as lgb
from kairos.kernel.frozenfeat import within_user_deviation

FEATS = [
    ('context (tab)',                [frozen_rate(tab)]),
    ('item quality',                 [frozen_rate(vid), np.log1p(dur)]),
    ('item x context',               [frozen_rate(pair(vid, tab))]),
    ('duration fit (user x durbkt)', [frozen_rate(pair(uid, durb))]),
    ('affinity (user x author)',     [frozen_rate(pair(uid, author))]),
    ('affinity (user x item)',       [frozen_rate(pair(uid, vid))]),
]

def fit_eval(cols):
    X = np.stack(cols, 1).astype(np.float32)
    names = [f'f{i}' for i in range(X.shape[1])]
    dtr,_ = within_user_deviation(X, names, d.user_id, tr, window_id=hz)
    dva,_ = within_user_deviation(X, names, d.user_id, va, window_id=hz)
    Xtr = np.concatenate([X[tr], dtr],1); Xva = np.concatenate([X[va], dva],1)
    best = -1
    for sd in (0,1):
        prm = dict(objective='binary', metric='auc', learning_rate=0.05, num_leaves=31,
                   min_data_in_leaf=200, verbose=-1, seed=sd, num_threads=7)
        box={'p':-1}
        def cbe(env):
            if env.iteration % 25 and env.iteration != env.end_iteration-1: return
            m = fast_evaluate(g, y, env.model.predict(Xva, num_iteration=env.iteration+1))
            if m['primary']>box['p']: box['p']=m['primary']
        ds = lgb.Dataset(Xtr, label=d.y_raw[tr])
        lgb.train(prm, ds, num_boost_round=200, valid_sets=[ds],
                  callbacks=[cbe, lgb.log_evaluation(0)])
        best = max(best, box['p'])
    return best

print(f"{'signal':<32} {'standalone':>11} {'cumulative':>11} {'marginal':>10}")
print("-"*68)
cum_cols, prev = [], None
for nm, cols in FEATS:
    solo = ev(cols[0])
    cum_cols += cols
    cum = fit_eval(cum_cols)
    marg = '' if prev is None else f"{cum-prev:+.4f}"
    print(f"{nm:<32} {solo:>11.4f} {cum:>11.4f} {marg:>10}")
    prev = cum
print()
print(f"{'combined models':<38} {'primary':>9}")
print("-"*50)
print(f"{'FM (official baseline)':<38} {0.6016:>9.4f}")
print(f"{'our best ensemble':<38} {0.6045:>9.4f}")

print()
print(f"{'ORACLE BOUNDS (not models)':<38} {'primary':>9}")
print("-"*50)
pt = d.col('play_time_ms').astype(np.float64)
wr = pt/np.maximum(dur,1)
for nm, s in (('perfect watch-ratio predictor', wr),
              ('perfect play-time predictor', pt),
              ('perfect label (ceiling)', d.y_raw.astype(np.float64))):
    print(f"{nm:<38} {ev(s):>9.4f}")

print("""
Reading the two columns differently matters. A signal can score at random STANDALONE (a
sparse affinity rate is a constant prior for most rows, and a constant cannot order a
user's list) while still paying inside a fitted model that has other features to gate it
on. The cumulative column is the honest one for "how far can this feature set go".""")
