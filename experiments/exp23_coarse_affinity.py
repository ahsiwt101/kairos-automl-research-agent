"""Density hypothesis: coarse-grained affinity should work where fine-grained fails.

exp19 found user x author (+0.0002) and user x item (+0.0002) contribute ~nothing, and
attributed it to 0.58% matrix density. But that is a claim about GRANULARITY, not about
personalisation. `tag` has 111 distinct values against author_id's 6,510 - a 59x denser
cross - so if the density explanation is right, user x tag should pay where user x author
does not. It is also the one video field the organizers' own 13-field ablation never
tested (theirs were author_id, music_id, video_type, upload_type).

Also tests VIDEO AGE (date - upload_dt), never used by anyone here, and item-side freshness
is a classic strong signal in short-video ranking.
"""
import sys, json; sys.path.insert(0,'.')
import numpy as np
from kairos.kernel.dataset import Data
from kairos.kernel.causal import frozen_prefix, window_horizons, smoothed_rate
from kairos.kernel.frozenfeat import OFFICIAL_WINDOWS, within_user_deviation
from kairos.kernel.fastmetrics import fast_evaluate, factorize
import lightgbm as lgb

d = Data(); fold = d.fold('official')
tr, va = fold.idx['train'], fold.idx['valid']
g,_ = factorize(d.user_id[va]); yva = d.y_raw[va]
date = d.date.astype(np.int64); hz = window_horizons(date, OFFICIAL_WINDOWS)
yall = d.y_raw.astype(np.float64); ones = np.ones(d.n, bool)
prior = float(yall[date <= 20220421].mean())
uid = d.user_id.astype(np.int64); vid = d.video_id.astype(np.int64)
tab = d.col('tab').astype(np.int64); dur = d.col('duration_ms').astype(np.float64)

def vattr(name):
    v = np.asarray(d.col(name,'vb'))
    out = np.zeros(d.n, dtype=np.int64)
    ok = vid < len(v); out[ok] = v[vid[ok]].astype(np.int64); return out
author = vattr('author_id'); tag = vattr('tag'); music = vattr('music_id')
vtype = vattr('video_type'); utype = vattr('upload_type')
upload_dt = vattr('upload_dt')

def pair(a,b): return np.unique(np.stack([a,b],1), axis=0, return_inverse=True)[1].astype(np.int64)
def rate(keys, alpha=20.0):
    l_,p_ = frozen_prefix(keys, date, yall, ones, hz)
    return smoothed_rate(p_, l_, prior, alpha), np.log1p(l_)

edges = np.quantile(dur[tr], np.linspace(0,1,11)[1:-1]); durb = np.searchsorted(edges,dur)
# video age: upload_dt is a sort-ordered factorised date code, so later code = later upload
day = (date//10000)*372 + ((date//100)%100)*31 + (date%100)
age = (day - day.min()) - upload_dt.astype(np.float64)

base = []
for keys in (tab, vid, pair(vid,tab)):
    r,n = rate(keys); base += [r,n]
base += [np.log1p(dur), tab.astype(np.float64)]

CANDIDATES = [
    ('user x author  (6,510 vals)', lambda: list(rate(pair(uid,author)))),
    ('user x item    (7,583 vals)', lambda: list(rate(pair(uid,vid)))),
    ('user x music   (6,283 vals)', lambda: list(rate(pair(uid,music)))),
    ('user x TAG       (111 vals)', lambda: list(rate(pair(uid,tag)))),
    ('user x upload_ty  (14 vals)', lambda: list(rate(pair(uid,utype)))),
    ('user x durbucket  (10 vals)', lambda: list(rate(pair(uid,durb)))),
    ('user x video_ty    (3 vals)', lambda: list(rate(pair(uid,vtype)))),
    ('video age (freshness)',       lambda: [age]),
    ('tag rate (item-side)',        lambda: list(rate(tag))),
]

def fit(cols, seeds=(0,1,2)):
    X = np.stack(cols,1).astype(np.float32)
    names=[f'f{i}' for i in range(X.shape[1])]
    dtr,_ = within_user_deviation(X,names,d.user_id,tr,window_id=hz)
    dva,_ = within_user_deviation(X,names,d.user_id,va,window_id=hz)
    Xtr=np.concatenate([X[tr],dtr],1); Xva=np.concatenate([X[va],dva],1)
    out=[]
    for sd in seeds:
        p=dict(objective='binary',metric='auc',learning_rate=0.05,num_leaves=63,
               min_data_in_leaf=200,feature_fraction=0.8,bagging_fraction=0.8,
               bagging_freq=1,verbose=-1,seed=sd,num_threads=7)
        box={'p':-1}
        def cbe(env):
            if env.iteration % 25 and env.iteration != env.end_iteration-1: return
            m=fast_evaluate(g,yva,env.model.predict(Xva,num_iteration=env.iteration+1))
            if m['primary']>box['p']: box['p']=m['primary']
        ds=lgb.Dataset(Xtr,label=d.y_raw[tr])
        lgb.train(p,ds,num_boost_round=300,valid_sets=[ds],
                  callbacks=[cbe,lgb.log_evaluation(0)])
        out.append(box['p'])
    return float(np.mean(out)), float(np.std(out))

b, bs = fit(base)
print(f"{'base (context + item quality)':<32} {b:>8.4f} +-{bs:.4f}\n")
print(f"{'added on top of base':<32} {'valid':>8} {'std':>7} {'marginal':>9}")
print("-"*60)
res={}
for nm, fn in CANDIDATES:
    m,s = fit(base + fn())
    res[nm]={'valid':m,'std':s,'marginal':m-b}
    print(f"{nm:<32} {m:>8.4f} {s:>7.4f} {m-b:>+9.4f}")
json.dump(res, open('runs/exp23_coarse_affinity.json','w'), indent=2)
best=max(res,key=lambda k:res[k]['valid'])
print(f"\nbest single addition: {best} ({res[best]['marginal']:+.4f})")
