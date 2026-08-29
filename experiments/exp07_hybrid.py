"""Two cheap, well-motivated interventions, then the hybrid.

(1) Recency weighting - 59% of training rows sit in a dense 3-day regime the evaluation
    window does not resemble, so reweight toward the regime being served.
(2) Rank fusion of the ID model (FM) with the behavioural model (GBDT on frozen features).
    They are complementary: FM has user x author / user x duration crosses but no notion of
    behavioural history or staleness; the GBDT has the reverse. Fusion is done on WITHIN-USER
    ranks because the metric only sees within-user order.
"""
import sys, time, json; sys.path.insert(0,'.')
import numpy as np
from kairos.kernel.dataset import Data
from kairos.kernel.features import Encoder
from kairos.kernel.frozenfeat import within_user_deviation
from kairos.kernel.fastmetrics import fast_evaluate, factorize
from kairos.models.train import train_fm, predict
import lightgbm as lgb

d = Data(); fold = d.fold('official')
tr, va, te = fold.idx['train'], fold.idx['valid'], fold.idx['test']
ytr, yva = d.y_raw[tr], d.y_raw[va]
gva, _ = factorize(d.user_id[va])
enc = Encoder(d).fit(tr)

def ev(s):  return fast_evaluate(gva, yva, s)

# ---------------- 1. recency weighting on the FM ----------------------------
print("=== FM recency weighting ===")
fm_models = {}
for tau in (None, 14, 7, 3):
    t0=time.time()
    r = train_fm(fold, enc, loss='bce', seed=0, recency_tau=tau)
    sv = predict(r['model'], enc, va)
    fm_models[f'fm_tau{tau}'] = (r, sv)
    print(f"  tau={str(tau):>4}  valid primary {r['valid']['primary']:.4f}  "
          f"(ep {r['best_epoch']})  {time.time()-t0:.0f}s")

best_fm_key = max(fm_models, key=lambda k: fm_models[k][0]['valid']['primary'])
print(f"  -> best on validation: {best_fm_key}")
r_fm, sv_fm = fm_models[best_fm_key]
st_fm = predict(r_fm['model'], enc, te)

# ---------------- 2. GBDT on frozen features, deviations grouped by window ---
print("\n=== GBDT on frozen features (window-grouped deviations) ===")
X = np.load('runs/X_frozen.npy'); names = json.load(open('runs/X_frozen_names.json'))
hz = np.load('runs/X_frozen_hz.npy')
dtr_, dn = within_user_deviation(X, names, d.user_id, tr, window_id=hz)
dva_, _  = within_user_deviation(X, names, d.user_id, va, window_id=hz)
dte_, _  = within_user_deviation(X, names, d.user_id, te, window_id=hz)
Xtr = np.concatenate([X[tr], dtr_],1); Xva = np.concatenate([X[va], dva_],1)
Xte = np.concatenate([X[te], dte_],1); allnames = names+dn
params = dict(objective='binary', metric='auc', learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, verbose=-1, seed=0, num_threads=8)
box={'primary':-1}
def cbe(env):
    if env.iteration % 20 and env.iteration != env.end_iteration-1: return
    m = ev(env.model.predict(Xva, num_iteration=env.iteration+1))
    if m['primary']>box['primary']: box.update(m); box['iter']=env.iteration+1
bst = lgb.train(params, lgb.Dataset(Xtr, label=ytr), num_boost_round=400,
                valid_sets=[lgb.Dataset(Xtr, label=ytr)],
                callbacks=[cbe, lgb.log_evaluation(0)])
sv_gb = bst.predict(Xva, num_iteration=box['iter'])
st_gb = bst.predict(Xte, num_iteration=box['iter'])
print(f"  valid primary {box['primary']:.4f} (iter {box['iter']})")

# ---------------- 3. rank fusion --------------------------------------------
def within_user_rank(scores, users):
    """Percentile rank inside each user's list; the metric is invariant to anything else."""
    order = np.lexsort((np.arange(len(scores)), -np.asarray(scores), users))
    u = users[order]
    starts = np.flatnonzero(np.r_[True, u[1:]!=u[:-1]])
    sizes = np.diff(np.r_[starts, len(u)])
    seg = np.repeat(np.arange(len(starts)), sizes)
    pos = np.arange(len(u)) - starts[seg]
    pct = 1.0 - pos/np.maximum(sizes[seg]-1, 1)
    out = np.empty(len(scores)); out[order] = pct
    return out

uva, ute = d.user_id[va], d.user_id[te]
rv_fm, rv_gb = within_user_rank(sv_fm, uva), within_user_rank(sv_gb, uva)
rt_fm, rt_gb = within_user_rank(st_fm, ute), within_user_rank(st_gb, ute)

print("\n=== rank fusion sweep (selected on validation only) ===")
best_w, best_p = None, -1
for w in np.arange(0, 1.01, 0.05):
    p = ev(w*rv_fm + (1-w)*rv_gb)['primary']
    if p > best_p: best_p, best_w = p, w
    if abs(w*20 - round(w*20)) < 1e-9 and int(round(w*20)) % 2 == 0:
        print(f"  w_fm={w:.2f}  valid primary {p:.4f}")
print(f"  -> best w_fm={best_w:.2f} valid {best_p:.4f}")

results = {}
for tag, sv_, st_ in (('FM alone', sv_fm, st_fm),
                      ('GBDT alone', sv_gb, st_gb),
                      ('fusion', best_w*rv_fm+(1-best_w)*rv_gb, best_w*rt_fm+(1-best_w)*rt_gb)):
    mv = ev(sv_); mt = fold.scorers['test'].score(st_, reason=f'exp07 {tag}')
    results[tag] = {'valid': mv, 'test': mt}
    print(f"{tag:14s} valid {mv['primary']:.4f} | test {mt['primary']:.4f} | "
          f"gap {mv['primary']-mt['primary']:+.4f}")
print("\nbaseline FM: valid 0.6016 | test 0.5946")
json.dump(results, open('runs/exp07_hybrid.json','w'), indent=2, default=float)
np.save('runs/sv_fm.npy', sv_fm); np.save('runs/st_fm.npy', st_fm)
np.save('runs/sv_gb.npy', sv_gb); np.save('runs/st_gb.npy', st_gb)
