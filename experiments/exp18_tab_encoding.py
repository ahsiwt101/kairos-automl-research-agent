"""Tab is ~60% of model gain but is encoded badly. Three targeted fixes, measured.

Diagnosis: tab long_view rates span 100x (tab3 0.004, tab4 0.489) with NO meaningful
numeric ordering, yet we hand LightGBM the raw code 0..14 as a continuous feature. Worse,
within_user_deviation is applied to it, producing `dev_tab` = the deviation of a
categorical CODE, which is arithmetically meaningless (a user on tabs [1,1,4] gets
[-0.67,-0.67,1.33]).

  A  baseline          current encoding, as shipped
  B  target-encoded    replace raw tab with a leakage-safe FROZEN tab long_view rate, so
                       the model gets the quantity it actually needs in one split
  C  categorical       keep the code but declare it categorical to LightGBM, which then
                       does optimal subset splits instead of threshold splits
  D  B + C

Selection is on validation; the winner is then confirmed on a backtest fold before being
believed, per the discipline established in section 6 of the findings.
"""
import sys, json; sys.path.insert(0,'.')
import numpy as np
from kairos.kernel.dataset import Data, FOLDS
from kairos.kernel.causal import frozen_prefix, window_horizons, smoothed_rate
from kairos.kernel.frozenfeat import within_user_deviation, windows_for_fold
from kairos.kernel.fastmetrics import fast_evaluate, factorize
import lightgbm as lgb

def build_variant(d, fold_name, variant):
    spec = FOLDS[fold_name]
    wins = windows_for_fold(spec)
    date = d.date.astype(np.int64)
    hz = window_horizons(date, wins)
    X = np.load('runs/X_frozen.npy') if fold_name == 'official' else None
    names = json.load(open('runs/X_frozen_names.json'))
    if X is None:
        from kairos.kernel.frozenfeat import build_frozen_matrix
        X, names, hz = build_frozen_matrix(d, windows=wins)
    X = X.copy(); names = list(names)
    tab = d.col('tab').astype(np.int64)
    j = names.index('tab')
    if variant in ('B', 'D'):
        y = d.y_raw.astype(np.float64)
        l_, p_ = frozen_prefix(tab, date, y, np.ones(d.n, bool), hz)
        prior = float(y[date <= spec['train'][1]].mean())
        X[:, j] = smoothed_rate(p_, l_, prior, 20.0).astype(np.float32)
        names[j] = 'tab_rate_frozen'
    return X, names, hz, j

def run(d, fold_name, variant, seeds=(0,1,2), rounds=300, also_test=False):
    fold = d.fold(fold_name)
    tr, va = fold.idx['train'], fold.idx['valid']
    ytr, yva = d.y_raw[tr], d.y_raw[va]
    gva,_ = factorize(d.user_id[va])
    X, names, hz, j = build_variant(d, fold_name, variant)
    parts = {}
    for tag, idx in (('tr',tr),('va',va),('te',fold.idx['test'])):
        dev,_ = within_user_deviation(X, names, d.user_id, idx, window_id=hz)
        parts[tag] = np.concatenate([X[idx], dev], 1)
    cat = [j] if variant in ('C','D') else []
    vs, ts = [], []
    for sd in seeds:
        p = dict(objective='binary', metric='auc', learning_rate=0.05, num_leaves=63,
                 min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
                 bagging_freq=1, verbose=-1, seed=sd, num_threads=7)
        box={'primary':-1}
        def cbe(env):
            if env.iteration % 25 and env.iteration != env.end_iteration-1: return
            m = fast_evaluate(gva, yva, env.model.predict(parts['va'],
                              num_iteration=env.iteration+1))
            if m['primary']>box['primary']: box.update(m); box['iter']=env.iteration+1
        ds = lgb.Dataset(parts['tr'], label=ytr, categorical_feature=cat, free_raw_data=False)
        b = lgb.train(p, ds, num_boost_round=rounds, valid_sets=[ds],
                      callbacks=[cbe, lgb.log_evaluation(0)])
        vs.append(box['primary'])
        if also_test: ts.append(b.predict(parts['te'], num_iteration=box['iter']))
    out = {'valid': float(np.mean(vs)), 'valid_std': float(np.std(vs))}
    if also_test:
        gte,_ = factorize(d.user_id[fold.idx['test']])
        out['test'] = float(fast_evaluate(gte, d.y_raw[fold.idx['test']],
                                          np.mean(ts,0))['primary'])
    return out

d = Data()
print(f"{'variant':<28} {'valid':>8} {'std':>8}")
print("-"*48)
res = {}
LABEL = {'A':'A baseline (as shipped)', 'B':'B tab target-encoded',
         'C':'C tab categorical', 'D':'D both'}
for v in ('A','B','C','D'):
    r = run(d, 'official', v)
    res[v] = r
    print(f"{LABEL[v]:<28} {r['valid']:>8.4f} {r['valid_std']:>8.4f}")
best = max(res, key=lambda k: res[k]['valid'])
print(f"\nbest on validation: {LABEL[best]} ({res[best]['valid']:.4f} vs "
      f"A {res['A']['valid']:.4f}, delta {res[best]['valid']-res['A']['valid']:+.4f})")
json.dump(res, open('runs/exp18_tab_encoding.json','w'), indent=2)

if best != 'A':
    print(f"\n=== backtest confirmation of {best} vs A on backtest_a ===")
    for v in ('A', best):
        r = run(d, 'backtest_a', v, seeds=(0,1), also_test=True)
        print(f"  {LABEL[v]:<28} valid {r['valid']:.4f}  test {r['test']:.4f}  "
              f"gap {r['valid']-r['test']:+.4f}")
