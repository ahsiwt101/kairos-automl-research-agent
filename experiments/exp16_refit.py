"""Should the final model be refit on train+validation?

Test is 8-17 days after training ends, and validation is 7 more days of data that sits
closer in time to it. Refitting on train+valid should help on both counts - but you cannot
verify it directly, because the moment you train on validation your validation score is
meaningless.

So validate the PROCEDURE rather than the model: on backtest folds, whose test windows are
in the public-label region, compare train-only against train+valid, using the boosting
round count chosen by the train-only run (the only honest way to pick it). If refitting
wins consistently there, apply it to the official fold.
"""
import sys, json, time; sys.path.insert(0,'.')
import numpy as np
from kairos.kernel.dataset import Data, FOLDS
from kairos.kernel.candidates import build_candidate_matrix, FAMILIES
from kairos.kernel.frozenfeat import within_user_deviation
from kairos.kernel.fastmetrics import fast_evaluate, factorize
import lightgbm as lgb

d = Data()
SEEDS = (0, 1)
rows = []
for fname in ('backtest_a', 'backtest_c', 'official'):
    spec = FOLDS[fname]; fold = d.fold(fname)
    tr, va, te = fold.idx['train'], fold.idx['valid'], fold.idx['test']
    ytr, yva = d.y_raw[tr], d.y_raw[va]
    gva,_ = factorize(d.user_id[va])
    X, names, hz = build_candidate_matrix(d, spec, 'frozen', FAMILIES)
    P = {}
    for tag, idx in (('tr',tr),('va',va),('te',te)):
        dev,_ = within_user_deviation(X, names, d.user_id, idx, window_id=hz)
        P[tag] = np.concatenate([X[idx], dev],1)

    # pass 1: train-only, pick the round count on validation
    best_iters, ts_train_only = [], []
    for sd in SEEDS:
        p = dict(objective='binary', metric='auc', learning_rate=0.05, num_leaves=63,
                 min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
                 bagging_freq=1, verbose=-1, seed=sd, num_threads=7)
        box={'primary':-1}
        def cbe(env):
            if env.iteration % 20 and env.iteration != env.end_iteration-1: return
            m = fast_evaluate(gva, yva, env.model.predict(P['va'], num_iteration=env.iteration+1))
            if m['primary']>box['primary']: box.update(m); box['iter']=env.iteration+1
        b = lgb.train(p, lgb.Dataset(P['tr'], label=ytr), num_boost_round=350,
                      valid_sets=[lgb.Dataset(P['tr'], label=ytr)],
                      callbacks=[cbe, lgb.log_evaluation(0)])
        best_iters.append(box['iter'])
        ts_train_only.append(b.predict(P['te'], num_iteration=box['iter']))
    n_rounds = int(np.median(best_iters))

    # pass 2: refit on train+valid for the SAME number of rounds
    Xb = np.concatenate([P['tr'], P['va']], 0); yb = np.concatenate([ytr, yva])
    ts_refit = []
    for sd in SEEDS:
        p = dict(objective='binary', metric='auc', learning_rate=0.05, num_leaves=63,
                 min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
                 bagging_freq=1, verbose=-1, seed=sd, num_threads=7)
        b = lgb.train(p, lgb.Dataset(Xb, label=yb), num_boost_round=n_rounds)
        ts_refit.append(b.predict(P['te']))

    def sc(pred):
        if fname == 'official':
            return fold.scorers['test'].score(pred, reason='exp16 refit')['primary']
        g,_ = factorize(d.user_id[te]); return fast_evaluate(g, d.y_raw[te], pred)['primary']
    a = sc(np.mean(ts_train_only,0)); bsc = sc(np.mean(ts_refit,0))
    rows.append({'fold': fname, 'rounds': n_rounds, 'train_only': a, 'refit': bsc,
                 'delta': bsc - a, 'extra_rows': int(len(va))})
    print(f"{fname:12s} rounds={n_rounds:4d}  train-only {a:.4f}  refit(+{len(va):,} rows) "
          f"{bsc:.4f}  delta {bsc-a:+.4f}", flush=True)

bt = [r['delta'] for r in rows if r['fold'].startswith('backtest')]
print(f"\nbacktest folds say refitting is worth {np.mean(bt):+.4f} on average "
      f"({', '.join(f'{x:+.4f}' for x in bt)})")
print("official fold shown for reference; the DECISION is made on the backtests only.")
json.dump(rows, open('runs/exp16_refit.json','w'), indent=2, default=float)
