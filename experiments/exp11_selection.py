"""THE experiment: do selection rules differ in the hidden-test score they deliver?

Each candidate is evaluated on four temporal folds.  For the three backtest folds the test
window is inside the public-label region, so its score is an honest held-out measurement.
For the official fold the test score is obtained ONCE per candidate through the audited
scorer, and is used only to grade the rules afterwards - never to choose.

Rules compared:
  greedy   argmax official-validation primary        (what a conventional agent does)
  transfer argmax mean over backtest-fold test scores
  robust   transfer, penalised for cross-fold instability and winner's-curse shrinkage
  oracle   argmax official test                       (unattainable; measures regret)
"""
import sys, json, time; sys.path.insert(0,'.')
import numpy as np
from kairos.kernel.dataset import Data, FOLDS
from kairos.kernel.candidates import POOL, build_candidate_matrix
from kairos.kernel.frozenfeat import within_user_deviation
from kairos.kernel.fastmetrics import fast_evaluate, factorize
import lightgbm as lgb

d = Data()
FOLD_NAMES = ['official', 'backtest_a', 'backtest_c']
SEEDS = (0, 1)
ROUNDS = 220
res = {name: {} for name, _, _ in POOL}

for fname in FOLD_NAMES:
    spec = FOLDS[fname]; fold = d.fold(fname)
    tr, va, te = fold.idx['train'], fold.idx['valid'], fold.idx['test']
    ytr, yva = d.y_raw[tr], d.y_raw[va]
    gva,_ = factorize(d.user_id[va])
    for cname, mode, fams in POOL:
        t0=time.time()
        X, names, hz = build_candidate_matrix(d, spec, mode, fams)
        Pp = {}
        for tag, idx in (('tr',tr),('va',va),('te',te)):
            dev,_ = within_user_deviation(X, names, d.user_id, idx, window_id=hz)
            Pp[tag] = np.concatenate([X[idx], dev],1)
        vs, ts = [], []
        for sd in SEEDS:
            p = dict(objective='binary', metric='auc', learning_rate=0.05, num_leaves=63,
                     min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
                     bagging_freq=1, verbose=-1, seed=sd, num_threads=7)
            box={'primary':-1}
            def cbe(env):
                if env.iteration % 20 and env.iteration != env.end_iteration-1: return
                m = fast_evaluate(gva, yva, env.model.predict(Pp['va'],
                                  num_iteration=env.iteration+1))
                if m['primary']>box['primary']: box.update(m); box['iter']=env.iteration+1
            ds = lgb.Dataset(Pp['tr'], label=ytr)
            b = lgb.train(p, ds, num_boost_round=ROUNDS, valid_sets=[ds],
                          callbacks=[cbe, lgb.log_evaluation(0)])
            vs.append(box['primary'])
            ts.append(b.predict(Pp['te'], num_iteration=box['iter']))
        st = np.mean(ts,0)
        if fname == 'official':
            mt = fold.scorers['test'].score(st, reason=f'exp11 pool {cname}')['primary']
        else:
            gte,_ = factorize(d.user_id[te])
            mt = fast_evaluate(gte, d.y_raw[te], st)['primary']
        res[cname][fname] = {'valid': float(np.mean(vs)), 'valid_std': float(np.std(vs)),
                             'test': float(mt)}
        print(f"{fname:11s} {cname:14s} valid {np.mean(vs):.4f} test {mt:.4f} "
              f"gap {np.mean(vs)-mt:+.4f}  {time.time()-t0:.0f}s", flush=True)
json.dump(res, open('runs/exp11_selection.json','w'), indent=2)
print("\nsaved runs/exp11_selection.json")
