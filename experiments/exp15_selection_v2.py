"""Selection-rule comparison over a pool that also varies the OBJECTIVE.

exp11/exp12 used a binary-objective-only pool. There the leak inflated validation hugely
(+0.13) but barely moved the test score, so greedy and robust selection tied - a corrupted
measurement costs nothing when every candidate is truly equivalent.

lambdarank over per-day groups is a standard choice for a ranking metric, and it is where
exploiting within-window label feedback becomes destructive rather than merely misleading.
Including it is what makes the pool representative of what an agent would actually search.
"""
import sys, json, time; sys.path.insert(0,'.')
import numpy as np
from kairos.kernel.dataset import Data, FOLDS
from kairos.kernel.candidates import POOL_V2, build_candidate_matrix
from kairos.kernel.frozenfeat import within_user_deviation
from kairos.kernel.fastmetrics import fast_evaluate, factorize
import lightgbm as lgb

d = Data()
FOLD_NAMES = ['official', 'backtest_a', 'backtest_c']
SEEDS = (0, 1); ROUNDS = 250
res = {n: {} for n,_,_,_,_ in POOL_V2}

for fname in FOLD_NAMES:
    spec = FOLDS[fname]; fold = d.fold(fname)
    tr, va, te = fold.idx['train'], fold.idx['valid'], fold.idx['test']
    ytr, yva = d.y_raw[tr], d.y_raw[va]
    gva,_ = factorize(d.user_id[va])
    for cname, mode, fams, obj, grp in POOL_V2:
        t0 = time.time()
        X, names, hz = build_candidate_matrix(d, spec, mode, fams)
        Pp = {}
        for tag, idx in (('tr',tr),('va',va),('te',te)):
            dev,_ = within_user_deviation(X, names, d.user_id, idx, window_id=hz)
            Pp[tag] = np.concatenate([X[idx], dev],1)
        if obj == 'lambdarank':
            g = d.user_id[tr].astype(np.int64)*100000 + d.date[tr] % 100000
            o = np.argsort(g, kind='stable'); gs = g[o]
            sizes = np.diff(np.r_[np.flatnonzero(np.r_[True, gs[1:]!=gs[:-1]]), len(gs)])
        vs, ts = [], []
        for sd in SEEDS:
            p = dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                     feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                     verbose=-1, seed=sd, num_threads=7)
            if obj == 'binary':
                p.update(objective='binary', metric='auc')
                ds = lgb.Dataset(Pp['tr'], label=ytr)
            else:
                p.update(objective='lambdarank', metric='ndcg', ndcg_eval_at=[5],
                         lambdarank_truncation_level=15)
                ds = lgb.Dataset(Pp['tr'][o], label=ytr[o], group=sizes)
            box={'primary':-1}
            def cbe(env):
                if env.iteration % 25 and env.iteration != env.end_iteration-1: return
                m = fast_evaluate(gva, yva, env.model.predict(Pp['va'],
                                  num_iteration=env.iteration+1))
                if m['primary']>box['primary']: box.update(m); box['iter']=env.iteration+1
            b = lgb.train(p, ds, num_boost_round=ROUNDS, valid_sets=[ds],
                          callbacks=[cbe, lgb.log_evaluation(0)])
            vs.append(box['primary']); ts.append(b.predict(Pp['te'], num_iteration=box['iter']))
        st = np.mean(ts,0)
        if fname=='official':
            mt = fold.scorers['test'].score(st, reason=f'exp15 {cname}')['primary']
        else:
            gte,_ = factorize(d.user_id[te]); mt = fast_evaluate(gte, d.y_raw[te], st)['primary']
        res[cname][fname] = {'valid': float(np.mean(vs)), 'valid_std': float(np.std(vs)),
                             'test': float(mt)}
        print(f"{fname:11s} {cname:17s} valid {np.mean(vs):.4f} test {mt:.4f} "
              f"gap {np.mean(vs)-mt:+.4f}  {time.time()-t0:.0f}s", flush=True)
json.dump(res, open('runs/exp15_selection_v2.json','w'), indent=2)
print("\nsaved runs/exp15_selection_v2.json")
