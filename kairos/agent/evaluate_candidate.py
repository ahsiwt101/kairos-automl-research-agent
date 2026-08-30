"""Train + score one candidate feature matrix. Runs as its own process.

Isolated for two reasons: lightgbm and torch each bundle their own libomp and abort if
loaded together, and a candidate that hangs or segfaults must cost one iteration rather
than the run.

Multi-seed by default.  Per-seed std is 0.0008 while the competition's convergence rule
needs +0.002 to be detected, so a single seed cannot reliably resolve the very improvement
the rules demand; 3 seeds put the standard error at 0.00046.
"""
import sys, json
import numpy as np
sys.path.insert(0, '.')
from kairos.kernel.dataset import Data
from kairos.kernel.frozenfeat import within_user_deviation
from kairos.kernel.fastmetrics import fast_evaluate, factorize
import lightgbm as lgb

PARAMS = dict(objective='binary', metric='auc', learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, verbose=-1, num_threads=8)
# Hyperparameters a candidate's train_cfg is allowed to override. This exact PARAMS dict
# has been used, UNTUNED, across every experiment this project has run - it is free
# points for whichever candidate is the first to try something else.
TUNABLE = ('learning_rate', 'num_leaves', 'min_data_in_leaf', 'feature_fraction',
          'bagging_fraction', 'bagging_freq', 'lambda_l1', 'lambda_l2',
          'max_depth', 'min_gain_to_split')
_BOUNDS = dict(learning_rate=(0.005, 0.3), num_leaves=(7, 255), min_data_in_leaf=(20, 2000),
              feature_fraction=(0.3, 1.0), bagging_fraction=(0.3, 1.0), bagging_freq=(0, 10),
              lambda_l1=(0.0, 10.0), lambda_l2=(0.0, 10.0), max_depth=(-1, 16),
              min_gain_to_split=(0.0, 1.0))


def _sanitize_hparams(hp):
    """Clip to sane ranges rather than trust the caller - a wild learning_rate or
    negative min_data_in_leaf would otherwise burn a whole iteration on a training
    pathology instead of a real result."""
    out = {}
    for k, v in (hp or {}).items():
        if k not in TUNABLE:
            raise ValueError(f"train_cfg.hparams: '{k}' is not tunable; choose from "
                             f"{TUNABLE}")
        lo, hi = _BOUNDS[k]
        out[k] = type(lo)(np.clip(v, lo, hi))
    return out


def evaluate(X, fold_name, seeds=(0, 1, 2), rounds=300, add_dev=True, hz=None,
             train_cfg=None):
    d = Data(); fold = d.fold(fold_name)
    tr, va = fold.idx['train'], fold.idx['valid']
    ytr, yva = d.y_raw[tr], d.y_raw[va]
    gva, _ = factorize(d.user_id[va])

    cfg = train_cfg or {}
    if cfg.get('mode') == 'scores':
        # X is already the FINAL per-row score - the candidate trained and blended its own
        # model(s) inside build(). No refit; just score it. This is the escape hatch from
        # "concatenate everything into one feature matrix for one downstream tree", which
        # measurably does not work here (three live attempts, all worse than the FM
        # baseline: LightGBM shatters a smooth continuous score into step functions).
        s = np.asarray(X, dtype=np.float64).reshape(-1)
        m = fast_evaluate(gva, yva, s[va])
        return {'objective': 'scores', 'valid_primary': m['primary'],
                'valid_std': 0.0, 'valid_gauc': m['GAUC'], 'valid_ndcg': m['nDCG@5'],
                'seeds': [m['primary']], 'best_iter': None}

    names = [f'f{i}' for i in range(X.shape[1])]
    if add_dev:
        dtr, _ = within_user_deviation(X, names, d.user_id, tr, window_id=hz)
        dva, _ = within_user_deviation(X, names, d.user_id, va, window_id=hz)
        Xtr = np.concatenate([X[tr], dtr], 1); Xva = np.concatenate([X[va], dva], 1)
    else:
        Xtr, Xva = X[tr], X[va]
    cfg = train_cfg or {}
    obj = cfg.get('objective', 'binary')
    grp = cfg.get('group', 'user_day')
    hparams = _sanitize_hparams(cfg.get('hparams'))
    if obj == 'lambdarank':
        g = d.user_id[tr].astype(np.int64)
        if grp == 'user_day':
            g = g * 100000 + d.date[tr] % 100000
        gorder = np.argsort(g, kind='stable'); gs = g[gorder]
        gsizes = np.diff(np.r_[np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]]), len(gs)])
    out = []
    for sd in seeds:
        p = dict(PARAMS, **hparams, seed=sd)
        if obj == 'lambdarank':
            p.pop('metric', None)
            p.update(objective='lambdarank', metric='ndcg', ndcg_eval_at=[5],
                     lambdarank_truncation_level=15)
        box = {'primary': -1}
        def cbe(env):
            if env.iteration % 20 and env.iteration != env.end_iteration - 1:
                return
            m = fast_evaluate(gva, yva, env.model.predict(Xva, num_iteration=env.iteration+1))
            if m['primary'] > box['primary']:
                box.update(m); box['iter'] = env.iteration + 1
        ds = (lgb.Dataset(Xtr[gorder], label=ytr[gorder], group=gsizes)
              if obj == 'lambdarank' else lgb.Dataset(Xtr, label=ytr))
        lgb.train(p, ds, num_boost_round=rounds, valid_sets=[ds],
                  callbacks=[cbe, lgb.log_evaluation(0)])
        out.append(box)
    prim = [o['primary'] for o in out]
    return {'objective': obj,
            'valid_primary': float(np.mean(prim)), 'valid_std': float(np.std(prim)),
            'valid_gauc': float(np.mean([o['GAUC'] for o in out])),
            'valid_ndcg': float(np.mean([o['nDCG@5'] for o in out])),
            'seeds': prim, 'best_iter': int(np.median([o['iter'] for o in out]))}


if __name__ == '__main__':
    cfg = json.load(open(sys.argv[1]))
    X = np.load(cfg['X_path'])
    hz = np.load(cfg['hz_path']) if cfg.get('hz_path') else None
    r = evaluate(X, cfg.get('fold', 'official'), tuple(cfg.get('seeds', (0, 1, 2))),
                 cfg.get('rounds', 300), cfg.get('add_dev', True), hz,
                 cfg.get('train_cfg'))
    json.dump(r, open(cfg['out'], 'w'))
    print('OK')
