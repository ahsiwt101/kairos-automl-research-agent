"""Sub-space experts: models trained on deliberately disjoint feature families.

Rank fusion is the only mechanism measured to pay on this benchmark, and fusion is rewarded
by DECORRELATION, not by individual member strength. Our existing members are less
decorrelated than their architectures suggest - FM and DIN correlate at Spearman +0.848
despite being unrelated model families, because both converge on the same dominant signals
(item quality and tab).

Forcing each expert onto a disjoint feature sub-space is a direct attack on that: an expert
that cannot see item identity cannot rediscover item quality, so its errors must be
differently distributed. Three sub-spaces, chosen so no feature appears in two:

    context   tab, hour, duration, staleness      - the surface and the moment
    item      item / author / item x tab rates    - intrinsic item quality
    user      user x tab, user x duration rates   - this user's dispositions

Whether that decorrelation actually materialises is an empirical question, not an
assumption - see experiments/exp26_subspace_experts.py, which measures the pairwise
Spearman and treats the FM/DIN +0.848 as the number to beat.

Leakage discipline matches the rest of the kernel: every rate comes from frozen_prefix at
the row's own window horizon, so no row sees a label from inside its evaluation window.
"""
import os
import numpy as np
from kairos.kernel.dataset import variant_path

CACHE_DIR = variant_path('runs/expert_cache')

SUBSPACES = ('context', 'item', 'user')


def _columns(data, subspace, hz, windows):
    """Feature block for one sub-space. Deliberately disjoint across sub-spaces."""
    from kairos.kernel.causal import frozen_prefix, smoothed_rate

    date = data.date.astype(np.int64)
    y = data.y_raw.astype(np.float64)
    ones = np.ones(data.n, dtype=bool)
    prior = float(y[date <= windows[-2][1]].mean())
    vid = data.video_id.astype(np.int64)
    uid = data.user_id.astype(np.int64)
    tab = data.col('tab').astype(np.int64)
    dur = data.col('duration_ms').astype(np.float64)

    def pair(a, b):
        return np.unique(np.stack([a, b], 1), axis=0, return_inverse=True)[1].astype(np.int64)

    def rate(keys):
        l_, p_ = frozen_prefix(keys, date, y, ones, hz)
        return [smoothed_rate(p_, l_, prior, 20.0), np.log1p(l_)]

    if subspace == 'context':
        # No item identity and no user identity - only the surface and the moment.
        from kairos.kernel.frozenfeat import _dayindex
        cols = rate(tab)
        cols += [np.log1p(dur),
                 data.col('hourmin').astype(np.float64) // 100,
                 (_dayindex(date) - _dayindex(hz)).astype(np.float64)]
        names = ['tab_rate', 'tab_logn', 'log_duration', 'hour', 'staleness']

    elif subspace == 'item':
        vb = data.col('author_id', 'vb')
        author = np.where(vid < len(vb), vb[np.minimum(vid, len(vb) - 1)], -1).astype(np.int64)
        cols = rate(vid) + rate(author) + rate(pair(vid, tab))
        names = ['item_rate', 'item_logn', 'author_rate', 'author_logn',
                 'item_tab_rate', 'item_tab_logn']

    elif subspace == 'user':
        edges = np.quantile(dur[date <= windows[-2][1]], np.linspace(0, 1, 11)[1:-1])
        durb = np.searchsorted(edges, dur).astype(np.int64)
        cols = rate(pair(uid, tab)) + rate(pair(uid, durb)) + rate(uid)
        names = ['user_tab_rate', 'user_tab_logn', 'user_dur_rate', 'user_dur_logn',
                 'user_rate', 'user_logn']
    else:
        raise ValueError(f"expert_score: '{subspace}' not in {SUBSPACES}")

    return np.stack(cols, 1).astype(np.float32), names


def build_expert_signal(data, fold, hz, subspace, seeds=(0, 1), rounds=250,
                        cache_dir=CACHE_DIR, force=False):
    """Returns float32 (n,) - out-of-sample score from a model that sees ONLY `subspace`."""
    if subspace not in SUBSPACES:
        raise ValueError(f"expert_score: '{subspace}' not in {SUBSPACES}")
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f'{fold.name}_{subspace}.npy')
    if os.path.exists(cache) and not force:
        return np.load(cache)

    import lightgbm as lgb
    from kairos.kernel.frozenfeat import within_user_deviation, windows_for_fold
    from kairos.kernel.dataset import FOLDS
    from kairos.kernel.fastmetrics import fast_evaluate, factorize

    windows = windows_for_fold(FOLDS[fold.name])
    X, names = _columns(data, subspace, hz, windows)
    tr, va = fold.idx['train'], fold.idx['valid']
    # within-user deviations: the metric only sees within-user contrasts, so the deviation
    # of each feature is the part that can actually reorder a list
    dev, _ = within_user_deviation(X, names, data.user_id, np.arange(data.n), window_id=hz)
    Xf = np.concatenate([X, dev], 1)
    ytr = data.y_raw[tr]
    gva, _ = factorize(data.user_id[va]); yva = data.y_raw[va]

    preds = []
    for sd in seeds:
        p = dict(objective='binary', metric='auc', learning_rate=0.05, num_leaves=63,
                 min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
                 bagging_freq=1, verbose=-1, seed=sd, num_threads=7)
        box = {'p': -1, 'it': rounds}
        def cbe(env):
            if env.iteration % 25 and env.iteration != env.end_iteration - 1:
                return
            m = fast_evaluate(gva, yva,
                              env.model.predict(Xf[va], num_iteration=env.iteration + 1))
            if m['primary'] > box['p']:
                box['p'] = m['primary']; box['it'] = env.iteration + 1
        ds = lgb.Dataset(Xf[tr], label=ytr)
        b = lgb.train(p, ds, num_boost_round=rounds, valid_sets=[ds],
                      callbacks=[cbe, lgb.log_evaluation(0)])
        preds.append(b.predict(Xf, num_iteration=box['it']))
        print(f"  expert[{subspace}] seed{sd}: valid {box['p']:.4f}", flush=True)

    out = np.mean(preds, 0).astype(np.float32)
    np.save(cache, out)
    return out
