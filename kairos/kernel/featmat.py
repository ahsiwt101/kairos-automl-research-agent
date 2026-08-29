"""Dense causal feature matrix: behaviour-derived signal the ID-embedding model cannot see.

Every column is produced by the prefix machinery in kairos.kernel.causal, so a row only
ever sees strictly-earlier events, and never a label past the fold horizon.
"""
import numpy as np
from kairos.kernel.causal import CausalFeatureBuilder, causal_prefix, smoothed_rate


def item_item_cf(data, train_idx, horizon, n_items=None, topk=None):
    """Cosine item-item similarity over co-long-viewed items, built from train only."""
    n_items = n_items or int(data.video_id.max()) + 1
    m = train_idx[(data.y_raw[train_idx] == 1) & (data.date[train_idx] <= horizon)]
    u = data.user_id[m].astype(np.int64)
    v = data.video_id[m].astype(np.int64)
    n_users = int(data.user_id.max()) + 1
    from scipy.sparse import csr_matrix
    M = csr_matrix((np.ones(len(u), dtype=np.float32), (u, v)), shape=(n_users, n_items))
    M.data[:] = 1.0                                    # binary: did the user long-view it
    C = (M.T @ M).toarray().astype(np.float32)         # co-occurrence
    norm = np.sqrt(np.maximum(np.diag(C), 1e-6))
    C /= norm[:, None]; C /= norm[None, :]
    np.fill_diagonal(C, 0.0)
    return C


def cf_scores(data, sim, horizon):
    """For each row: mean similarity between its item and the items this user long-viewed
    STRICTLY EARLIER.  Computed per user on their time-ordered rows, so it is causal."""
    n = data.n
    out = np.zeros(n, dtype=np.float32)
    cnt = np.zeros(n, dtype=np.float32)
    order = data.time_order
    u = data.user_id[order]
    v = data.video_id[order].astype(np.int64)
    y = data.y_raw[order].astype(bool) & (data.date[order] <= horizon)
    starts = np.flatnonzero(np.r_[True, u[1:] != u[:-1]])
    sizes = np.diff(np.r_[starts, n])
    res_o = np.zeros(n, dtype=np.float32); res_c = np.zeros(n, dtype=np.float32)
    for s, sz in zip(starts, sizes):
        if sz < 2:
            continue
        vv = v[s:s + sz]; yy = y[s:s + sz]
        S = sim[np.ix_(vv, vv)]                        # (sz, sz)
        prior = np.tril(np.ones((sz, sz), dtype=np.float32), -1) * yy[None, :]
        cc = prior.sum(1)
        res_o[s:s + sz] = (S * prior).sum(1)
        res_c[s:s + sz] = cc
    out[order] = res_o; cnt[order] = res_c
    return out, cnt


def build_matrix(data, fold, include_cf=True, horizon=None):
    """Returns (X float32 (N,F), names). Rows are ALL rows, aligned to data order.

    `horizon` overrides the fold's label horizon.  This matters more than it looks:
    features built at the fold horizon give VALIDATION rows label history right up to the
    row, while TEST rows are 1-10 days stale.  Validating on fresh history then deploying
    on stale history is a train/serve skew that inflates validation precisely for the
    recency-sensitive features - i.e. it corrupts model SELECTION, not just the score."""
    horizon = fold.horizon if horizon is None else int(horizon)
    cb = CausalFeatureBuilder(data, horizon)
    tr = fold.idx['train']
    edges = np.quantile(data.col('duration_ms')[tr].astype(np.float64),
                        np.linspace(0, 1, 11)[1:-1])

    cols, names = [], []

    def add(name, arr):
        cols.append(np.asarray(arr, dtype=np.float32)); names.append(name)

    for fam, fn in (('item', cb.item), ('author', cb.author), ('user', cb.user),
                    ('user_author', cb.user_author), ('user_item', cb.user_item),
                    ('user_tab', cb.user_tab)):
        n_, l_, p_, rate = fn()
        add(f'{fam}_rate', rate)
        add(f'{fam}_logn', np.log1p(l_))
    n_, l_, p_, rate = cb.user_durbucket(edges)
    add('user_dur_rate', rate); add('user_dur_logn', np.log1p(l_))

    # raw item side
    dur = data.col('duration_ms').astype(np.float64)
    add('log_duration', np.log1p(dur))
    add('dur_bucket', cb.dur_bucket.astype(np.float32))
    add('tab', data.col('tab').astype(np.float32))
    add('hour', (data.col('hourmin').astype(np.float32) // 100))

    # deviation features: the metric is within-user, so a feature's value RELATIVE to the
    # rest of that user's list is what can move the ranking at all.
    return np.stack(cols, 1), names, cb


def add_cf_columns(X, names, data, fold, horizon=None):
    horizon = fold.horizon if horizon is None else int(horizon)
    sim = item_item_cf(data, fold.idx['train'], horizon)
    s, c = cf_scores(data, sim, horizon)
    mean_s = s / np.maximum(c, 1.0)
    X = np.concatenate([X, np.stack([s, mean_s, np.log1p(c)], 1).astype(np.float32)], 1)
    return X, names + ['cf_sum', 'cf_mean', 'cf_hist_logn']


def within_user_deviation(X, names, user_id, cols=None):
    """Append each selected column minus its mean over the SAME user's rows in the same
    split.  Within-user ranking is invariant to per-user constants, so the deviation is
    the part of a feature that can actually change the ordering."""
    cols = cols or list(range(X.shape[1]))
    order = np.argsort(user_id, kind='stable')
    u = user_id[order]
    starts = np.flatnonzero(np.r_[True, u[1:] != u[:-1]])
    sizes = np.diff(np.r_[starts, len(u)])
    seg = np.repeat(np.arange(len(starts)), sizes)
    dev = np.empty((X.shape[0], len(cols)), dtype=np.float32)
    for j, c in enumerate(cols):
        v = X[order, c]
        means = np.bincount(seg, weights=v.astype(np.float64)) / sizes
        dev[order, j] = (v - means[seg]).astype(np.float32)
    return dev, [f'dev_{names[c]}' for c in cols]
