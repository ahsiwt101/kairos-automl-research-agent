"""Deployment-faithful feature matrix: every aggregate frozen at its window's start.

Retrain boundaries are placed so that no row ever sees a label from inside its own
evaluation window.  A `staleness` column is included so the model can learn how much to
discount history by its age - the official test window is up to 10 days past its horizon,
further than any training window, and without that column the model has no way to express
"this rate is old, trust it less".
"""
import numpy as np
from kairos.kernel.causal import frozen_prefix, window_horizons, smoothed_rate

# (lo, hi, horizon).  Aligned to the official split boundaries; the early dense period is
# cut finer so that fewer rows are left with no history at all.
OFFICIAL_WINDOWS = [
    (20220408, 20220410, 20220407),   # no prior data exists; history columns are 0
    (20220411, 20220412, 20220410),
    (20220413, 20220414, 20220412),
    (20220415, 20220417, 20220414),
    (20220418, 20220421, 20220417),
    (20220422, 20220428, 20220421),   # validation: frozen at end of train
    (20220429, 20220508, 20220428),   # test: frozen at end of validation
]


def _dayindex(d):
    """Calendar-ish day index good enough for differencing within Apr-May 2022."""
    d = np.asarray(d).astype(np.int64)
    return (d // 10000) * 372 + ((d // 100) % 100) * 31 + (d % 100)


def build_frozen_matrix(data, windows=OFFICIAL_WINDOWS, alpha=20.0, with_cf=True):
    d = data
    date = d.date.astype(np.int64)
    hz = window_horizons(date, windows)
    y = d.y_raw.astype(np.float64)
    labeled = np.ones(len(y), dtype=bool)      # frozen_prefix filters by horizon itself
    prior_mean = float(y[date <= 20220421].mean())

    vb = d.col('author_id', 'vb')
    v = d.video_id
    author = np.where(v < len(vb), vb[np.minimum(v, len(vb) - 1)], -1).astype(np.int64)
    tab = d.col('tab').astype(np.int64)
    dur = d.col('duration_ms').astype(np.float64)
    tr_mask = date <= 20220421
    edges = np.quantile(dur[tr_mask], np.linspace(0, 1, 11)[1:-1])
    durb = np.searchsorted(edges, dur).astype(np.int64)
    uid = d.user_id.astype(np.int64)

    def pair(a, b):
        """Collision-proof composite key.

        Arithmetic packing (uid * 1e7 + author) silently collides once a component exceeds
        its assumed range - author_id already reaches 8.73e6 against a 1e7 budget on
        KuaiRand-Pure, and the larger variants will blow through it with no error, just
        wrong features.  Factorising the pair costs one sort and cannot collide.
        """
        return np.unique(np.stack([a, b], 1), axis=0, return_inverse=True)[1].astype(np.int64)

    fams = {
        'item':        v.astype(np.int64),
        'author':      author,
        'user':        uid,
        'user_author': pair(uid, author),
        'user_item':   pair(uid, v.astype(np.int64)),
        'user_tab':    pair(uid, tab),
        'user_dur':    pair(uid, durb),
        'item_tab':    pair(v.astype(np.int64), tab),
    }
    cols, names = [], []
    for fam, keys in fams.items():
        l_, p_ = frozen_prefix(keys, date, y, labeled, hz)
        cols.append(smoothed_rate(p_, l_, prior_mean, alpha)); names.append(f'{fam}_rate')
        cols.append(np.log1p(l_));                             names.append(f'{fam}_logn')

    cols.append(np.log1p(dur));                    names.append('log_duration')
    cols.append(durb.astype(np.float64));          names.append('dur_bucket')
    cols.append(tab.astype(np.float64));           names.append('tab')
    cols.append(d.col('hourmin').astype(np.float64) // 100); names.append('hour')
    cols.append((_dayindex(date) - _dayindex(hz)).astype(np.float64))
    names.append('staleness_days')

    X = np.stack(cols, 1).astype(np.float32)
    if with_cf:
        Xc, nc = _frozen_cf(d, windows, hz)
        X = np.concatenate([X, Xc], 1); names = names + nc
    return X, names, hz


def _frozen_cf(d, windows, hz):
    """Item-item CF where BOTH the similarity matrix and the user's history are frozen."""
    from scipy.sparse import csr_matrix
    n_items = int(d.video_id.max()) + 1
    n_users = int(d.user_id.max()) + 1
    out = np.zeros((d.n, 3), dtype=np.float32)
    date = d.date.astype(np.int64)
    for h in np.unique(hz):
        rows = np.flatnonzero(hz == h)
        if h < 0 or len(rows) == 0:
            continue
        hist = np.flatnonzero((date <= h) & (d.y_raw == 1))
        if len(hist) < 100:
            continue
        M = csr_matrix((np.ones(len(hist), np.float32),
                        (d.user_id[hist].astype(np.int64), d.video_id[hist].astype(np.int64))),
                       shape=(n_users, n_items))
        M.data[:] = 1.0
        C = (M.T @ M).toarray().astype(np.float32)
        nrm = np.sqrt(np.maximum(np.diag(C), 1e-6))
        C /= nrm[:, None]; C /= nrm[None, :]; np.fill_diagonal(C, 0.0)
        Mu = M.tocsr()
        u = d.user_id[rows].astype(np.int64)
        it = d.video_id[rows].astype(np.int64)
        # score = sum of sim(this item, each item in the user's FROZEN history)
        sums = np.zeros(len(rows), np.float32)
        cnts = np.asarray(Mu[u].sum(1)).ravel().astype(np.float32)
        indptr, indices = Mu.indptr, Mu.indices
        for j in range(len(rows)):
            uu = u[j]
            hidx = indices[indptr[uu]:indptr[uu + 1]]
            if len(hidx):
                sums[j] = C[it[j], hidx].sum()
        out[rows, 0] = sums
        out[rows, 1] = sums / np.maximum(cnts, 1.0)
        out[rows, 2] = np.log1p(cnts)
        del C, M
    return out, ['cf_sum', 'cf_mean', 'cf_hist_logn']


def within_user_deviation(X, names, user_id, split_idx, window_id=None):
    """Deviation of each column from its mean over the same user's rows in the same WINDOW.

    Deviating within the whole split is wrong for training: the train split spans several
    frozen windows, so a per-window-constant statistic varies across them in training and
    is identically zero at evaluation time.  The model then learns splits on a feature
    that can never fire when it is served.  Grouping by (user, window) makes the training
    and evaluation feature distributions structurally identical.
    """
    Xs = X[split_idx]; u = user_id[split_idx].astype(np.int64)
    if window_id is not None:
        w = window_id[split_idx].astype(np.int64)
        uw, u = np.unique(np.stack([u, w], 1), axis=0, return_inverse=True)[0], None
        u = np.unique(np.stack([user_id[split_idx].astype(np.int64),
                                window_id[split_idx].astype(np.int64)], 1),
                      axis=0, return_inverse=True)[1].astype(np.int64)
    order = np.argsort(u, kind='stable')
    us = u[order]
    starts = np.flatnonzero(np.r_[True, us[1:] != us[:-1]])
    sizes = np.diff(np.r_[starts, len(us)])
    seg = np.repeat(np.arange(len(starts)), sizes)
    dev = np.empty_like(Xs)
    for c in range(Xs.shape[1]):
        vv = Xs[order, c].astype(np.float64)
        mean = np.bincount(seg, weights=vv) / sizes
        dev[order, c] = (vv - mean[seg]).astype(np.float32)
    return dev, [f'dev_{n}' for n in names]


def windows_for_fold(spec, n_train_chunks=5):
    """Build the frozen-window schedule for an arbitrary fold.

    Same rule everywhere: a row may use labels only up to the start of the window it
    belongs to.  Training is cut into chunks so that training rows carry a staleness
    profile resembling the one they will be served under, instead of always seeing history
    right up to themselves.
    """
    (tlo, thi), (vlo, vhi), (slo, shi) = spec['train'], spec['valid'], spec['test']
    days = sorted({int(x) for x in _daterange(tlo, thi)})
    if not days:
        return [(vlo, vhi, tlo - 1), (slo, shi, vhi)]
    chunks = np.array_split(np.array(days), min(n_train_chunks, len(days)))
    wins = []
    for i, ch in enumerate(chunks):
        lo, hi = int(ch[0]), int(ch[-1])
        hz = int(chunks[i - 1][-1]) if i > 0 else lo - 1
        wins.append((lo, hi, hz))
    wins.append((vlo, vhi, thi))     # validation frozen at end of train
    wins.append((slo, shi, vhi))     # test frozen at end of validation
    return wins


def _daterange(lo, hi):
    """Enumerate yyyymmdd values between lo and hi (April-May 2022 only)."""
    out = []
    y, m, dd = lo // 10000, (lo // 100) % 100, lo % 100
    days_in = {4: 30, 5: 31}
    cur = (m, dd)
    while True:
        v = y * 10000 + cur[0] * 100 + cur[1]
        if v > hi:
            break
        if v >= lo:
            out.append(v)
        nd = cur[1] + 1
        if nd > days_in.get(cur[0], 31):
            cur = (cur[0] + 1, 1)
        else:
            cur = (cur[0], nd)
        if cur[0] > 12:
            break
    return out
