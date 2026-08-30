"""Strictly causal prefix aggregates.

Every history feature here answers: "at the instant this impression was logged, what had
already been observed?"  Two independent hazards are handled:

  TARGET LEAKAGE - a training row must not see its own label, nor any label from a row
      logged after it.  Naively computing "this item's long_view rate" over the whole
      training window leaks the row into its own feature, and the model learns to trust a
      signal that does not exist at inference.  We use time-ordered prefix sums, so row r
      sees only rows strictly before it.

  HORIZON LEAKAGE - labels after the fold's horizon do not exist for us at all.  Exposure
      after the horizon IS observable (we are given the impression list to rank), so
      counts may include those rows while positives may not.  The two are tracked
      separately: `prior_n` counts exposures, `prior_labeled` counts rows whose label we
      are entitled to, and `prior_pos` sums only those labels.

The whole thing is one vectorised primitive - sort by (key, time), take within-key prefix
sums - so an arbitrary number of history features costs one lexsort each, not a Python loop.
"""
import numpy as np


def _as_key_array(keys, n, who):
    """Coerce whatever the caller passed into one 1-D grouping key of length n.

    The parameter is named `keys`, so passing a LIST of column arrays is a reasonable
    reading - and combining several columns into one grouping key is a thing callers
    genuinely want. Rather than reject it (numpy's own failure here is the useless
    "object too deep for desired array"), accept it and factorize the tuple, which is
    exactly what the caller meant. Collision-proof, unlike arithmetic packing.
    """
    if isinstance(keys, (list, tuple)) and len(keys) and not np.isscalar(keys[0]):
        arrs = [np.asarray(k).ravel() for k in keys]
        if len(arrs) == 1:
            keys = arrs[0]
        else:
            return np.unique(np.stack(arrs, 1), axis=0, return_inverse=True)[1].astype(np.int64)
    keys = np.asarray(keys)
    if keys.ndim == 2:                       # (n, k) or (k, n) column stack
        if keys.shape[0] != n and keys.shape[1] == n:
            keys = keys.T
        if keys.shape[1] == 1:
            keys = keys[:, 0]
        else:
            return np.unique(keys, axis=0, return_inverse=True)[1].astype(np.int64)
    keys = keys.ravel()
    if len(keys) != n:
        raise ValueError(f"{who}: `keys` has {len(keys)} entries but there are {n} rows; "
                         f"it must have exactly one entry per log row.")
    return keys


def causal_prefix(keys, time_ms, y, labeled, tiebreak=None):
    """For each row: counts over all EARLIER rows sharing the same key.

    keys      int64 (N,)   grouping key (item, or user*K+author, ...)
    time_ms   int64 (N,)   event time; ties broken by `tiebreak` (default: row index)
    y         (N,)         label, 0/1
    labeled   (N,) bool    whether this row's label is inside the horizon

    returns prior_n, prior_labeled, prior_pos  -- all (N,), aligned to the input order.
    """
    keys = _as_key_array(keys, len(np.asarray(time_ms)), 'causal_prefix')
    n = len(keys)
    tb = np.arange(n) if tiebreak is None else tiebreak
    order = np.lexsort((tb, time_ms, keys))
    k = keys[order]
    yl = (y * labeled)[order].astype(np.float64)
    lab = labeled[order].astype(np.float64)

    new = np.empty(n, dtype=bool)
    new[0] = True
    np.not_equal(k[1:], k[:-1], out=new[1:])
    starts = np.flatnonzero(new)
    seg = np.repeat(np.arange(len(starts)), np.diff(np.append(starts, n)))

    pos_in_seg = np.arange(n) - starts[seg]                      # 0-based index within key
    cum_y = np.cumsum(yl); cum_l = np.cumsum(lab)
    base_y = np.where(starts[seg] > 0, cum_y[starts[seg] - 1], 0.0)
    base_l = np.where(starts[seg] > 0, cum_l[starts[seg] - 1], 0.0)
    prior_pos = cum_y - yl - base_y                              # exclude self
    prior_lab = cum_l - lab - base_l

    out_n = np.empty(n); out_l = np.empty(n); out_p = np.empty(n)
    out_n[order] = pos_in_seg
    out_l[order] = prior_lab
    out_p[order] = prior_pos
    return out_n, out_l, out_p


def smoothed_rate(prior_pos, prior_labeled, prior_mean, alpha=20.0):
    """Beta-smoothed rate; shrinks to the global prior when history is thin."""
    return (prior_pos + alpha * prior_mean) / (prior_labeled + alpha)


class CausalFeatureBuilder:
    """Builds a table of causal history features over ALL rows at once.

    Because everything is prefix-based, train / valid / test rows can be processed in a
    single pass: each row only ever sees its own past.
    """

    def __init__(self, data, horizon):
        self.d = data
        self.horizon = int(horizon)
        self.labeled = (data.date <= self.horizon)
        self.y = data.y_raw.astype(np.float64)
        self.t = data.time_ms
        self.prior_mean = float(self.y[self.labeled].mean())
        self._cache = {}

    def _key_stats(self, name, keys, alpha=20.0):
        if name not in self._cache:
            n_, l_, p_ = causal_prefix(keys, self.t, self.y, self.labeled)
            self._cache[name] = (n_, l_, p_,
                                 smoothed_rate(p_, l_, self.prior_mean, alpha))
        return self._cache[name]

    # ---- individual feature families -------------------------------------
    def item(self):
        """Item's long_view rate among impressions logged before this one."""
        return self._key_stats('item', self.d.video_id.astype(np.int64))

    def author(self):
        vb = self.d.col('author_id', 'vb')
        v = self.d.video_id
        a = np.where(v < len(vb), vb[np.minimum(v, len(vb) - 1)], -1).astype(np.int64)
        self.author_id = a
        return self._key_stats('author', a)

    def user(self):
        return self._key_stats('user', self.d.user_id.astype(np.int64))

    def user_author(self):
        """Has this user engaged with this author before?  A user-varying AND
        item-varying signal, so unlike a pure user statistic it survives within-user
        ranking (constant-per-user terms cancel out of the metric entirely)."""
        if not hasattr(self, 'author_id'):
            self.author()
        return self._key_stats('user_author',
                               self.d.user_id.astype(np.int64) * 10_000_000 + self.author_id)

    def user_item(self):
        """Repeat exposure: the eval splits contain repeated (user, video) pairs."""
        return self._key_stats('user_item',
                               self.d.user_id.astype(np.int64) * 100_000 + self.d.video_id)

    def user_durbucket(self, edges):
        """User's historical tolerance for videos of this length.

        `long_view` is definitionally duration-dependent, and users differ in how much
        length they will sit through, so this cross carries the duration confound at the
        level where it actually acts."""
        db = np.searchsorted(edges, self.d.col('duration_ms').astype(np.float64))
        self.dur_bucket = db
        return self._key_stats('user_dur',
                               self.d.user_id.astype(np.int64) * 100 + db.astype(np.int64))

    def user_tab(self):
        return self._key_stats('user_tab',
                               self.d.user_id.astype(np.int64) * 100
                               + self.d.col('tab').astype(np.int64))


# ---------------------------------------------------------------------------
# Window-frozen aggregates
# ---------------------------------------------------------------------------
def frozen_prefix(keys, date, y, labeled, horizon_per_row):
    """Per-row aggregates over all rows of the same key dated <= that ROW'S OWN horizon.

    `causal_prefix` above answers "what had happened before this instant", which is
    correct for a streaming system but WRONG as a model of this task.  Evaluation ranks a
    user's whole impression list as a set, and the labels of the other rows in that list
    do not exist at scoring time.  Worse, they are precisely what the metric asks us to
    predict, so letting a row see its list-mates' labels inflates validation enormously
    while being unavailable on the hidden test - it corrupts model selection.

    The faithful construction freezes history at the START of each evaluation window, the
    way a periodically-retrained production model actually sees the world.  Passing a
    per-row horizon lets train, valid and test all be built under one rule.
    """
    date = np.asarray(date).astype(np.int64)
    keys = _as_key_array(keys, len(date), 'frozen_prefix')
    n = len(keys)
    uk, kid = np.unique(keys, return_inverse=True)
    kid = kid.astype(np.int64)
    SC = np.int64(100_000_000)

    comb = kid * SC + date
    uc, inv = np.unique(comb, return_inverse=True)
    lab = np.bincount(inv, weights=labeled.astype(np.float64), minlength=len(uc))
    pos = np.bincount(inv, weights=(y * labeled).astype(np.float64), minlength=len(uc))

    uc_kid = uc // SC
    seg_start = np.searchsorted(uc_kid, uc_kid, side='left')      # first index of each kid
    cl = np.cumsum(lab); cp = np.cumsum(pos)
    base_l = np.where(seg_start > 0, cl[seg_start - 1], 0.0)
    base_p = np.where(seg_start > 0, cp[seg_start - 1], 0.0)
    cum_l = cl - base_l                                            # cumulative within kid
    cum_p = cp - base_p

    tgt = kid * SC + horizon_per_row.astype(np.int64)
    j = np.searchsorted(uc, tgt, side='right') - 1
    own = (j >= 0) & (uc_kid[np.clip(j, 0, len(uc) - 1)] == kid)
    jj = np.clip(j, 0, len(uc) - 1)
    out_l = np.where(own, cum_l[jj], 0.0)
    out_p = np.where(own, cum_p[jj], 0.0)
    return out_l, out_p


def window_horizons(date, windows):
    """Map each row to the horizon of the window it belongs to.

    windows: list of (lo, hi, horizon).  A row dated in [lo, hi] may use labels <= horizon.
    Rows outside every window get horizon = -1 (no history), never a peek forward.
    """
    h = np.full(len(date), -1, dtype=np.int64)
    for lo, hi, hz in windows:
        m = (date >= lo) & (date <= hi)
        h[m] = hz
    return h


def frozen_prefix_decayed(keys, date, y, labeled, horizon_per_row, halflife_days=7.0):
    """Like frozen_prefix, but weights older evidence less (exponential half-life).

    Item quality drifts - a video that performed well two weeks ago is weaker evidence
    about today than one that performed well yesterday - but frozen_prefix weights every
    observation inside the horizon equally.

    The trick that makes this one cumsum rather than a per-row scan: the weighted sum for
    a row with horizon h is
        S(h) = sum_{d<=h} n_d * 2^{-(h-d)/H}  =  2^{-h/H} * sum_{d<=h} n_d * 2^{d/H}
    so pre-multiplying each day's count by 2^{d/H} makes the inner term an ordinary
    prefix sum over days, and the per-row factor 2^{-h/H} is applied afterwards.
    """
    date = np.asarray(date).astype(np.int64)
    keys = _as_key_array(keys, len(date), 'frozen_prefix_decayed')
    n = len(keys)
    uk, kid = np.unique(keys, return_inverse=True)
    kid = kid.astype(np.int64)
    SC = np.int64(100_000_000)

    # day index (compact, monotone) so the exponent stays small and well-conditioned
    ud, dinv = np.unique(date, return_inverse=True)
    day_idx = _dayindex(ud)
    day_idx = day_idx - day_idx.min()
    w_day = np.exp2(day_idx / float(halflife_days))          # 2^{d/H}, per distinct day

    comb = kid * SC + date
    uc, inv = np.unique(comb, return_inverse=True)
    wrow = w_day[dinv]
    lab = np.bincount(inv, weights=(labeled * wrow).astype(np.float64), minlength=len(uc))
    pos = np.bincount(inv, weights=(y * labeled * wrow).astype(np.float64), minlength=len(uc))

    uc_kid = uc // SC
    seg_start = np.searchsorted(uc_kid, uc_kid, side='left')
    cl = np.cumsum(lab); cp = np.cumsum(pos)
    base_l = np.where(seg_start > 0, cl[seg_start - 1], 0.0)
    base_p = np.where(seg_start > 0, cp[seg_start - 1], 0.0)
    cum_l = cl - base_l
    cum_p = cp - base_p

    tgt = kid * SC + horizon_per_row.astype(np.int64)
    j = np.searchsorted(uc, tgt, side='right') - 1
    own = (j >= 0) & (uc_kid[np.clip(j, 0, len(uc) - 1)] == kid)
    jj = np.clip(j, 0, len(uc) - 1)
    # rescale by 2^{-h/H} for each row's own horizon
    hz_day = _dayindex(horizon_per_row.astype(np.int64)) - _dayindex(ud).min()
    scale = np.exp2(-hz_day / float(halflife_days))
    out_l = np.where(own, cum_l[jj] * scale, 0.0)
    out_p = np.where(own, cum_p[jj] * scale, 0.0)
    return out_l, out_p


def _dayindex(d):
    d = np.asarray(d).astype(np.int64)
    return (d // 10000) * 372 + ((d // 100) % 100) * 31 + (d % 100)


def hierarchical_rate(child_pos, child_n, parent_rate, alpha=20.0):
    """Empirical-Bayes shrinkage toward a PARENT rate instead of the global mean.

    For an item with 5 impressions, the global average (0.33) is a poor prior; that item's
    AUTHOR's rate, estimated over hundreds of impressions, is a much better one. Standard
    fixed-alpha smoothing throws that structure away.
    """
    return (child_pos + alpha * parent_rate) / (child_n + alpha)
