"""The surface an LLM-authored candidate is allowed to touch.

Deliberately includes `causal_prefix`, which is the WRONG primitive for this task even
though it looks right.  The agent is permitted to make that mistake: the auditor catches
it from the output, and the recovery is the interesting behaviour to demonstrate.  Hiding
the footgun would hide the result.
"""
import numpy as np
from kairos.kernel.dataset import Data
from kairos.kernel import causal, frozenfeat


class Context:
    def __init__(self, fold_name='official'):
        self.data = Data()
        self.fold = self.data.fold(fold_name)
        self.causal_prefix = causal.causal_prefix
        self.frozen_prefix = causal.frozen_prefix
        self.window_horizons = causal.window_horizons
        self.smoothed_rate = causal.smoothed_rate
        self.OFFICIAL_WINDOWS = frozenfeat.OFFICIAL_WINDOWS
        self.within_user_deviation = frozenfeat.within_user_deviation
        self._fm = None

    # convenience accessors so candidates need no knowledge of the cache layout
    def col(self, name, table='log'):
        """A raw column. NOTE the shapes differ by table:
             table='log' -> one entry per LOG ROW      (len == ctx.data.n)
             table='vb'  -> one entry per VIDEO        (len == n_videos)
             table='uf'  -> one entry per USER         (len == n_users)
        Use video_attr() / user_attr() to get the side tables broadcast to log rows."""
        return self.data.col(name, table)

    def video_attr(self, name):
        """A video-side attribute broadcast to one entry PER LOG ROW.

        `col(name, 'vb')` is indexed by video_id and has ~7.5k entries, while build() must
        return arrays of length ctx.data.n (~1.44M). Mixing the two is the single most
        common way a candidate dies, so this does the gather for you.
        """
        v = self.data.col(name, 'vb')
        vid = self.data.video_id
        out = np.zeros(self.data.n, dtype=np.asarray(v).dtype)
        ok = vid < len(v)
        out[ok] = np.asarray(v)[vid[ok]]
        return out

    def user_attr(self, name):
        """A user-side attribute broadcast to one entry PER LOG ROW.

        Careful: a pure user attribute is CONSTANT across a user's evaluation list, and
        within-user ranking is invariant to per-user constants - so on its own it cannot
        change the metric at all. It only does work when crossed with something
        item-varying.
        """
        u = self.data.col(name, 'uf')
        uid = self.data.user_id
        out = np.zeros(self.data.n, dtype=np.asarray(u).dtype)
        ok = uid < len(u)
        out[ok] = np.asarray(u)[uid[ok]]
        return out

    def check(self, X, names):
        """Validate a candidate matrix against the harness contract.

        Call this before returning. It raises exactly what the harness would, so a shape
        or NaN mistake surfaces inside your own code - where you can still fix it - rather
        than one wasted iteration later.
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"build() must return a 2-D matrix, got shape {X.shape}")
        if X.shape[0] != self.data.n:
            raise ValueError(
                f"build() returned {X.shape[0]} rows but the log has {self.data.n}. Every "
                f"column must be one value PER LOG ROW. A video-side column has "
                f"~{len(self.data.col('author_id', 'vb'))} entries - broadcast it with "
                f"ctx.video_attr(name) instead of reshaping.")
        if len(names) != X.shape[1]:
            raise ValueError(f"{len(names)} names for {X.shape[1]} columns")
        if not np.isfinite(X).all():
            bad = [names[j] for j in range(X.shape[1])
                   if not np.isfinite(X[:, j]).all()]
            raise ValueError(f"non-finite values in columns {bad}; guard divisions with "
                             f"np.maximum(denom, eps)")
        return X

    @property
    def train_idx(self):
        return self.fold.idx['train']

    @property
    def valid_idx(self):
        return self.fold.idx['valid']

    @property
    def baseline_score(self):
        """Out-of-sample FM score per row - the official baseline's own prediction.

        Trained per frozen window on data at or before that window's horizon, so it never
        scores a row it was fit on and never uses a label past that row's horizon. Use it
        as a feature to build on rather than trying to rediscover it."""
        if self._fm is None:
            from kairos.kernel.baseline_signal import build_fm_signal
            self._fm = build_fm_signal(self.data, self.OFFICIAL_WINDOWS)
        return self._fm

    def labels_visible(self):
        """Labels with anything past the fold horizon replaced by -1."""
        return self.fold.vault.visible()


def make_context(fold_name='official'):
    return Context(fold_name)
