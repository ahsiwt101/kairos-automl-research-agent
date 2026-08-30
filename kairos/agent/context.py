"""The surface an LLM-authored candidate is allowed to touch.

Deliberately includes `causal_prefix`, which is the WRONG primitive for this task even
though it looks right.  The agent is permitted to make that mistake: the auditor catches
it from the output, and the recovery is the interesting behaviour to demonstrate.  Hiding
the footgun would hide the result.

`ctx.col()` deliberately does NOT expose every log column, though. Columns like `is_click`,
`is_like`, `play_time_ms` are OUTCOMES of the impression, not context known before it is
served - and long_view is itself a near-deterministic function of play_time_ms (a perfect
watch-ratio predictor scores 0.80 primary; see exp09). Our replay harness happens to have
these columns for every row including valid/test because it is historical logs, but a real
ranking model never has this row's own outcome at scoring time. Exposing them raw would let
a candidate "discover" a spectacular result that is pure leakage and that none of the
existing structural checks would catch (it is neither user-constant nor future-dated).
They are reachable only through auxiliary_signal(), which time-gates them exactly like
frozen_prefix does for the scored label.
"""
import numpy as np
from kairos.kernel.dataset import Data
from kairos.kernel import causal, frozenfeat

# Pre-exposure: known before the item is served, safe as a same-row feature.
_SAFE_RAW_COLUMNS = {'tab', 'duration_ms', 'hourmin', 'date', 'time_ms', 'is_rand',
                     'user_id', 'video_id'}
# Outcomes of the impression - only reachable through time-gated aggregation.
_OUTCOME_COLUMNS = {'is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward',
                    'is_hate', 'long_view', 'is_profile_enter', 'play_time_ms',
                    'profile_stay_time', 'comment_stay_time'}


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
        self._mf = {}
        self._aux = {}

    # convenience accessors so candidates need no knowledge of the cache layout
    def col(self, name, table='log'):
        """A raw column. NOTE the shapes differ by table:
             table='log' -> one entry per LOG ROW      (len == ctx.data.n)
             table='vb'  -> one entry per VIDEO        (len == n_videos)
             table='uf'  -> one entry per USER         (len == n_users)
        Use video_attr() / user_attr() to get the side tables broadcast to log rows.

        Raises for outcome columns (is_click, is_like, play_time_ms, ...): this row's own
        outcome is not known before the item is served, so it cannot be a feature for
        predicting this row's own label. Use auxiliary_signal(name) for a leakage-safe,
        out-of-sample HISTORICAL aggregate of the same signal.
        """
        if table == 'log' and name in _OUTCOME_COLUMNS:
            raise ValueError(
                f"ctx.col('{name}'): this is an OUTCOME of the impression, not something "
                f"known before it is served. Using this row's own {name} to predict this "
                f"row's own long_view leaks the answer (long_view is itself close to a "
                f"deterministic function of play_time_ms). Use "
                f"ctx.auxiliary_signal('{name}') instead - a properly time-gated, "
                f"out-of-sample historical rate, exactly like ctx.baseline_score.")
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

    def mf_factors(self, dim=16):
        """Leakage-safe implicit-feedback matrix-factorization embeddings, per row.

        Returns (U, V): float32 (n, dim) each. dot(U[i], V[i]) is a personalised
        collaborative-filtering score for row i; the raw vectors can also be crossed with
        other features or fed to the tree dimension-by-dimension. This is a DIFFERENT
        inductive bias than the FM's pointwise user_id x video_id embeddings (a low-rank
        factorization of the whole interaction matrix at once, useful precisely because
        train is only 0.58% dense), so it is complementary rather than redundant with
        ctx.baseline_score - combining the two is more promising than either alone.

        Trained per frozen window on positives dated at/before that window's horizon;
        rows for users/items with no prior history are exactly zero (cold start), never a
        leak. Cold coverage on this data is ~65-75% non-zero; treat the rest as missing.
        """
        if dim not in self._mf:
            from kairos.kernel.mf_signal import build_mf_factors
            self._mf[dim] = build_mf_factors(self.data, self.OFFICIAL_WINDOWS, dim=dim)
        return self._mf[dim]

    def auxiliary_signal(self, name):
        """Out-of-sample propensity for an auxiliary feedback signal.

        name in {'is_click','is_like','is_follow','is_comment','is_forward'}. This is the
        legitimate route to these signals - ctx.col(name) is blocked because a row's own
        outcome cannot be a feature for its own label, but a model's out-of-sample belief
        about the row (trained on strictly earlier data, exactly like ctx.baseline_score)
        is not leakage and may carry information long_view's own history misses.
        """
        if name not in self._aux:
            from kairos.kernel.baseline_signal import build_auxiliary_signal
            self._aux[name] = build_auxiliary_signal(self.data, self.OFFICIAL_WINDOWS, name)
        return self._aux[name]

    def labels_visible(self):
        """Labels with anything past the fold horizon replaced by -1."""
        return self.fold.vault.visible()


def make_context(fold_name='official'):
    return Context(fold_name)
