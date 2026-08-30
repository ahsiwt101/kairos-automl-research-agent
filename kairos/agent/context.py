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
from kairos.kernel.dataset import Data, FOLDS
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
        self.fold_name = fold_name
        self.data = Data()
        self.fold = self.data.fold(fold_name)
        self.causal_prefix = causal.causal_prefix
        self.frozen_prefix = causal.frozen_prefix
        self.window_horizons = causal.window_horizons
        self.smoothed_rate = causal.smoothed_rate
        # This fold's OWN window schedule - NOT always the official one. The name
        # OFFICIAL_WINDOWS is kept for backward compatibility with earlier documented
        # usage; it is aliased to self.windows, which is correct for whichever fold this
        # Context was built for. Getting this wrong would matter a lot: baseline_score /
        # mf_factors / cf_score / auxiliary_signal below all derive from it, and if they
        # silently used the OFFICIAL schedule while running against a backtest fold, they
        # would train on more recent data than that fold's own protocol allows - exactly
        # the kind of fold-contamination bug that would corrupt a backtest confirmation.
        self.windows = frozenfeat.windows_for_fold(FOLDS[fold_name])
        self.OFFICIAL_WINDOWS = self.windows
        self.within_user_deviation = frozenfeat.within_user_deviation
        self._fm = None
        self._mf = {}
        self._aux = {}
        self._cf = None
        self._refit = None
        self._din = None
        self._expert = {}

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
            self._fm = build_fm_signal(self.data, self.windows,
                                       cache=f'runs/fm_signal_{self.fold_name}.npy')
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
            self._mf[dim] = build_mf_factors(
                self.data, self.windows, dim=dim,
                cache_dir=f'runs/mf_cache_{self.fold_name}')
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
            self._aux[name] = build_auxiliary_signal(
                self.data, self.windows, name,
                cache_dir=f'runs/aux_cache_{self.fold_name}')
        return self._aux[name]

    def cf_score(self):
        """IDF-weighted item-item collaborative filtering, per row.

        Returns (score, hist_count): float32 (n,) each. score is the mean similarity
        between the row's candidate item and the items in THIS USER's frozen history;
        hist_count is how many history items that mean was taken over (0 = cold start,
        usable as a confidence weight - a mean over 1 item is much noisier than over 50).

        Similarity is IDF-weighted co-occurrence (Breese/Heckerman/Kadie): two users who
        both watched an obscure video is stronger evidence of taste overlap than two users
        who both watched a blockbuster, so raw co-occurrence undervalues the former.

        Leakage-safe per frozen window, same construction as mf_factors and baseline_score.
        """
        if self._cf is None:
            from kairos.kernel.cf_signal import build_cf_score
            self._cf = build_cf_score(self.data, self.windows,
                                      cache_dir=f'runs/cf_cache_{self.fold_name}')
        return self._cf

    def refit_score(self):
        """The baseline FM, fit with the best data available FOR EACH SPLIT.

        Same model as ctx.baseline_score, but train/valid rows are scored by a model
        trained on TRAIN ONLY while test rows are scored by one refit on TRAIN+VALIDATION.
        Refitting on validation is worth about +0.002 (confirmed on two independent
        backtest folds), and resolving the split asymmetry inside this primitive means you
        get that benefit automatically: weights you fit on validation are fitted against
        honest train-only scores, while the test predictions you finally emit carry the
        refit. No test label is used anywhere.

        Prefer this over ctx.baseline_score when you want the FM's information.
        """
        if self._refit is None:
            from kairos.kernel.refit_signal import build_refit_signal
            self._refit = build_refit_signal(self.data, self.fold)
        return self._refit

    def din_score(self):
        """Out-of-sample DIN sequence-model score per row.

        Target attention over the items this user actually long-viewed - a different
        inductive bias from the FM's per-ID crosses and from aggregate history rates.
        Measured standalone at valid 0.6014, level with the FM (0.6017) from an unrelated
        architecture, which is what makes it useful to combine with rather than a weaker
        copy of what you already have.
        """
        if self._din is None:
            from kairos.kernel.din_signal import build_din_signal
            from kairos.kernel.causal import window_horizons
            import numpy as _np
            hz = window_horizons(self.data.date.astype(_np.int64), self.windows)
            self._din = build_din_signal(self.data, self.fold, hz)
        return self._din

    def expert_score(self, subspace):
        """A model trained on ONE disjoint feature family. subspace in
        {'context', 'item', 'user'}.

            context   tab, hour, duration, staleness    (no item or user identity)
            item      item / author / item x tab rates  (no user information)
            user      user x tab, user x duration rates (no item identity)

        Individually weak - context 0.5718, item 0.5906, user 0.5357 on validation, all
        below the FM's 0.6005. That is the point. Fusion is rewarded by DECORRELATION, not
        by member strength, and these are far more decorrelated than the strong models are
        from each other: mean pairwise Spearman +0.362 between experts, against +0.848
        between the FM and DIN (which are unrelated architectures that nonetheless both
        converge on item quality). An expert that cannot see item identity cannot
        rediscover item quality, so its errors are distributed differently.

        Blending several of these is likely to beat blending two strong, correlated models.
        """
        if subspace not in self._expert:
            from kairos.kernel.expert_signal import build_expert_signal
            from kairos.kernel.causal import window_horizons
            import numpy as _np
            hz = window_horizons(self.data.date.astype(_np.int64), self.windows)
            self._expert[subspace] = build_expert_signal(self.data, self.fold, hz, subspace)
        return self._expert[subspace]

    def labels_visible(self):
        """Labels with anything past the fold horizon replaced by -1."""
        return self.fold.vault.visible()


def make_context(fold_name='official'):
    return Context(fold_name)
