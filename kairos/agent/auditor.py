"""Temporal-validity auditor: catch validation illusions before they are believed.

Measured on this benchmark, the dominant failure mode is not a weak model but a lying
validation set.  A greedy agent drove validation 0.6016 -> 0.7158 while its hidden-test
score FELL 0.5946 -> 0.5749.  Every check here exists to make that failure loud.

The checks are structural where possible, because a heuristic threshold ("suspiciously
high") is itself a guess.  The strongest one is exact: within-user ranking is invariant to
any quantity constant across a user's list, so a USER-level statistic must have exactly
zero within-user variance.  Non-zero variance is a proof of within-window label feedback,
not an indication of it.
"""
import numpy as np
from kairos.kernel.fastmetrics import fast_evaluate, factorize


class AuditFinding:
    def __init__(self, check, severity, detail, value=None):
        self.check, self.severity, self.detail, self.value = check, severity, detail, value

    def __repr__(self):
        return f"[{self.severity}] {self.check}: {self.detail}"

    def as_dict(self):
        return {'check': self.check, 'severity': self.severity, 'detail': self.detail,
                'value': None if self.value is None else float(self.value)}


def _within_user_var(col, user_id):
    order = np.argsort(user_id, kind='stable')
    u = user_id[order]
    starts = np.flatnonzero(np.r_[True, u[1:] != u[:-1]])
    sizes = np.diff(np.r_[starts, len(u)])
    seg = np.repeat(np.arange(len(starts)), sizes)
    v = col[order].astype(np.float64)
    mean = np.bincount(seg, weights=v) / sizes
    return float(np.abs(v - mean[seg]).max())


class Auditor:
    """Runs on every candidate. Returns findings; BLOCK severity vetoes acceptance."""

    USER_LEVEL_PREFIXES = ('user_rate', 'user_logn')   # statistics of the user alone

    def __init__(self, data, fold):
        self.d, self.fold = data, fold

    # -- 1. structural: user-level features must be constant within a user's list ----
    def check_user_constancy(self, X, names, split='valid', tol=1e-6):
        idx = self.fold.idx[split]
        u = self.d.user_id[idx]
        out = []
        for j, n in enumerate(names):
            if n in self.USER_LEVEL_PREFIXES:
                mx = _within_user_var(X[idx, j], u)
                if mx > tol:
                    out.append(AuditFinding(
                        'user_constancy', 'BLOCK',
                        f"'{n}' varies within a user's {split} list (max dev {mx:.2e}). A "
                        f"user-level statistic can only vary if it absorbed labels from "
                        f"inside the evaluation window.", mx))
        return out

    # -- 2. structural: no feature may depend on a label past the horizon -----------
    def check_horizon(self, hz, split='test'):
        idx = self.fold.idx[split]
        bad = int((hz[idx] > self.fold.horizon).sum())
        if bad:
            return [AuditFinding('horizon', 'BLOCK',
                                 f"{bad} {split} rows use a horizon beyond {self.fold.horizon}",
                                 bad)]
        return []

    # -- 3. staleness parity between validation and serving ------------------------
    def check_staleness_parity(self, X, names, warn_days=3.0):
        if 'staleness_days' not in names:
            return [AuditFinding('staleness_parity', 'WARN',
                                 'no staleness column: the model cannot discount aged history')]
        j = names.index('staleness_days')
        sv = float(X[self.fold.idx['valid'], j].mean())
        st = float(X[self.fold.idx['test'], j].mean())
        if abs(sv - st) > warn_days:
            return [AuditFinding('staleness_parity', 'WARN',
                                 f"validation history averages {sv:.1f} days old but test "
                                 f"averages {st:.1f}; validation is being scored on an "
                                 f"easier feature distribution than it will serve under.",
                                 st - sv)]
        return []

    # -- 4. behavioural: label-shuffle probe ---------------------------------------
    def check_shuffle(self, fit_predict, seed=0, floor=0.52):
        """Refit on shuffled training labels. Any real pipeline must collapse to chance
        (primary ~= 0.475 here); anything above `floor` is reading the answer."""
        rng = np.random.default_rng(seed)
        s = fit_predict(shuffle_rng=rng)
        idx = self.fold.idx['valid']
        g, _ = factorize(self.d.user_id[idx])
        m = fast_evaluate(g, self.d.y_raw[idx], s)
        if m['primary'] > floor:
            return [AuditFinding('label_shuffle', 'BLOCK',
                                 f"primary {m['primary']:.4f} with SHUFFLED training labels "
                                 f"(chance is ~0.475). The pipeline reads the target.",
                                 m['primary'])]
        return [AuditFinding('label_shuffle', 'OK',
                             f"shuffled-label primary {m['primary']:.4f}, at chance",
                             m['primary'])]

    # -- 5. the empirical alarm: implausible validation gain ------------------------
    def check_gain_plausibility(self, valid_primary, incumbent, jump=0.02):
        """A single-iteration validation jump this large has, on this benchmark, always
        been leakage rather than skill. Flag for backtest confirmation, do not auto-reject."""
        if valid_primary - incumbent > jump:
            return [AuditFinding('gain_plausibility', 'VERIFY',
                                 f"validation jumped +{valid_primary-incumbent:.4f} in one "
                                 f"step; require backtest-fold confirmation before trusting",
                                 valid_primary - incumbent)]
        return []

    def run(self, X=None, names=None, hz=None, valid_primary=None, incumbent=None):
        f = []
        if X is not None and names is not None:
            f += self.check_user_constancy(X, names, 'valid')
            f += self.check_user_constancy(X, names, 'test')
            f += self.check_staleness_parity(X, names)
        if hz is not None:
            f += self.check_horizon(hz, 'test') + self.check_horizon(hz, 'valid')
        if valid_primary is not None and incumbent is not None:
            f += self.check_gain_plausibility(valid_primary, incumbent)
        return f

    @staticmethod
    def blocked(findings):
        return any(x.severity == 'BLOCK' for x in findings)
