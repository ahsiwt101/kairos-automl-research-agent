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
        return self.data.col(name, table)

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
