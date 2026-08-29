"""Choosing what to submit, when validation is known to be unreliable.

The competition scores the validation-best checkpoint, once, on a hidden test set.  That
is an argmax over up to 50 correlated validation reads - a textbook winner's curse - and
on this benchmark we measured something worse than noise: validation gain is
ANTI-correlated with test gain once a pipeline starts exploiting within-window label
feedback (validation 0.6016 -> 0.7158 while test fell 0.5946 -> 0.5749).

So selection cannot be argmax over validation.  Instead each candidate is scored on
BACKTEST FOLDS - train/valid/test triples that live entirely inside the public-label
region.  Their test windows are genuine held-out future windows, so a candidate's mean
score across them estimates the thing we actually care about, generalisation across a
temporal gap, rather than fit to one particular week.

Three corrections stack:
  transfer   average over backtest folds instead of trusting the single official valid
  stability  penalise variance across folds; a candidate that wins one fold and loses
             another is being selected by noise
  shrinkage  the maximum of N noisy estimates is biased upward by roughly sigma*sqrt(2 ln N);
             subtract it so candidates are not rewarded merely for being numerous
"""
import math
import numpy as np


class Candidate:
    def __init__(self, name, spec, valid_primary=None):
        self.name, self.spec = name, spec
        self.valid_primary = valid_primary      # official-fold validation (reported, not trusted)
        self.fold_scores = {}                   # fold name -> held-out primary on that fold
        self.seed_scores = []                   # repeats, for a noise estimate
        self.audit = []

    @property
    def transfer(self):
        v = list(self.fold_scores.values())
        return float(np.mean(v)) if v else float('nan')

    @property
    def stability(self):
        v = list(self.fold_scores.values())
        return float(np.std(v)) if len(v) > 1 else 0.0

    @property
    def noise(self):
        return float(np.std(self.seed_scores)) if len(self.seed_scores) > 1 else 0.0

    def __repr__(self):
        return (f"<{self.name} valid={self.valid_primary:.4f} transfer={self.transfer:.4f} "
                f"+-{self.stability:.4f}>")


def selection_bias(n_candidates, sigma):
    """Expected upward bias of the max of n roughly-independent noisy estimates."""
    if n_candidates < 2 or sigma <= 0:
        return 0.0
    return sigma * math.sqrt(2.0 * math.log(n_candidates))


def robust_score(cand, n_candidates, lam=1.0, sigma_floor=0.0008):
    """The quantity the agent maximises instead of validation primary."""
    sigma = max(cand.noise, sigma_floor)          # FM's 5-seed std is 0.0008
    return cand.transfer - lam * cand.stability - selection_bias(n_candidates, sigma)


def select(candidates, lam=1.0, require_clean_audit=True):
    """Pick what to submit. Returns (winner, ranked_table)."""
    pool = [c for c in candidates
            if not (require_clean_audit and any(a.severity == 'BLOCK' for a in c.audit))]
    if not pool:
        raise RuntimeError("every candidate failed the temporal-validity audit")
    n = len(pool)
    ranked = sorted(pool, key=lambda c: -robust_score(c, n, lam))
    return ranked[0], [{'name': c.name,
                        'valid': c.valid_primary,
                        'transfer': c.transfer,
                        'stability': c.stability,
                        'robust': robust_score(c, n, lam)} for c in ranked]


def greedy_select(candidates):
    """The control arm: what a conventional agent does - argmax on validation."""
    return max(candidates, key=lambda c: (c.valid_primary if c.valid_primary is not None
                                          else -1e9))


def evaluate_selection_rule(candidates, truth_key, rule):
    """How much true score does a selection rule leave on the table?

    `truth_key` is a callable giving the candidate's honest held-out score (available for
    backtest folds, and exactly once for the official fold).  Returns the regret of the
    rule versus an oracle that could see the truth.
    """
    picked = rule(candidates)
    best = max(candidates, key=truth_key)
    return {'picked': picked.name, 'picked_true': truth_key(picked),
            'oracle': best.name, 'oracle_true': truth_key(best),
            'regret': truth_key(best) - truth_key(picked)}
