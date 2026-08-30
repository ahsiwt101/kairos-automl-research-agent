"""The run log, and the structure the agent reasons over.

This is not paperwork.  Two of the five judging criteria are read directly out of it:
Autonomy is scored on the number of manual interventions, and Robustness on how failures
were handled - so the log is a deliverable, and it has to be machine-written per iteration.

It is also the agent's working memory.  Each entry records not just what was tried and
what happened, but what the agent PREDICTED would happen, which lets us score the agent's
causal understanding separately from its luck: a hypothesis family that keeps predicting
the right slice movement earns more of the remaining budget, even when individual gains
are small.
"""
import json, os, time, hashlib
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


@dataclass
class Hypothesis:
    statement: str                  # what we believe is wrong / recoverable
    mechanism: str                  # WHY it should work, in terms of the data
    predicted_effect: str           # which diagnostic slice should move, and which way
    predicted_gain: float           # expected primary delta, agent's own estimate
    family: str                     # e.g. 'objective', 'history', 'debias', 'ensemble'
    source: str = 'agent'           # 'agent' | 'diagnostic' | 'literature' | 'prior'
    prediction: dict = field(default_factory=dict)   # {diagnostic, direction}, scorable


@dataclass
class Outcome:
    valid_primary: float = float('nan')
    valid_gauc: float = float('nan')
    valid_ndcg: float = float('nan')
    delta_vs_incumbent: float = float('nan')
    prediction_hit: Optional[bool] = None   # did the predicted slice move as predicted?
    diagnostics: dict = field(default_factory=dict)
    seconds: float = 0.0


@dataclass
class Entry:
    iteration: int
    hypothesis: Hypothesis
    action_kind: str                       # 'patch' | 'hparam' | 'ensemble' | 'select'
    code_diff: str
    outcome: Outcome
    decision: str = 'pending'              # 'accept' | 'reject' | 'rollback' | 'crash'
    reason: str = ''
    errors: list = field(default_factory=list)
    budget: dict = field(default_factory=dict)
    interventions: int = 0                 # human touches during this iteration; target 0
    t: str = field(default_factory=lambda: time.strftime('%Y-%m-%d %H:%M:%S'))


class Ledger:
    def __init__(self, path='runs/ledger.jsonl', run_name='kairos', baseline=-1e9):
        self.path, self.run_name, self.baseline = path, run_name, baseline
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.entries = []
        self.tokens_in = self.tokens_out = 0
        self.t_start = time.time()
        self.interventions = 0

    def add(self, entry: Entry):
        entry.budget = {'tokens_in': self.tokens_in, 'tokens_out': self.tokens_out,
                        'wall_clock_s': round(time.time() - self.t_start, 1),
                        'iterations_used': len(self.entries) + 1}
        self.entries.append(entry)
        with open(self.path, 'a') as fh:
            fh.write(json.dumps(asdict(entry), default=float) + '\n')
        return entry

    def log_error(self, iteration, kind, detail, recovery):
        """Record a failure AND what the agent did about it. Robustness is scored on the
        recovery, explicitly not on the failure count."""
        rec = {'iteration': iteration, 'kind': kind, 'detail': str(detail)[:800],
               'recovery': recovery, 't': time.strftime('%H:%M:%S')}
        with open(self.path.replace('.jsonl', '_errors.jsonl'), 'a') as fh:
            fh.write(json.dumps(rec) + '\n')
        return rec

    # ---- what the scheduler reads ---------------------------------------
    def family_track_record(self):
        """Per hypothesis family: attempts, hit rate on its own predictions, mean gain.
        This is the bandit's prior - families that understand the problem get more budget."""
        out = {}
        for e in self.entries:
            f = e.hypothesis.family
            r = out.setdefault(f, {'n': 0, 'hits': 0, 'scored': 0, 'gain': 0.0,
                                   'best_gain': 0.0})
            r['n'] += 1
            g = e.outcome.delta_vs_incumbent
            if g == g:
                r['gain'] += g
                r['best_gain'] = max(r['best_gain'], g)
            if e.outcome.prediction_hit is not None:
                r['scored'] += 1
                r['hits'] += int(e.outcome.prediction_hit)
        for r in out.values():
            r['mean_gain'] = r['gain'] / max(r['n'], 1)
            r['hit_rate'] = r['hits'] / max(r['scored'], 1) if r['scored'] else None
        return out

    def stall_counter(self, eps=0.002):
        """Consecutive iterations without a > eps improvement in validation primary.

        The competition ends a run at 3.  This is therefore not a diagnostic but a hard
        resource: the scheduler must treat consecutive misses as a spendable budget.

        `best_so_far` is seeded from `self.baseline`, not from -inf. Task requirement #1
        is reproducing the official baseline before iterating, so the run's real starting
        point is the baseline's validation score, not an arbitrarily low floor. Seeding
        from -inf would let the agent earn "progress" credit for merely beating its own
        earlier bad guesses, which lets a run continue - and keep spending tokens - well
        past the point the stated rule (no +eps improvement in N consecutive iterations)
        would actually end it.
        """
        best_so_far, stall = self.baseline, 0
        for e in self.entries:
            p = e.outcome.valid_primary
            if p != p:
                stall += 1
                continue
            if p > best_so_far + eps:
                best_so_far, stall = max(best_so_far, p), 0
            else:
                best_so_far = max(best_so_far, p)
                stall += 1
        return stall

    def converged(self, eps=0.002, n=3, max_iters=50, max_seconds=6*3600):
        s = self.stall_counter(eps)
        if s >= n:
            return True, f'stalled: {s} consecutive iterations without +{eps} on validation'
        if len(self.entries) >= max_iters:
            return True, f'iteration cap {max_iters} reached'
        if time.time() - self.t_start >= max_seconds:
            return True, 'wall-clock ceiling reached'
        return False, ''

    def summary(self):
        conv, why = self.converged()
        return {'run': self.run_name, 'iterations': len(self.entries),
                'wall_clock_s': round(time.time() - self.t_start, 1),
                'tokens_in': self.tokens_in, 'tokens_out': self.tokens_out,
                'manual_interventions': self.interventions,
                'stall': self.stall_counter(), 'converged': conv, 'reason': why,
                'best_valid': max([e.outcome.valid_primary for e in self.entries
                                   if e.outcome.valid_primary == e.outcome.valid_primary],
                                  default=float('nan')),
                'families': self.family_track_record()}
