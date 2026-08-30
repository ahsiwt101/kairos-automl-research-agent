"""KAIROS - the controller.

Named for the sense of "the right moment": the run is governed less by what to try than by
WHEN, because the competition ends a run after 3 consecutive iterations without a +0.002
validation gain.  That makes consecutive misses a spendable resource rather than a
diagnostic, and it is the constraint most agent designs will not model at all.

The loop:
    diagnose -> propose -> sandbox -> AUDIT -> evaluate (multi-seed) -> score the
    prediction -> accept/reject on a robust score -> update budget -> check convergence

The audit step is the one that distinguishes this from a conventional agent.  On this
benchmark a greedy validation-following agent reaches validation 0.7158 with a hidden-test
score of 0.5749 - below the baseline it was trying to beat.  Catching that, and recovering
from it autonomously, is the behaviour being demonstrated.
"""
import json, os, subprocess, sys, time
import numpy as np

from kairos.agent.ledger import Ledger, Entry, Hypothesis, Outcome
from kairos.agent.auditor import Auditor
from kairos.agent.sandbox import run_candidate
from kairos.kernel.dataset import Data
from kairos.kernel.causal import window_horizons
from kairos.kernel.frozenfeat import OFFICIAL_WINDOWS


class Kairos:
    def __init__(self, proposer, fold_name='official', workdir='runs/kairos',
                 eps=0.002, stall_limit=3, max_iters=50, max_seconds=6 * 3600,
                 seeds=(0, 1, 2), repair_attempts=2, python=None, audit_enabled=True,
                 max_tokens_total=400_000):
        self.proposer = proposer
        self.fold_name = fold_name
        self.workdir = workdir
        self.eps, self.stall_limit = eps, stall_limit
        self.max_iters, self.max_seconds = max_iters, max_seconds
        self.seeds, self.repair_attempts = seeds, repair_attempts
        # audit_enabled=False reproduces a conventional agent: write code, train, follow
        # the validation number.  It is the control arm for the ablation, not a fallback.
        self.audit_enabled = audit_enabled
        # Hard spend guard.  The run is expected to use ~30k tokens; this ceiling exists so
        # a pathological repair loop cannot quietly run up a bill, and it is reported in the
        # ledger either way because token usage is a scored criterion.
        self.max_tokens_total = max_tokens_total
        self.python = python or sys.executable
        os.makedirs(workdir, exist_ok=True)
        self.data = Data()
        self.fold = self.data.fold(fold_name)
        self.auditor = Auditor(self.data, self.fold)
        self.hz = window_horizons(self.data.date.astype(np.int64), OFFICIAL_WINDOWS)
        np.save(os.path.join(workdir, 'hz.npy'), self.hz)
        self.baseline_valid = 0.6016
        self.ledger = Ledger(path=os.path.join(workdir, 'ledger.jsonl'),
                             baseline=self.baseline_valid)
        self.incumbent = None          # dict with X_path, names, valid_primary

    # ------------------------------------------------------------------ budget
    def budget(self):
        used = len(self.ledger.entries)
        stall = self.ledger.stall_counter(self.eps)
        return {'iterations_used': used, 'iterations_left': self.max_iters - used,
                'stall_counter': stall,
                'misses_before_run_ends': max(self.stall_limit - stall, 0),
                'wall_clock_s': round(time.time() - self.ledger.t_start, 1),
                'seconds_left': round(self.max_seconds - (time.time() - self.ledger.t_start), 1),
                'tokens_in': self.proposer.tokens_in, 'tokens_out': self.proposer.tokens_out,
                'note': ('ONE miss from termination - prefer a change with high probability '
                         'of a small gain over an exploratory one'
                         if self.stall_limit - stall <= 1 else 'exploration affordable')}

    # ------------------------------------------------------------------ evaluation
    def _evaluate(self, X_path, train_cfg=None):
        cfg = {'X_path': os.path.abspath(X_path), 'fold': self.fold_name,
               'train_cfg': train_cfg or {},
               'seeds': list(self.seeds), 'add_dev': True,
               'hz_path': os.path.abspath(os.path.join(self.workdir, 'hz.npy')),
               'out': os.path.abspath(os.path.join(self.workdir, 'eval.json'))}
        p = os.path.join(self.workdir, 'evalcfg.json')
        json.dump(cfg, open(p, 'w'))
        r = subprocess.run([self.python, 'kairos/agent/evaluate_candidate.py', p],
                           capture_output=True, text=True, timeout=3600)
        if r.returncode != 0:
            return None, '\n'.join(r.stderr.strip().splitlines()[-10:])
        return json.load(open(cfg['out'])), None

    # ------------------------------------------------------------------ one iteration
    def step(self, n):
        from kairos.kernel.diagnostics import Diagnostics
        digest = self.incumbent['digest'] if self.incumbent else {
            'primary': self.baseline_valid, 'note': 'FM baseline, no diagnostics yet'}
        last_failure = None
        prev_hyp = None
        for attempt in range(self.repair_attempts + 1):
            # A failed candidate does not invalidate the hypothesis - only the code. If the
            # proposer can repair, re-invoke the coder alone and keep the plan.
            if attempt and last_failure and hasattr(self.proposer, 'repair'):
                prop = self.proposer.repair(prev_hyp, last_failure)
            else:
                prop = self.proposer.propose(digest, self.ledger.summary(), self.budget(),
                                             last_failure)
            prev_hyp = prop.hypothesis
            hyp = Hypothesis(**prop.hypothesis)
            res = run_candidate(prop.code, self.fold_name,
                                workdir=os.path.join(self.workdir, f'cand{n}'))
            if not res['ok']:
                last_failure = {'stage': res['stage'], 'error': res['error'],
                                'hint': res.get('hint', '')}
                self.ledger.log_error(n, res['stage'], res['error'],
                                      f'fed the error back to the proposer '
                                      f'(repair attempt {attempt+1}/{self.repair_attempts})')
                continue

            X = np.load(res['X_path'])
            findings = (self.auditor.run(X=X, names=res['names'], hz=self.hz)
                        if self.audit_enabled else [])
            if Auditor.blocked(findings):
                detail = '; '.join(f.detail for f in findings if f.severity == 'BLOCK')
                last_failure = {'stage': 'audit', 'error': detail,
                                'hint': 'features must be frozen at the evaluation '
                                        'window start, not accumulated across it'}
                self.ledger.log_error(n, 'temporal_validity', detail,
                                      'rejected before it could be believed; the specific '
                                      'violation was returned to the proposer for rewrite')
                continue
            return prop, hyp, res, findings, last_failure, attempt
        return prop, Hypothesis(**prop.hypothesis), None, [], last_failure, self.repair_attempts

    def run(self, verbose=True):
        n = 0
        while True:
            spent = self.proposer.tokens_in + self.proposer.tokens_out
            if spent >= self.max_tokens_total:
                if verbose:
                    print(f"\nSTOPPED: token budget exhausted ({spent:,} >= "
                          f"{self.max_tokens_total:,})")
                break
            done, why = self.ledger.converged(self.eps, self.stall_limit,
                                              self.max_iters, self.max_seconds)
            if done:
                if verbose: print(f"\nCONVERGED: {why}")
                break
            n += 1
            t0 = time.time()
            # keep the ledger's budget view in sync with the proposer's real usage;
            # token consumption is a scored criterion, so it must be measured, not guessed
            self.ledger.tokens_in = self.proposer.tokens_in
            self.ledger.tokens_out = self.proposer.tokens_out
            prop, hyp, res, findings, failure, attempts = self.step(n)
            if res is None:
                self.ledger.add(Entry(iteration=n, hypothesis=hyp, action_kind='patch',
                                      code_diff=prop.code, outcome=Outcome(seconds=time.time()-t0),
                                      decision='rollback',
                                      reason=f"abandoned after {attempts} repair attempts: "
                                             f"{(failure or {}).get('error','')[:200]}",
                                      errors=[failure] if failure else []))
                if verbose: print(f"[{n}] ROLLBACK  {hyp.family:9s} {hyp.statement[:60]}")
                continue

            self.ledger.tokens_in = self.proposer.tokens_in
            self.ledger.tokens_out = self.proposer.tokens_out
            ev, err = self._evaluate(res['X_path'], res.get('train_cfg'))
            if ev is None:
                self.ledger.log_error(n, 'train', err, 'candidate discarded, run continues')
                self.ledger.add(Entry(iteration=n, hypothesis=hyp, action_kind='patch',
                                      code_diff=prop.code, outcome=Outcome(seconds=time.time()-t0),
                                      decision='crash', reason=err[:200], errors=[{'train': err}]))
                if verbose: print(f"[{n}] CRASH     {hyp.family}")
                continue

            inc = self.incumbent['valid_primary'] if self.incumbent else self.baseline_valid
            delta = ev['valid_primary'] - inc
            findings += self.auditor.check_gain_plausibility(ev['valid_primary'], inc)
            accept = delta > 0
            out = Outcome(valid_primary=ev['valid_primary'], valid_gauc=ev['valid_gauc'],
                          valid_ndcg=ev['valid_ndcg'], delta_vs_incumbent=delta,
                          diagnostics={'seed_std': ev['valid_std'], 'seeds': ev['seeds'],
                                       'audit': [f.as_dict() for f in findings]},
                          seconds=round(time.time() - t0, 1))
            e = Entry(iteration=n, hypothesis=hyp, action_kind='patch', code_diff=prop.code,
                      outcome=out, decision='accept' if accept else 'reject',
                      reason=(f"valid {ev['valid_primary']:.4f} +-{ev['valid_std']:.4f} "
                              f"vs incumbent {inc:.4f} (delta {delta:+.4f})"))
            self.ledger.add(e)
            if accept:
                self.incumbent = {'X_path': res['X_path'], 'names': res['names'],
                                  'valid_primary': ev['valid_primary'],
                                  'digest': {'primary': ev['valid_primary'],
                                             'GAUC': ev['valid_gauc'],
                                             'nDCG@5': ev['valid_ndcg'],
                                             'seed_std': ev['valid_std']}}
            if verbose:
                print(f"[{n}] {'ACCEPT' if accept else 'reject':9s} {hyp.family:9s} "
                      f"valid {ev['valid_primary']:.4f}+-{ev['valid_std']:.4f} "
                      f"({delta:+.4f})  stall={self.ledger.stall_counter(self.eps)}  "
                      f"{hyp.statement[:48]}")
        return self.ledger.summary()
