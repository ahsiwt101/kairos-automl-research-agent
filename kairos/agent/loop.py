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
from kairos.agent.sandbox import run_candidate, _run_capped
from kairos.kernel.dataset import Data, FOLDS
from kairos.kernel.causal import window_horizons
from kairos.kernel.frozenfeat import OFFICIAL_WINDOWS, windows_for_fold


class Kairos:
    def __init__(self, proposer, fold_name='official', workdir='runs/kairos',
                 eps=0.002, stall_limit=3, max_iters=50, max_seconds=6 * 3600,
                 seeds=(0, 1, 2), repair_attempts=2, python=None, audit_enabled=True,
                 max_tokens_total=400_000, prior_summary=None, baseline_valid=0.6016,
                 prewarm=('mf', 'cf', 'aux'), confirm_fold='backtest_a',
                 min_iters=0):
        self.proposer = proposer
        self.fold_name = fold_name
        self.workdir = workdir
        self.eps, self.stall_limit = eps, stall_limit
        # FAQ 2.9.1 lets a team declare its own eps, N and a minimum-iteration floor, so
        # long as they are fixed before the run and recorded in the log. A floor exists
        # because N=3 can end a run after three iterations - which satisfies the stopping
        # rule but shows no trajectory, and the trajectory is what a research agent is
        # actually being judged on.
        self.min_iters = int(min_iters)
        self.max_iters, self.max_seconds = max_iters, max_seconds
        self.seeds, self.repair_attempts = seeds, repair_attempts
        # audit_enabled=False reproduces a conventional agent: write code, train, follow
        # the validation number.  It is the control arm for the ablation, not a fallback.
        self.audit_enabled = audit_enabled
        # Hard spend guard.  The run is expected to use ~30k tokens; this ceiling exists so
        # a pathological repair loop cannot quietly run up a bill, and it is reported in the
        # ledger either way because token usage is a scored criterion.
        self.max_tokens_total = max_tokens_total
        # Each Kairos instance starts a blank ledger, so without this the agent has no way
        # to know a strategy already lost in an earlier run - a real research agent should
        # carry that forward the way a lab notebook would.
        self.prior_summary = prior_summary
        self.python = python or sys.executable
        os.makedirs(workdir, exist_ok=True)
        self.data = Data()
        self.fold = self.data.fold(fold_name)
        self.auditor = Auditor(self.data, self.fold)
        self.hz = window_horizons(self.data.date.astype(np.int64), OFFICIAL_WINDOWS)
        np.save(os.path.join(workdir, 'hz.npy'), self.hz)
        # Which torch-backed primitives to compute up front. Prewarming exists to dodge the
        # torch/lightgbm OpenMP abort, so anything NOT prewarmed must be unreachable rather
        # than lazily built inside a candidate subprocess - see Context(blocked=...).
        # On KuaiRand-Pure (1.4M rows) the full set costs ~4 min; on 1k (11.7M) it is ~1h,
        # so a transfer probe trims it deliberately and tells the agent what it has.
        self.prewarm = tuple(prewarm)
        # Both budgets scale with the dataset. Pure's 1.4M rows build well inside 900s;
        # 1k's 11.7M rows do not, and a build that is merely SLOW should not be reported to
        # the agent as a failed hypothesis. The memory cap must also leave the parent room:
        # the parent holds the columnar cache and every prewarmed signal, so a cap set too
        # close to total RAM gets the PARENT killed rather than the child - which is the
        # silent-vanish failure this guard exists to prevent.
        self.cand_mem_gb = float(os.environ.get('KAIROS_CAND_MEM_GB', '6'))
        self.cand_timeout = int(os.environ.get('KAIROS_CAND_TIMEOUT', '900'))
        # the fold _backtest_confirm re-runs candidates against; warmed alongside the
        # working fold so confirmation never pays a cold-cache cost inside the sandbox
        self.confirm_fold = confirm_fold
        self._prewarm_caches()
        # The score a candidate must beat to be accepted. Defaults to the official FM
        # baseline, but a run that resumes work should be given the BEST KNOWN result
        # instead - otherwise each fresh run restarts from the baseline and can "accept" a
        # candidate that is worse than one an earlier run already found. Same class of bug
        # as the stall counter seeding from -inf: comparing against the wrong reference.
        self.baseline_valid = baseline_valid
        # Diagnostics of the starting incumbent (the official FM baseline). Without this
        # the FIRST iteration has nothing to compare against, so its prediction scores as
        # None - and the first iteration is the one most worth scoring.
        self.base_digest = self._baseline_digest()
        self.ledger = Ledger(path=os.path.join(workdir, 'ledger.jsonl'),
                             baseline=self.baseline_valid)
        self.incumbent = None          # dict with X_path, names, valid_primary

    def _prewarm_caches(self):
        """Compute every torch-backed ctx primitive here, in the trusted parent process,
        before any candidate runs.

        torch (baseline_score / auxiliary_signal, on a cold cache) and lightgbm (which
        candidates are free to import for their own models, in 'scores' mode) crash if
        loaded into the SAME process - each bundles its own OpenMP runtime and aborts on
        the second load. Every candidate executes inside its own sandboxed subprocess where
        it could plausibly do both, so the fix is to make sure a candidate's ctx access
        never triggers a cold-cache torch import at all - only ever a cheap np.load.
        """
        from kairos.agent.context import make_context
        from kairos.kernel.baseline_signal import AUX_COLUMNS

        # Warm BOTH the working fold and the confirmation fold. Every ctx signal caches per
        # FOLD (fm_signal_{fold}.npy, refit_cache/{fold}_...), and _backtest_confirm re-runs
        # the candidate's own code against `confirm_fold`. Warming only the working fold
        # therefore left the confirmation run to build those signals from scratch INSIDE the
        # candidate subprocess - which is both the slow path and precisely the torch-beside-
        # lightgbm situation this method exists to prevent.
        #
        # On Pure that cost ~1 min and never showed. On 1k it is two windowed FMs over
        # 11.7M rows, so every large-gain candidate was rejected for exceeding its budget -
        # reported as "the verifier is too expensive" when the real cause was a cold cache.
        for fold_name in dict.fromkeys((self.fold_name, self.confirm_fold)):
            c = make_context(fold_name)
            _ = c.baseline_score
            if 'refit' in self.prewarm:
                _ = c.refit_score()
            if 'din' in self.prewarm:
                _ = c.din_score()
            if 'expert' in self.prewarm:
                for sub in ('context', 'item', 'user'):
                    _ = c.expert_score(sub)
            if 'mf' in self.prewarm:
                _ = c.mf_factors(16)
            if 'cf' in self.prewarm:
                _ = c.cf_score()
        if 'aux' in self.prewarm:
            for name in AUX_COLUMNS:
                _ = c.auxiliary_signal(name)

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
                'mode': ('consolidate' if self.stall_limit - stall <= 1 else 'explore'),
                'note': ('CONSOLIDATE: one miss ends the run. Propose a minimal variation '
                         'on the incumbent, or a family with a non-negative track record. '
                         'An exploratory swing here will be rejected before it is run.'
                         if self.stall_limit - stall <= 1 else
                         'EXPLORE: misses are affordable, a speculative swing is fine.'),
                'family_track_record': self.ledger.family_track_record()}

    # ------------------------------------------------------------------ evaluation
    def _evaluate(self, X_path, train_cfg=None):
        cfg = {'X_path': os.path.abspath(X_path), 'fold': self.fold_name,
               'train_cfg': train_cfg or {},
               'seeds': list(self.seeds), 'add_dev': True,
               'hz_path': os.path.abspath(os.path.join(self.workdir, 'hz.npy')),
               'pred_out': os.path.abspath(os.path.join(self.workdir, 'valid_pred.npy')),
               'out': os.path.abspath(os.path.join(self.workdir, 'eval.json'))}
        p = os.path.join(self.workdir, 'evalcfg.json')
        json.dump(cfg, open(p, 'w'))
        # Same memory bound as the build step. Evaluation trains the downstream model over
        # every seed, so on a large variant it is the LARGER of the two - leaving it
        # unguarded meant the OS killed the parent here instead, and the run vanished with
        # an empty log exactly as it did before the build step was capped.
        r, killed = _run_capped([self.python, 'kairos/agent/evaluate_candidate.py', p],
                                3600, self.cand_mem_gb)
        if killed:
            return None, (f'evaluation exceeded its {killed} budget'
                          + (f' ({self.cand_mem_gb:g} GB)' if killed == 'memory' else ''))
        if r.returncode != 0:
            return None, '\n'.join(r.stderr.strip().splitlines()[-10:])
        return json.load(open(cfg['out'])), None

    # ------------------------------------------------------------------ backtest confirm
    def _backtest_confirm(self, prop, threshold=0.035, backtest_fold=None):
        """Re-run this candidate's OWN code against a backtest fold and measure its OWN
        valid-test gap there (backtest folds have a genuinely unsealed test window).

        This exists because the structural leakage checks (constancy on a named column)
        cannot see every leak shape - a candidate is free to write its own streaming/causal
        aggregate over ANY user x item cross under ANY column name, and such a cross is
        SUPPOSED to vary within a user's list, so no name- or shape-based check catches it.
        Measured on this project: an honest candidate's valid-test gap is ~0.005-0.010; the
        leaky pattern (within-window label feedback) is ~0.12-0.15 - wide separation either
        way, and this check does not need to know anything about what the candidate's code
        does internally to catch it.
        """
        backtest_fold = backtest_fold or self.confirm_fold
        res = run_candidate(prop.code, backtest_fold, timeout=self.cand_timeout,
                            workdir=os.path.join(self.workdir, 'backtest_confirm'))
        if not res['ok']:
            return None, f"could not run on {backtest_fold}: {res.get('error','')[:200]}"
        hz_path = os.path.join(self.workdir, f'hz_{backtest_fold}.npy')
        if not os.path.exists(hz_path):
            hz_bt = window_horizons(self.data.date.astype(np.int64),
                                    windows_for_fold(FOLDS[backtest_fold]))
            np.save(hz_path, hz_bt)
        ecfg = {'X_path': os.path.abspath(res['X_path']), 'fold': backtest_fold,
               'train_cfg': res.get('train_cfg') or {}, 'seeds': list(self.seeds),
               'add_dev': True, 'hz_path': os.path.abspath(hz_path), 'also_test': True,
               'out': os.path.abspath(os.path.join(self.workdir,
                                                    'backtest_confirm_eval.json'))}
        cfg_path = os.path.join(self.workdir, 'backtest_confirm_cfg.json')
        json.dump(ecfg, open(cfg_path, 'w'))
        r, killed = _run_capped([self.python, 'kairos/agent/evaluate_candidate.py', cfg_path],
                                3600, self.cand_mem_gb)
        if killed:
            return None, f'backtest scoring exceeded its {killed} budget'
        if r.returncode != 0:
            tail = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else 'unknown error'
            return None, f"could not score on {backtest_fold}: {tail[:150]}"
        ev = json.load(open(ecfg['out']))
        gap = ev['valid_primary'] - ev['test_primary']
        # Two INDEPENDENT signals, because one leak shape evades the other.
        #   gap-widening leak (e.g. causal_prefix keyed on a single id, horizoned per
        #     fold): inflates valid only -> caught by the gap check. Measured: causal_all
        #     on backtest_a is valid 0.7156 / test 0.5915, gap +0.1241.
        #   globally-inflating leak (e.g. a hand-rolled streaming aggregate with no
        #     per-fold horizon at all): inflates valid AND test roughly equally, since it
        #     is blind to fold boundaries entirely - the gap stays small or even negative.
        #     Measured directly on the candidate that motivated this check: backtest_a
        #     valid 0.6855 / test 0.6871, gap -0.0016 - passes the gap check outright.
        # The ceiling check closes that hole: HONEST_CEILING is the best backtest_a score
        # among every frozen-construction pipeline measured this project (frozen_all:
        # valid 0.5928 / test 0.5934); a wide margin above it on EITHER split, honest gap
        # or not, is not something any known-clean pipeline has ever produced here.
        HONEST_CEILING = {'backtest_a': 0.60, 'backtest_b': 0.60, 'backtest_c': 0.58}
        ceiling = HONEST_CEILING.get(backtest_fold, 0.62) + 0.05
        over_ceiling = max(ev['valid_primary'], ev['test_primary']) - ceiling
        # bool() is load-bearing: these comparisons yield numpy bools, and the caller
        # tests `ok is not True`. np.bool_(True) is not the True singleton, so without the
        # coercion every CONFIRMED candidate would be misread as unverifiable and rejected.
        ok = bool((gap < threshold) and (over_ceiling < 0))
        detail = (f"{backtest_fold}: valid {ev['valid_primary']:.4f} test "
                 f"{ev['test_primary']:.4f} gap {gap:+.4f} (threshold {threshold}) | "
                 f"ceiling {ceiling:.3f}, over by {over_ceiling:+.4f}")
        return ok, detail

    def _baseline_digest(self):
        """Diagnostics of the official baseline, so iteration 1's prediction is checkable."""
        from kairos.kernel.diagnostics import Diagnostics
        from kairos.agent.context import make_context
        try:
            base = make_context(self.fold_name).baseline_score
            va = self.fold.idx['valid']
            return Diagnostics(self.fold, 'valid',
                               base[va].astype(np.float64)).digest()
        except Exception:
            return None

    def _digest_for(self, pred_path):
        """Full diagnostics digest for a candidate, from its validation predictions.

        The loop previously stored only a four-key summary as the "digest", which is why
        the agent's slice-level predictions could never be checked - the quantities it was
        predicting about were not in the dict. This computes the real thing.
        """
        from kairos.kernel.diagnostics import Diagnostics
        try:
            pred = np.load(pred_path)
            return Diagnostics(self.fold, 'valid', pred.astype(np.float64)).digest()
        except Exception:
            return None

    # ------------------------------------------------------------------ one iteration
    def step(self, n):
        from kairos.kernel.diagnostics import Diagnostics
        digest = self.incumbent['digest'] if self.incumbent else {
            'primary': self.baseline_valid, 'note': 'FM baseline, no diagnostics yet',
            **({'prior_run_summary': self.prior_summary} if self.prior_summary else {})}
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
            # Enforce the scheduler's mode. With one miss left the run ends on a failure,
            # so a family that has already lost repeatedly is not worth a full training
            # run - re-ask the planner once (planner-only, ~2k tokens) instead of spending
            # ~90s of compute and the last of the stall budget on it.
            bud = self.budget()
            if bud['mode'] == 'consolidate' and attempt == 0:
                rec = bud['family_track_record'].get(hyp.family)
                if rec and rec['n'] >= 2 and rec['mean_gain'] < 0:
                    self.ledger.log_error(
                        n, 'scheduler',
                        f"family '{hyp.family}' has mean gain {rec['mean_gain']:+.4f} over "
                        f"{rec['n']} attempts and the run ends on the next miss",
                        're-asked the planner for a consolidating move before spending a '
                        'training run')
                    last_failure = {'stage': 'scheduler',
                                    'error': f"'{hyp.family}' has lost {rec['n']} times "
                                             f"(mean {rec['mean_gain']:+.4f}). One miss "
                                             f"ends the run.",
                                    'hint': 'propose a minimal variation on the incumbent, '
                                            'or a family with a non-negative track record'}
                    continue
            res = run_candidate(prop.code, self.fold_name, timeout=self.cand_timeout,
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
            # the floor suppresses only the CONVERGENCE rule, never the hard caps
            scored = sum(1 for e in self.ledger.entries
                         if (e.outcome.valid_primary == e.outcome.valid_primary))
            if done and scored < self.min_iters and 'stalled' in (why or ''):
                done, why = False, None
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

            new_digest = self._digest_for(os.path.join(self.workdir, 'valid_pred.npy'))
            prev_digest = self.incumbent.get('digest') if self.incumbent else self.base_digest
            from kairos.kernel.diagnostics import check_prediction
            hit = check_prediction(getattr(hyp, 'prediction', None) or {},
                                   prev_digest, new_digest)
            inc = self.incumbent['valid_primary'] if self.incumbent else self.baseline_valid
            delta = ev['valid_primary'] - inc
            gain_findings = self.auditor.check_gain_plausibility(ev['valid_primary'], inc)
            findings += gain_findings
            accept = delta > 0
            reason = (f"valid {ev['valid_primary']:.4f} +-{ev['valid_std']:.4f} "
                     f"vs incumbent {inc:.4f} (delta {delta:+.4f})")
            # An implausibly large validation jump gets checked against a backtest fold's
            # OWN unsealed test score before it is trusted - regardless of what the
            # candidate's code does or what it names its columns. This is what the leaked
            # 'history' family candidate needed and did not have: it scored +0.09 on
            # validation from a hand-rolled streaming aggregate that no structural check
            # was watching, and shipped.
            # Confirm EVERY accepted candidate, not only implausible-looking ones.
            # The previous condition (`accept and gain_findings`) meant a candidate whose
            # gain looked ordinary was never independently checked - and the artifact this
            # project submitted was exactly that case: +0.0018, below the implausibility
            # threshold, so the flagship verifier never ran on the thing that shipped.
            #
            # The RESPONSE stays proportionate to the evidence:
            #   implausible gain -> confirmation is REQUIRED; anything but a pass blocks.
            #   ordinary gain    -> confirmation is run for the record; a DISCONFIRMATION
            #                       blocks, but a merely unverifiable one (infrastructure
            #                       failure) is logged without vetoing an honest accept.
            if accept and self.audit_enabled:
                ok, detail = self._backtest_confirm(prop)
                reason += f" | backtest confirm: {detail}"
                # ok is True (confirmed), False (DISCONFIRMED - the backtest ran and the
                # gain did not survive) or None (UNVERIFIABLE - the backtest could not run
                # at all). Both non-True cases block acceptance, because an unverified
                # +0.07 is exactly what we must never ship. But they are NOT the same
                # evidence and must not carry the same message: telling the agent its
                # hypothesis "likely leaked" when the verifier merely timed out is a false
                # accusation that steers it off a possibly-sound idea. Same principle as
                # check_prediction() returning None rather than False for what it cannot
                # verify.
                required = bool(gain_findings)
                if ok is False or (ok is None and required):
                    accept = False
                elif ok is None:
                    self.ledger.log_error(
                        n, 'unconfirmed_accept',
                        f'accepted on validation (+{delta:.4f}); backtest confirmation '
                        f'could not run ({detail})',
                        'accepted anyway - the gain is ordinary, so an infrastructure '
                        'failure in the checker does not veto it; recorded as unconfirmed')
                if ok is not True and not accept:
                    if ok is False:
                        kind = 'implausible_gain'
                        msg = (f"validation jumped +{delta:.4f} and the backtest "
                               f"DISCONFIRMED it ({detail}) - likely within-window label "
                               f"feedback from a leak shape the structural checks do not "
                               f"cover")
                        rec = ('rejected without spending a repair attempt; counts as a '
                               'normal miss against the stall budget, not a crash')
                    else:
                        kind = 'unverified_gain'
                        msg = (f"validation jumped +{delta:.4f} but the backtest could "
                               f"not be run to check it ({detail}). This is an "
                               f"INFRASTRUCTURE failure, not evidence of a leak - the "
                               f"hypothesis is untested, not refuted")
                        rec = ('rejected because a large gain must never be accepted '
                               'unverified; re-propose it in a cheaper form the backtest '
                               'can afford to confirm')
                    self.ledger.log_error(n, kind, msg, rec)
            out = Outcome(valid_primary=ev['valid_primary'], valid_gauc=ev['valid_gauc'],
                          valid_ndcg=ev['valid_ndcg'], delta_vs_incumbent=delta,
                          prediction_hit=hit,
                          diagnostics={'seed_std': ev['valid_std'], 'seeds': ev['seeds'],
                                       'audit': [f.as_dict() for f in findings]},
                          seconds=round(time.time() - t0, 1))
            e = Entry(iteration=n, hypothesis=hyp, action_kind='patch', code_diff=prop.code,
                      outcome=out, decision='accept' if accept else 'reject', reason=reason)
            self.ledger.add(e)
            if accept:
                self.incumbent = {'X_path': res['X_path'], 'names': res['names'],
                                  'valid_primary': ev['valid_primary'],
                                  'digest': new_digest or {
                                      'primary': ev['valid_primary'],
                                      'GAUC': ev['valid_gauc'],
                                      'nDCG@5': ev['valid_ndcg'],
                                      'seed_std': ev['valid_std']}}
            if verbose:
                mark = {True: 'pred:HIT ', False: 'pred:miss', None: '         '}[hit]
                print(f"[{n}] {'ACCEPT' if accept else 'reject':9s} {hyp.family:9s} {mark} "
                      f"valid {ev['valid_primary']:.4f}+-{ev['valid_std']:.4f} "
                      f"({delta:+.4f})  stall={self.ledger.stall_counter(self.eps)}  "
                      f"{hyp.statement[:48]}")
        return self.ledger.summary()
