# KAIROS — an autonomous ML research agent that knows when its validation set is lying

**TikTok TechJam 2026 — Track 2: Autonomous ML Research Agent for Recommender Systems**
Benchmark: KuaiRand-Pure, within-user ranking on `long_view`, primary = mean(GAUC, nDCG@5).

| | valid | hidden test | vs baseline | produced by |
|---|---|---|---|---|
| Official FM baseline (reproduced exactly) | 0.6016 | 0.5946 | — | organizers |
| Greedy validation-following agent (control) | 0.7339 | 0.5790 | −0.0156 | control arm |
| Hand-built ensemble (reference ceiling) | 0.6045 | 0.5976 | +0.0030 | human |
| **KAIROS accepted candidate — the submission** | **0.6034** | **0.5988** | **+0.0042** | **agent** |

**3 iterations · 0 manual interventions within the run · 376 s · 36,490 tokens (≈$0.35)**
— converged on the competition's own N=3-without-+0.002 rule.

**What "0 interventions" does and does not mean.** No human touched the run once it
started: no code was edited, no candidate was hand-fixed, no result was overridden. But the
run is *seeded* with [`PRIOR_PURE`](kairos/agent/prior.py) — a human-written lab notebook
carrying the incumbent's validation score, a "WHAT WON" section naming the winning
architecture, and a ruled-out list distilled from roughly thirty earlier experiments. The
candidate accepted at iteration 1 implements the architecture that prior describes. So the
honest statement is **zero interventions within the run, plus one human-authored prior**,
and the prior is a substantial input. It carries a contamination rule (nothing in it may
cite or derive from the test split) and its provenance is in the module docstring. Reported
this way because a judge who finds `prior.py` after reading a bare "0 manual interventions"
should find it unsurprising rather than misleading.

**The submission is the agent's output by construction.** The hand-built ensemble in the
table above is a *reference ceiling* we measured to know what the agent was competing
against — it was never a candidate for the submission slot. That matters because it has
HIGHER validation (0.6045) and LOWER test (0.5976) than the agent's candidate: if the
submission had been chosen on validation across both, the hand-built one would have won and
scored worse. It was excluded because it is not agent-produced, not because of its scores.

**Seed determinism.** The accepted candidate reports `valid_std: 0.0` from a single seed.
That is not a violation of the >=3-seed rule used elsewhere in this repo: under
`mode='scores'` the candidate performs rank fusion of already-computed signals, which
involves no stochastic training, so repeated seeds are bit-identical by construction.
Claims about *trained* models in this repo use >=3 seeds.

Reproduce: `./run_live.sh` · full writeups in [`reports/`](reports/)

---

## Overview

Most agents for this task are a loop: propose a change, train, read the validation score,
keep it if it went up. On this benchmark that loop is actively harmful, and we can measure
how harmful.

A pipeline built the natural way — summarise each user's and item's history as a
long_view rate, computed with a correct time-ordered prefix so no row sees its own label —
reaches **validation 0.7158**, close to the 0.8484 validation oracle, while its **hidden-test
score falls to 0.5749**, below the 0.5946 baseline it was trying to beat. Validation gain
and test gain are *anti-correlated* in this regime.

The cause is structural. Evaluation ranks each user's impression list **as a set**. A
time-ordered history feature lets a validation row see the labels of its own list-mates —
which is exactly the quantity the metric asks the model to predict, and which does not
exist for test rows, whose labels stop at the horizon. The fix is to freeze every
aggregate at the **start of its evaluation window**, which collapses the validation→test
gap from +0.1409 to +0.0070.

KAIROS is built around that finding. Its distinguishing component is not its search but
its **temporal-validity auditor**, which vetoes a candidate before its score is believed,
and its **selection rule**, which chooses on transfer across backtest folds rather than
argmax over a single validation set.

It is also built to be *checkable*. Each iteration commits to a falsifiable prediction —
one named diagnostic, one direction — which we verify after the candidate runs. That
separates understanding from luck, and the number moves: the prediction hit-rate has gone
**0/2 → 2/3** since adding an adversarial critic that audits whether a stated prediction
actually follows from the stated mechanism.

## What is in here

| path | what it is |
|---|---|
| `kairos/kernel/fastmetrics.py` | vectorised GAUC / nDCG@5, exact to 4.4e-16 vs the official `evaluate.py`, 12x faster |
| `kairos/kernel/dataset.py` | cached columnar loader; test labels sealed behind an audited `Scorer` |
| `kairos/kernel/causal.py` | streaming prefix aggregates **and** window-frozen aggregates |
| `kairos/kernel/frozenfeat.py` | deployment-faithful feature matrix + per-fold window schedules |
| `kairos/kernel/diagnostics.py` | per-slice headroom attribution and pairwise inversion attribution |
| `kairos/kernel/candidates.py` | pipeline pool spanning the leaky↔honest axis |
| `kairos/agent/auditor.py` | temporal-validity checks; BLOCK vetoes a candidate |
| `kairos/agent/sandbox.py` | AST allowlist + subprocess isolation for LLM-authored code |
| `kairos/agent/selection.py` | transfer / stability / shrinkage corrections to argmax |
| `kairos/agent/ledger.py` | per-iteration run log; also the stall-budget accounting |
| `kairos/agent/loop.py` | the controller |
| `experiments/` | every experiment and every contract test |

The official starter kit (`evaluate.py`, `data.py`, `baseline.py`, `submit.py`,
`ablation_features.py`, `baseline_scores.json`) is committed **byte-for-byte unmodified**;
`evaluate.py` is the sole authority on scoring. The organizers' original README is
preserved verbatim as `STARTER_KIT_README.md` (renamed only so GitHub displays this file).

## Setup

```bash
# dataset (194 MB, no registration)
curl -LO https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar xzf KuaiRand-Pure.tar.gz

python3 -m venv .venv
./.venv/bin/pip install numpy scipy pandas pyarrow lightgbm torch anthropic
# macOS only: lightgbm needs libomp
brew install libomp
```

Note: `torch` and `lightgbm` each bundle their own OpenMP runtime and abort if imported
into the same process. Everything here keeps them in separate processes. The documented
`KMP_DUPLICATE_LIB_OK` workaround is deliberately **not** used — it can silently produce
wrong numbers, which is not acceptable on a benchmark.

## Reproducing

```bash
# contract tests first - these are the guarantees everything else rests on
./.venv/bin/python experiments/verify_metrics.py       # fast metrics == official evaluate.py
./.venv/bin/python experiments/verify_causal.py        # prefix aggregates vs brute force
./.venv/bin/python experiments/verify_frozen.py        # frozen aggregates vs brute force
./.venv/bin/python experiments/audit_assumptions.py    # silent-failure assumptions
./.venv/bin/python experiments/verify_baseline.py      # 5-seed reproduction of the baseline

# findings
./.venv/bin/python experiments/exp02_loss_ablation.py  # objective alignment: no effect
./.venv/bin/python experiments/exp03_diagnose.py       # where the metric is actually lost
./.venv/bin/python experiments/exp06_frozen.py         # the leak, and the fix
./.venv/bin/python experiments/exp11_selection.py      # selection-rule comparison

# the agent
./.venv/bin/python experiments/exp08_agent_smoke.py    # end-to-end, incl. catch-and-recover
```

## Verified reproduction of the official baseline

Seeds 0–4, matching the published protocol:

| | ours | published |
|---|---|---|
| valid primary | 0.6016 | 0.6016 |
| test primary | 0.5946 | 0.5946 |
| test std (5 seeds) | 0.0008 | 0.0008 |
| test GAUC / nDCG@5 | 0.6610 / 0.5282 | 0.6610 / 0.5282 |

Row order is also byte-identical to `data.load()` on all three splits — checked because
submission alignment is positional and `(user_id, video_id)` is not unique (3.06% of test
rows are repeated pairs).

## Rules compliance

Judging is by code review of the pipeline and run logs, so each rule is listed with the
code that enforces it and the test that pins it.

| rule | how this repo satisfies it | pinned by |
|---|---|---|
| Never train on the test split or use its labels — including for model selection, early stopping, threshold tuning, or feature statistics | `also_test` is **hard-refused** on the official fold (`evaluate_candidate.py` raises); training, early stopping and candidate acceptance all read validation only; test-row features use horizon 20220428, so no test label enters any feature | `verify_train_split_only.py` |
| Training data is the train split only (FAQ 2.9.2) | every trainer is clamped to `date <= 20220421`; the one ambiguous case is an explicit flag defaulting to strict — see [`reports/DATA_DISCIPLINE.md`](reports/DATA_DISCIPLINE.md) | `verify_train_split_only.py` |
| `log_random` may not be training data (FAQ 2.9.2.1) | never loaded by any training path | — |
| KuaiRand-1k/27k may not be auxiliary training data for Pure (FAQ 2.9.2.2) | enforced structurally: every signal cache is keyed by variant and a cached array of the wrong length is **refused**, so a 1k signal cannot enter a Pure run even by mistake | `verify_variant_isolation.py` |
| No external training data or pretrained weights | nothing outside KuaiRand is read; no pretrained weights | — |
| Convergence: ε = 0.002, N = 3; crashed iterations neither advance nor reset the window (FAQ 2.9.1) | implemented in `Ledger.stall_counter`; crashes are skipped, not counted as misses | `verify_stall_counter.py` |
| Caps: 50 iterations, 6 h wall-clock | `max_iters` and `max_seconds` in `Kairos`; runs converge on the ε/N rule long before either | `verify_stall_counter.py` |
| Submission is the validation-best checkpoint, scored once | selection reads `valid_primary` only; every consultation of the test split is appended to `runs/scorer_audit.log` with its reason | audit log |

**On the §2.3 metric line.** §2.3 lists "NDCG@10 / Recall@50, click = positive". This
contradicts §2.4, §2.6 and Appendix A.4, which all specify **GAUC / nDCG@5 on `long_view`**,
and A.4 notes Recall@50 is 0.999+ for every model including random scoring. §2.4 states the
task is "pinned in the Starter Kit", and the shipped `evaluate.py` computes GAUC / nDCG@5.
We follow the Starter Kit.

## Limitations, honestly

- **Absolute gains are modest, and the headroom is genuinely small.** The scored
  improvement comes mostly from ensembling and disciplined selection, not from a stronger
  model. Our own decomposition is the reason to expect that: context plus item quality
  alone reaches 0.5955 — already above the official baseline — and every personalisation
  feature combined adds roughly 0.006 on top. The honest ceiling on this benchmark is near
  0.601, not the 0.8645 oracle, so +0.0042 is a meaningful share of what is actually
  available rather than a small share of what is theoretically available.
- **Several well-motivated directions produced nothing measurable:** objective alignment
  (all six losses within seed noise), recency weighting, and watch-time regression as a
  ranking target.
- **The noise floor is close to the decision threshold.** Per-seed std is 0.0008 while the
  competition's convergence rule needs +0.002, so single-seed conclusions are unsafe here.
  Every claim above uses ≥3 seeds; two of our own early conclusions were retracted when
  re-run with seeds.
- **Backtest folds are imperfect analogues.** They live inside the public-label region, so
  their test windows are 6–7 days against the official 10, and their training windows are
  shorter.
- **The 27k bonus benchmark is not attempted.** KuaiRand-1k is ported and measured (see
  below); 27k is 322M interactions and was out of reach on a 16GB laptop.
- **The 1k transfer probe is incomplete.** The port is done, the FM baseline is reproduced
  (valid 0.5778 / test 0.5856) and the porting process found five latent defects in our own
  code, but the agent run there has not converged — every large-gain candidate was rejected
  because backtest confirmation could not finish inside its budget, which is a limitation of
  our verifier's cost on 11.7M rows, not a result about the agent.
- **The leak detector's absolute-ceiling threshold is calibrated on Pure.**
  `HONEST_CEILING` in `_backtest_confirm` comes from measurements on KuaiRand-Pure's
  backtest folds. On another variant it has no calibration behind it, so the detector's
  gap check transfers but its ceiling check does not.
- **Prediction hit-rate is measured on a small sample.** 2 of 3 in the latest run, up from
  0 of 2, and the adversarial critic is the plausible cause — but three scored predictions
  is not enough to call that an effect rather than a coincidence.

## Contributions

Solo entry. All code in `kairos/` and `experiments/` written for this submission; the
files listed as the official starter kit are the organizers' and are unmodified.
