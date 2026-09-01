# KAIROS — an autonomous ML research agent that knows when its validation set is lying

**TikTok TechJam 2026 — Track 2: Autonomous ML Research Agent for Recommender Systems**
Benchmark: KuaiRand-Pure, within-user ranking on `long_view`, primary = mean(GAUC, nDCG@5).

| | valid | hidden test | vs baseline | produced by |
|---|---|---|---|---|
| Official FM baseline (reproduced exactly) | 0.6016 | 0.5946 | — | organizers |
| Greedy validation-following agent (control) | 0.7339 | 0.5790 | −0.0156 | control arm |
| Hand-built ensemble (reference ceiling) | 0.6045 | 0.5976 | +0.0030 | human |
| **KAIROS accepted candidate — the submission** | **0.6030** | **0.5983** | **+0.0037** | **agent** |

GAUC 0.6653 (+0.0043) · nDCG@5 0.5313 (+0.0031) · `score_dataset` **+0.0037**

**10 iterations · 2 accepts · 0 manual interventions within the run · 941 s ·
164,789 tokens (≈$1.37) · 0 GPU-hours.** Both accepted candidates were independently
backtest-confirmed. Full trajectory in
[`reports/ITERATION_LOG.md`](reports/ITERATION_LOG.md), numbers in
[`reports/RESULTS.md`](reports/RESULTS.md).

**How the run ended.** A convergence rule was declared before the run and recorded in
[`run_submission.sh`](run_submission.sh), as FAQ 2.9.1 permits: ε = 0.002, **N = 5**,
minimum-iteration floor **10**. Both stopping conditions became true at the same moment:
after iteration 10 the stall counter stood at 10 (>= N=5) with the floor satisfied, **and**
the self-imposed 150k token budget was spent. The loop checks the budget first, so the log
records `STOPPED: token budget exhausted (164,789 >= 150,000)` — verbatim in
[`runs/live_submission.log`](runs/live_submission.log). The run summary reports both
separately (`stop_kind: token_budget`, `converged_predicate: true`) rather than collapsing
them, because "converged" alone would misdescribe how the loop exited. The scored
submission is the validation-best checkpoint at the point the run stopped, which is what
FAQ 2.9.1(c) requires; the hard caps (50 iterations, 6 h) were never approached.

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
It is committed as [`reference_handbuilt_ensemble.csv`](reference_handbuilt_ensemble.csv) —
named so it cannot be mistaken for a submission — and scores 0.5976 under
`experiments/verify_submission.py reference_handbuilt_ensemble.csv`.

**Seed determinism.** The accepted candidate reports `valid_std: 0.0` from a single seed.
That is not a violation of the >=3-seed rule used elsewhere in this repo: under
`mode='scores'` the candidate performs rank fusion of already-computed signals, which
involves no stochastic training, so repeated seeds are bit-identical by construction.
Claims about *trained* models in this repo use >=3 seeds.

Reproduce the submitted run: `export ANTHROPIC_API_KEY=... && ./run_submission.sh` ·
re-score the shipped submission in ~30 s with no API key:
`./.venv/bin/python experiments/verify_submission.py` ·
full writeups in [`reports/`](reports/)

## How this scores against the judging criteria

Every claim below links to the artifact that evidences it. Nothing here is asserted without
something committed to check it against.

### Technical Execution (35%) — primary metric and robustness

**Primary metric.** `score_dataset` = **+0.0037** (GAUC +0.0043, nDCG@5 +0.0031) on the
hidden test set, from the validation-best checkpoint, scored once.
[`reports/RESULTS.md`](reports/RESULTS.md) · re-score it yourself in ~30 s with no API key:
`./.venv/bin/python experiments/verify_submission.py`.

Read against the attainable range rather than against 1.0: the organizers note a perfect
ranking reaches 0.8645 and random sits at 0.4753. Our own decomposition says the *honest*
ceiling is far lower still — context plus item quality alone reaches 0.5955, already above
the official baseline, and every personalisation feature combined adds ~0.006
([`reports/FINDINGS.md`](reports/FINDINGS.md) §6). On this benchmark +0.0037 is a
meaningful share of what is actually available, not a small share of what is theoretically
available.

**Robustness** — judged on how failure is handled, not whether it occurs. Every mechanism
below exists because the corresponding failure actually happened and is recorded:

| failure mode | how the agent handles it | evidence |
|---|---|---|
| candidate raises | traceback fed back to the coder; up to 2 repair attempts, planner not re-invoked | `ledger_errors.jsonl` |
| candidate hangs | per-candidate wall-clock budget, process killed and reported | `verify_mem_guard.py` |
| candidate exhausts memory | parent-side RSS watchdog kills the child, not the parent | `verify_mem_guard.py` |
| candidate leaks a label | structural audit, then independent backtest confirmation on every accept | `verify_agent_mechanisms.py` |
| a check cannot be run | reported as *unverifiable*, never as *disproved* | `verify_agent_mechanisms.py` |
| cached signal is the wrong shape | refused at load with both row counts named | `verify_variant_isolation.py` |

Ten test suites run green; `experiments/verify_*.py` is the whole list.

### Innovation & Problem Insight (20%) — what the agent targeted, and why

The distinctive claim is not the search but the **verification**. A greedy
validation-following agent — our control arm — reached **0.7339 on validation and 0.5790 on
hidden test**, i.e. *below* the official baseline while appearing to have solved the task.
Selection regret measured at **0.0131** ([`reports/FINDINGS.md`](reports/FINDINGS.md) §7).
KAIROS is built around that finding:

1. **A temporal-validity auditor** that vetoes a candidate before its score is believed.
2. **Transfer-based selection** across three temporal backtest folds, not argmax on one
   validation set.
3. **Backtest confirmation of every accepted candidate**, using two independent leak
   signals — a valid−test gap check and an absolute-ceiling check — because each catches a
   leak shape the other misses.
4. **Falsifiable predictions**: each iteration commits to one named diagnostic and a
   direction, checked afterwards. This is what let us *disprove our own claim* that an
   adversarial critic improved reasoning — see Limitations.
5. **A contamination rule on the agent's prior**: nothing fed to the agent may cite or
   derive from the test split ([`kairos/agent/prior.py`](kairos/agent/prior.py)).

Across the stack, not just the model: features, objective, model family, ensembling,
*and the evaluation loop itself* — which is where the actual finding is.

### Impact & Relevance (20%) — autonomy

**0 manual interventions within the run**, plus one human-authored prior, stated in full
[above](#what-0-interventions-does-and-does-not-mean). The agent proposed, coded, ran,
evaluated, and accepted or rejected each candidate on its own evaluation of results. Ten
iterations, two accepts, both independently confirmed.

### Feasibility & Practicality (15%) — resource consumption

| | |
|---|---|
| Tokens (in + out) | **164,789** |
| Agent wall-clock | **941 s** (~16 min) |
| Iterations | **10** of the 50 cap |
| GPU-hours | **0** — CPU only |
| Approximate API cost | **≈$1.37** |

The whole submitted campaign costs about a dollar and a quarter and finishes inside twenty
minutes on a laptop, which is the point: this is a loop a researcher could actually run.

---

## Deliverables

Where each item the track asks for lives in this repo.

| # | Deliverable | Where |
|---|---|---|
| 1 | Written project description (Devpost) | [`reports/DEVPOST.md`](reports/DEVPOST.md) — includes tools, APIs, libraries and datasets used |
| 2 | Public repository with a README | this file; setup, reproduction steps, limitations and contributions below |
| 3 | Per-iteration run log — hypothesis, code diff, metrics, error/recovery events | [`reports/ITERATION_LOG.md`](reports/ITERATION_LOG.md) (Pure) · [`reports/ITERATION_LOG_1K.md`](reports/ITERATION_LOG_1K.md) (1k bonus) · raw ledgers in [`runs/kairos_submission_repro/`](runs/kairos_submission_repro/) |
| 3 | Manual-intervention summary | **0 within the run**, plus one human-authored prior — stated in full [above](#what-0-interventions-does-and-does-not-mean) and in [`reports/RESULTS.md`](reports/RESULTS.md) |
| 4 | Final model output in the starter-kit schema | [`submission.csv`](submission.csv) — 170,588 rows, passes `python3 submit.py --check` |
| 4 | Results table + absolute delta over the baseline | [`reports/RESULTS.md`](reports/RESULTS.md), generated by `experiments/results_table.py` |
| 4 | Resource usage — tokens, wall-clock, iterations, GPU-hours | [`reports/RESULTS.md`](reports/RESULTS.md) — 164,789 tokens, 941 s, 10/50 iterations, 0 GPU-hours |
| — | Bonus benchmark (KuaiRand-1k) | [`reports/RESULTS_1K.md`](reports/RESULTS_1K.md) |

Everything in Deliverables 3 and 4 is **generated from run artifacts**, not typed by hand:
`experiments/export_run_log.py` renders the ledgers, `experiments/results_table.py` renders
the results table, and `experiments/verify_submission.py` re-scores `submission.csv` with the
organizers' own `evaluate.py` and exits non-zero if any reported metric has drifted.

### The rest of `reports/`

| file | what it is |
|---|---|
| [`FINDINGS.md`](reports/FINDINGS.md) | the full technical record — every experiment, including the ~30 that produced nothing |
| [`DATA_DISCIPLINE.md`](reports/DATA_DISCIPLINE.md) | exactly which rows each model is fitted on, and the one ambiguous case |
| [`BUILD_REVIEW.md`](reports/BUILD_REVIEW.md) | a self-review of the build, written mid-project |
| [`UPDATE.md`](reports/UPDATE.md) | what changed after that review |
| [`RESEARCH_PROPOSALS.md`](reports/RESEARCH_PROPOSALS.md) | twelve proposed research directions, scored against measurement — 1 of 12 paid off |

The last three are **historical**: they describe the earlier 3-iteration campaign and carry
a banner saying so. They are kept because the negative results and the retracted claims in
them are part of the evidence, not despite it.

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
separates understanding from luck. On the submitted campaign the agent's predictions scored
**1 of 10**: it beat the baseline while the diagnostics it named largely did not move. We
report that rather than the flattering earlier figure — see [Limitations](#limitations-honestly).

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
# the headline number, straight from the shipped submission file - no API key, ~30 s
./.venv/bin/python experiments/verify_submission.py    # official evaluate.py on submission.csv

# contract tests - these are the guarantees everything else rests on
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

# regenerate deliverables 3 and 4 from the run artifacts
./.venv/bin/python experiments/export_run_log.py       # -> reports/ITERATION_LOG*.md
./.venv/bin/python experiments/results_table.py        # -> reports/RESULTS.md
```

To re-run the submitted campaign end to end (needs an API key, ~16 min, ≈$1.37):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
./run_submission.sh    # writes to runs/kairos_submission_rerun/; refuses to clobber the archive
```

`run_submission.sh` is the reproduction entry point: paths are relative to the repo root and
the key comes from the environment. `run_live.sh` and `run_agent.sh` are development
scripts kept for provenance — `run_live.sh` in particular reflects an earlier configuration
and is not the submitted run.

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
| Convergence: a team may declare its own ε, N and floor if fixed before the run and recorded (FAQ 2.9.1) | declared ε = 0.002, **N = 5**, floor 10, recorded in `run_submission.sh` before the run; implemented in `Ledger.stall_counter`, and crashed iterations are skipped rather than counted as misses | `verify_stall_counter.py` |
| The scored submission is the validation-best checkpoint at the point the run stops (FAQ 2.9.1c) | satisfied — though the run stopped on the self-imposed 150k token budget, not on the ε/N rule; the stop reason is quoted verbatim from `runs/live_submission.log` in `reports/RESULTS.md` | run log |
| Caps: 50 iterations, 6 h wall-clock | `max_iters` and `max_seconds` in `Kairos`; the submitted run used 10 iterations and 941 s, so neither cap was approached | `verify_stall_counter.py` |
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
  0.601, not the 0.8645 oracle, so +0.0037 is a meaningful share of what is actually
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
- **The 1k transfer probe has not finished.** The port is done, the FM baseline is
  reproduced (valid 0.5778 / test 0.5856), and the porting process found five latent defects
  in our own code. A campaign is **in flight at the time of writing**: iteration 2 accepted a
  candidate at validation 0.6522 (+0.0744 over 1k's own baseline, single seed,
  backtest-confirmed). No 1k hidden-test score has been taken and no 1k submission file
  exists. A gain that large is, on this project's own argument, a reason for suspicion until
  confirmed rather than a result to advertise. Current state in
  [`reports/RESULTS_1K.md`](reports/RESULTS_1K.md). An earlier 1k attempt did stall — every
  large-gain candidate was rejected because backtest confirmation blew its timeout — and the
  cause turned out to be narrower than "the verifier is too expensive": prewarm did not cover
  the confirmation fold, so each check rebuilt two windowed FMs over 11.7M rows inside the
  sandbox.
- **The leak detector's absolute-ceiling threshold is calibrated on Pure.**
  `HONEST_CEILING` in `_backtest_confirm` comes from measurements on KuaiRand-Pure's
  backtest folds. On another variant it has no calibration behind it, so the detector's
  gap check transfers but its ceiling check does not.
- **The prediction hit-rate did not hold up.** An earlier 3-iteration run scored 2 of 3
  and we suggested the adversarial critic had lifted it from 0 of 2. The submitted
  10-iteration campaign scored **1 of 10**. Ten is a more honest sample than three, so the
  earlier reading was noise and the claim is withdrawn: the agent beats the baseline while
  the diagnostics it predicts will move largely do not move. Predictions remain worth
  scoring — an agent that commits to a falsifiable claim can be checked, and this is what
  being checked looks like when the answer is unflattering.

## Contributions

Solo entry. All code in `kairos/` and `experiments/` written for this submission; the
files listed as the official starter kit are the organizers' and are unmodified.
