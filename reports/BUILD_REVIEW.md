# KAIROS Build Review

**TikTok TechJam 2026 · Track 2 — Autonomous ML Research Agent for Recommender Systems**

An autonomous ML research agent for KuaiRand-Pure — what was built, what it measured,
and what is still in the way.

| | test primary | vs baseline | produced by |
|---|---|---|---|
| Official FM baseline | 0.5946 | — | organizers |
| Hand-built ensemble | 0.5976 | +0.0030 | human (reference ceiling) |
| **KAIROS accepted candidate** | **0.5988** | **+0.0042** | **agent — this is the submission** |
| Manual interventions | **0** | | across every agent run |

---

## Contents

1. [Where it stands](#1-where-it-stands)
2. [What was built](#2-what-was-built)
3. [How the agent works](#3-how-the-agent-works)
4. [Research findings](#4-research-findings)
5. [What worked, what didn't](#5-what-worked-what-didnt)
6. [Roadblocks cleared](#6-roadblocks-cleared)
7. [Roadblocks still open](#7-roadblocks-still-open)
8. [What's left](#8-whats-left)

---

## 1. Where it stands

The submission is produced by the agent, not by hand. That distinction is the point of the track.

| Pipeline | Valid | Hidden test | vs baseline | Produced by |
|---|---|---|---|---|
| Official FM baseline | 0.6016 | 0.5946 | — | organizers |
| Greedy validation-following agent | 0.7339 | 0.5790 | −0.0156 | control arm |
| Multi-signal ensemble | 0.6045 | 0.5976 | +0.0030 | human |
| **KAIROS accepted candidate** | **0.6034** | **0.5988** | **+0.0042** | **agent** |

The agent beat both the baseline and the hand-built ensemble. It reached the winning
architecture — blend several decorrelated model outputs at the *rank* level rather than
concatenating signals into one tree — in a single iteration, once it had the capability
to express it.

### The submitted run

| Iteration | Decision | Valid | Δ incumbent | Prediction |
|---|---|---|---|---|
| 1 · ensemble | **accept** | 0.6034 | +0.0018 | unscored |
| 2 · ensemble | reject | 0.6000 | −0.0034 | miss |
| 3 · ensemble | reject | 0.6007 | −0.0027 | miss |

Converged on the competition's own rule — three consecutive iterations without a +0.002
validation gain. **376 seconds, 36,490 tokens (≈$0.35), zero manual interventions.**

---

## 2. What was built

≈6,700 lines across the agent, a measurement kernel, and the tests that keep both honest.
43 commits.

| Component | Lines | Purpose |
|---|---|---|
| `kairos/agent/` | 1,900 | the loop, LLM proposer, sandbox, auditor, ledger, selection |
| `kairos/kernel/` | 1,811 | data, exact metrics, leakage-safe feature primitives, diagnostics |
| `kairos/models/` | 374 | FM with pluggable losses, DIN sequence model |
| `experiments/` | 2,629 | 11 contract-test files, 28 controlled experiments |

### Guarantees the rest of the work rests on

- **Metric exactness.** A vectorised GAUC/nDCG@5 identical to the official `evaluate.py`
  to 4.4e-16 across seven stress cases including heavy ties — and 13× faster, which is
  what makes thousands of evaluations affordable.
- **Baseline reproduction.** Every published figure matched on the published protocol:
  valid 0.6016, test 0.5946, seed std 0.0008, GAUC 0.6610, nDCG 0.5282.
- **Row-order equivalence.** Byte-identical to `data.load()` on all three splits — checked
  because submission alignment is positional and `(user_id, video_id)` is not unique
  (3.06% of test rows repeat).
- **Sealed test labels.** Reachable only through an audited scorer; every consultation is
  logged. Current count: **58**, all research probes. The agent's own selection never
  reads it.

---

## 3. How the agent works

One iteration, start to finish. Nothing here is scripted — the content of every step comes
from the model.

| Step | What happens | Who does it |
|---|---|---|
| Diagnose | slice metrics, headroom attribution, pairwise inversion analysis | our kernel |
| Plan | decide what to try and why; commit to a falsifiable prediction | Claude Opus 5 |
| Code | write the feature/model construction as real Python | Claude Sonnet 5 |
| Sandbox | AST allowlist, subprocess isolation, timeout | our kernel |
| Audit | temporal-validity checks; backtest confirmation on implausible gains | our kernel |
| Evaluate | train at 3 seeds, score on validation | our kernel |
| Judge | accept/reject; score the prediction; update the stall budget | our kernel |

Splitting plan from code is a cost decision — Feasibility is a scored criterion. It also
makes repair 4× faster (12s vs 50s): when code crashes the *idea* is still sound, so only
the coder re-runs.

### Three mechanisms that make it more than a loop

**Prediction scoring.** The agent must name one of ten checkable diagnostics and a
direction. After the run we check whether it moved that way. This measures understanding
separately from luck — a family that keeps predicting correctly earns more budget.

**Stall-aware scheduling.** The competition ends a run after three consecutive misses,
which makes misses a spendable resource. With one left, the loop enters `consolidate` and
re-asks the planner rather than spending a training run on a family that has already lost
twice.

**Temporal-validity auditing.** A candidate's score is not believed until it survives
structural checks and, for implausible jumps, confirmation on a backtest fold with a
genuinely unsealed test split.

---

## 4. Research findings

### Validation actively lies, and the lie scales with effort

The most natural feature-engineering step on this benchmark — summarise each user's history
as a `long_view` rate with a correct time-ordered prefix — produces this:

| Pipeline | Valid | Hidden test | Gap |
|---|---|---|---|
| FM baseline | 0.6016 | 0.5946 | +0.0070 |
| naive causal history, binary | 0.6158 | 0.5904 | +0.0254 |
| naive causal history, LambdaRank | 0.7158 | **0.5749** | **+0.1409** |

**Cause:** evaluation ranks a user's impressions *as a set*, so a time-ordered history
feature lets a validation row see the labels of its own list-mates — precisely the quantity
the metric asks it to predict, and unavailable for test rows whose labels stop at the
horizon. Freezing every aggregate at its evaluation window's start collapses the gap from
+0.1409 to +0.0070.

> **The sharpest number in the project.** Across a ten-pipeline pool, official validation
> rank-correlates **−0.612** with hidden-test performance. Backtest transfer correlates
> **+0.685**. Validation is not merely noisy here — it is *negatively* correlated with the
> thing being scored. Optimising it harder makes the submission worse.

### What that costs, measured

| Selection rule | Picks | Hidden test | Regret |
|---|---|---|---|
| greedy — argmax validation | causal_all_lmr | 0.5790 | 0.0131 |
| transfer — backtest mean | frozen_all_bin | 0.5910 | 0.0011 |
| robust — transfer-corrected | frozen_all_bin | 0.5910 | 0.0011 |
| oracle (unattainable) | frozen_all_lmr | 0.5921 | 0.0000 |

The leak also *inverts design conclusions*: LambdaRank looks like the best idea in the pool
on validation and is the worst on test with leaky features, while genuinely being the best
with honest ones. A greedy agent doesn't merely score lower — it learns the opposite lesson
about its own objective. Our control-arm ablation puts the auditor's worth at **+0.0156**.

### Why ~0.60 is the practical ceiling

Nested model fits, each rung adding one information source:

| Signal | Standalone | Cumulative | Marginal |
|---|---|---|---|
| context (tab) | 0.5399 | 0.5405 | — |
| **+ item quality** | 0.5807 | **0.5955** | **+0.0550** |
| + item × context | 0.5877 | 0.5966 | +0.0011 |
| + duration fit | 0.4914 | 0.5935 | −0.0030 |
| + affinity (user × author) | 0.4825 | 0.5938 | +0.0002 |
| + affinity (user × item) | 0.4818 | 0.5940 | +0.0002 |

**Context plus item quality reaches 0.5955; every personalisation feature after that is
worth roughly nothing.** The FM's 0.6016 means its ID embeddings buy about +0.006 of
personalisation. On this benchmark the recommender part of the recommender system is worth
~0.006 of primary — which explains the organizers' null ablations, our six-loss null, and
why every honest gain here lands in the thousandths.

Oracle bounds, for reference: a perfect watch-ratio predictor scores 0.8023, a perfect
play-time predictor 0.8418, and the true label 0.8484.

### A data audit we acted on

`video_features_statistic_pure.csv` holds 51 per-video aggregates including `play_progress`
— direct measures of exactly the item quality that carries all the signal. Its counters are
**51–66× larger than this sample's own impression counts**, so they are platform-wide
production numbers, and their temporal coverage *cannot be established from the data at
all*. Using a temporally-unverifiable feature to raise our score would refute the project's
own thesis, so we exclude it and document the decision rather than benefit silently.

---

## 5. What worked, what didn't

Roughly 30 controlled interventions. Two mechanisms worked. Reporting the other 28 is the
honest part.

### Worked

| Intervention | Effect | Evidence |
|---|---|---|
| Rank-fusion ensembling of decorrelated models | +0.0030 → +0.0042 | the only mechanism that repeatedly pays |
| Refit on train+validation (FM only) | +0.0021 | confirmed on two backtest folds |
| Seed averaging | +0.0017 | saturates at 3 seeds |
| Selection discipline | avoids −0.0156 | control-arm ablation |

### Didn't

| Direction | Result | Verdict |
|---|---|---|
| Six loss functions, incl. a metric-exact GAUC surrogate | 0.5874–0.5954 | within noise |
| Watch-time regression (L2 / Huber) | 0.5605 / 0.5754 | worse |
| D2Q duration-deconfounded quantile | 0.5846 | worse |
| Recency weighting (3 seeds) | 0.6017 → 0.6020 | noise |
| Refit on train+validation (GBDT) | +0.0000 | declined |
| Four `tab` encodings | 0.5982–0.5988 | within noise |
| Eight item-quality estimators (decay × hierarchical) | best +0.0002 | within noise |
| Coarse affinity (user × tag, 111 values) | +0.0006 | marginal |
| Static features, model capacity | — | organizer-tested |

**On the 2025–26 literature.** The current published work on short-video ranking — D2Q,
CREAD, TPM, EGMN, TranSUN, DADF, Relative Advantage Debiasing — is almost entirely
*watch-time regression*, optimised for MAE/XAUC. Our task is binary `long_view` ranking
scored by GAUC/nDCG@5, and we measured that target family losing by 0.02–0.03 on identical
features. At a 33% positive rate, direct binary classification already estimates the
threshold-crossing probability efficiently, so the distributional machinery those papers
add buys nothing here.

---

## 6. Roadblocks cleared

Each of these blocked real progress. Several were bugs in our own instrumentation that
would have silently corrupted results.

### The agent could not express the one intervention that wins — FIXED

Refitting on train+validation is worth +0.0021, confirmed on two backtest folds — and
`train_parts` appeared *zero times* in the agent's context or prompt. A human found it; the
agent had no path to it. This is why every earlier run landed just under the baseline.

**Fix:** exposed `ctx.refit_score()` with the split asymmetry handled *inside* the
primitive — train/valid rows get train-only predictions, test rows get the refit — so
weights fitted on validation stay honest by construction. Measured +0.0024 standalone. Also
added `ctx.din_score()`.

### Three signature mechanisms were designed but dead — FIXED

`prediction_hit` was declared in the ledger and never written by anything.
`family_track_record()` was computed and never read. `misses_before_run_ends` was passed to
the LLM as advice with no enforcement. All three looked implemented.

**Fix:** wired all three, with a dedicated test file — because "it is wired" was exactly
the claim that had been silently false.

### A leak shape the auditor structurally could not see — FIXED

A live candidate hand-rolled its own streaming aggregate over user×author / user×tag
crosses, never touching the flagged-dangerous primitive. It scored +0.0936 on validation
and was **accepted**. The structural check only inspects user-level constants, and a cross
feature is *supposed* to vary within a user's list — so no name- or shape-based check could
ever catch it.

**Fix:** a backtest confirmation gate that ignores the candidate's code entirely: re-run it
on a fold with an unsealed test split and check two independent signals — the valid/test
gap, and the absolute score against the best honest result ever measured there. Verified
against the actual leaked candidate and against a known-honest one.

### The stall counter let runs continue past the stated rule — FIXED

It seeded its best-score tracker at −∞ instead of the real incumbent, so the agent earned
false progress credit for beating its own early bad guesses — spending API budget on
iterations the competition's own rule would have ended.

**Fix:** seed from the baseline; pinned with a regression test; re-ran the
already-published control-arm ablation under the correction (conclusion unchanged,
iteration count corrected).

### An `int8` overflow in the metric cross-check — FIXED

Python's `sum()` over `numpy.int8` silently wraps past 127, so a 200-positive user summed
to −56. It corrupted the reference comparison in one debug script.

**Fix:** cast before the reference call. Audited every call site — the real scoring path was
never affected, because `fast_evaluate` casts to float64 immediately.

### A silent composite-key collision waiting to happen — FIXED

Composite keys were packed arithmetically as `uid*1e7 + author`. Max `author_id` on Pure is
8,733,983 — 13% of headroom. The larger KuaiRand variants would have blown through it with
no error, just wrong features.

**Fix:** factorise the pair instead of packing. Cannot collide.

### An unforgiving API cost the agent whole iterations — FIXED

The agent burned ~7 iterations on two API defects: a parameter named `keys` that rejected a
list of key arrays (the natural reading), and per-video arrays (7,538) versus per-row
arrays (1,436,609). numpy reported the first as `object too deep for desired array` —
useless for repair.

**Fix:** accept what a reasonable caller passes (auto-factorise composite keys), add
`ctx.video_attr()` / `ctx.user_attr()` broadcasting, and `ctx.check()` so the agent
validates in its own code. **Lesson: with an LLM repair loop, API ergonomics is a
performance characteristic, not a style preference.**

### A validation error that discarded whole working iterations — FIXED

An invalid hyperparameter name raised inside the *scoring* subprocess, which has no repair
loop — silently throwing away an iteration including a `build()` that had run correctly,
over a one-word typo (`n_estimators`, an XGBoost name).

**Fix:** moved key validation into the same subprocess and retry loop as the candidate's
own code.

### Environment and credential friction — FIXED

Missing `libomp` and `scikit-learn`; torch and LightGBM aborting when loaded into one
process (each bundles its own OpenMP runtime); an identity-linked Anthropic key that needed
a workspace-scoped one; exhausted credits, twice; ModelArk models requiring per-account
activation.

**Fix:** installed dependencies; isolated torch and LightGBM into separate processes and
pre-warm every torch-backed cache before any candidate runs; switched to a workspace key.
The documented `KMP_DUPLICATE_LIB_OK` workaround was deliberately *not* used — it can
silently produce wrong numbers.

### Two analysis errors caught before they reached the report — FIXED

A double-ranking bug silently degenerated an ensemble to "FM alone" (caught only because it
contradicted an earlier result). And a decomposition table labelled standalone scores as
"marginal", which read as features *destroying* signal.

**Fix:** corrected both; the decomposition now reports standalone and cumulative as
explicitly separate columns.

---

## 7. Roadblocks still open

Honest accounting of what is unresolved, and what each one would cost to close.

### The agent's predictions do not verify — OPEN

Track record on the submitted run: **0 hits from 2 scored predictions.** It beats the
baseline, but the diagnostics it claims will move do not move. So we can demonstrate the
mechanism works, and what it currently says is that the agent is succeeding for reasons it
cannot articulate — a genuinely interesting result, but not a flattering one.

*Needs more runs to know whether this is a real pattern or three data points. The first
iteration was also unscored (no baseline digest existed at the time — since fixed, but
after this run).*

### The absolute gain is small — OPEN

+0.0042 is 1.6% of the headroom above the baseline. Our own decomposition says this is
structural — personalisation is worth ~0.006 total on this data — but that reasoning is
only convincing if a reviewer accepts the decomposition.

*If a competitor reports a much larger gain, the most likely explanations are the
temporally-unverifiable statistic file or within-window label feedback. We can detect both;
we cannot prevent others from using them.*

### Two rules questions never answered — EXTERNAL

**§2.3 contradiction:** the Limits table says "NDCG@10 / Recall@50, click = positive",
contradicting the starter kit, §2.4, §2.6 and Appendix A.4, which all say `long_view` /
GAUC / nDCG@5. We follow `evaluate.py`, which the kit declares authoritative.

**`log_random`:** 1.19M rows of randomised exposure spanning the validation *and test*
windows. We use it for nothing, since training on it would inject label information from
the test period.

*Both should be raised with the organizers. Neither blocks work; the second could
invalidate a competitor's approach.*

### DIN trains on shorter histories than it serves — OPEN

Training rows average 7.9 history items (31% empty); validation and test average ~17 (4%
empty). Training rows sit early in the timeline with less accumulated history — inherent to
the frozen-window construction, and a genuine train/serve mismatch.

*DIN still scores 0.6023, above baseline, so it is not fatal. Fixing it
(history-length-stratified sampling) is untested.*

### Bonus datasets not attempted — OPEN

KuaiRand-1k (11.7M interactions) and 27k (322M) are extra credit. The 1k transfer run is
the strongest remaining Feasibility story — the agent re-running from a distilled prior at
near-zero token cost.

*1k is tractable. 27k is a serious engineering job on 8 cores / 16GB and is out of scope.*

### Deliverables not finished — OPEN

The repo is **private** (deliberately — it holds the leakage findings mid-competition) and
must be flipped public at submission. Writeups exist but still foreground the hand-built
work rather than the agent. No video.

*One command to publish; the writeup reframe is Phase 5 of the current plan.*

---

## 8. What's left

In the order I would do it.

1. **Reframe the writeups around the agent.** Its hypotheses, its prediction hit-rate, its
   recovery events, its token and intervention counts — with the hand-built modelling
   demoted to "the ceiling it is measured against."
2. **More agent runs.** Three iterations is a thin basis for the prediction-scoring result,
   and more runs may also improve the score.
3. **KuaiRand-1k transfer.** Extra credit plus the Feasibility narrative.
4. **Ask the organizers** about §2.3 and `log_random`.
5. **Flip the repo public** and produce the optional ~3 minute video.

---

> **The claim worth defending.** The differentiator is not the delta. It is that this
> project can *tell the difference* between a real gain and a validation illusion —
> demonstrated by catching its own agent committing one at +0.0936 validation and −0.0156
> test, and by an auditor whose worth is measured at +0.0156.
>
> On a benchmark where validation rank-correlates **−0.612** with the hidden test, that
> capability is worth more than the score it produces.

---

*KAIROS · TikTok TechJam 2026, Track 2 · 43 commits on `score-push` · 58 audited
hidden-test consultations · submission produced autonomously*
