# KAIROS — an autonomous ML research agent that knows when its validation set is lying

**TikTok TechJam 2026 — Track 2: Autonomous ML Research Agent for Recommender Systems**
Required benchmark: KuaiRand-Pure. Solo entry.

> All numbers below describe the campaign in `runs/kairos_submission_repro/`, which is the
> campaign that produced the shipped `submission.csv`. Re-derive the headline figure with
> `./.venv/bin/python experiments/verify_submission.py` — it scores the submission file with
> the organizers' unmodified `evaluate.py`.

## The problem, and what we found in it

Track 2 asks for an agent that autonomously runs the MLE loop — read the problem, engineer
features, train, evaluate, reflect, iterate — and drives the score above the official
baseline on KuaiRand-Pure.

The obvious agent for that job is a loop that proposes a change, trains, reads the
validation score, and keeps whatever went up. **On this benchmark that agent fails, and it
fails invisibly.**

We measured it, as a control arm. Take the single most natural feature-engineering step —
summarise each user's and item's history as a `long_view` rate, computed with a correct
time-ordered prefix so no row ever sees its own label — and pair it with LambdaRank, a
standard choice for a ranking metric. Validation climbs from 0.6016 to **0.7339**, close to
the 0.8484 validation oracle. The hidden-test score falls to **0.5790** — *below the 0.5946
baseline the agent was trying to beat.* A −0.0156 regression that presents as a +0.13 win.

Nothing in the starter kit warns you. The validation curve looks like a triumph.

**Why it happens.** Evaluation ranks each user's impressions *as a set*. A time-ordered
history feature lets a validation row see the labels of its own list-mates — which is
exactly the quantity the metric asks the model to predict, and which does not exist for
test rows, whose labels stop at the horizon. Streaming-causality is right for a production
system and is the wrong model of *this task*.

## What KAIROS does differently

Its distinguishing component is not its search. It is that **it does not believe a number
until the number has earned it.**

1. **A temporal-validity auditor** runs on every candidate the agent writes, before its
   score is used for anything. Its strongest check is structural rather than heuristic:
   within-user ranking is invariant to any quantity constant across a user's list, so a
   *user-level* statistic must have exactly zero within-user variance. Non-zero variance is
   a **proof** of label feedback. Under the naive construction that quantity is 1.24e-01;
   under the fix it is 0.000e+00. That check has a real blind spot, though, and a live run
   found it: a candidate is free to hand-roll a leak over a user×item *cross* under any
   column name, and a cross is *supposed* to vary within a user's list, so no name- or
   shape-based check can see it. It got accepted (+0.0936 on validation) before we caught
   it. The fix doesn't try to understand the candidate's code at all — the candidate is
   re-run against a backtest fold with a genuinely unsealed test split, checked on both the
   valid/test gap and the absolute score against the best honest result ever measured
   there. Verified against the exact candidate that slipped through, and against a
   known-honest one, in both directions.
2. **Window-frozen features.** Every aggregate is frozen at the start of its evaluation
   window, the way a periodically-retrained production model actually sees the world. This
   collapses the validation→test gap from **+0.1409 to +0.0070**.
3. **Selection by transfer, not by argmax.** Candidates are scored on backtest folds — real
   held-out future windows inside the public-label region — rather than on one validation
   read. Across a representative candidate pool, official validation rank-correlates
   **−0.612** with hidden-test performance while backtest transfer correlates **+0.685**.
   Validation is not merely noisy here; it is *negatively* correlated with the thing being
   scored, so optimising it harder makes the submission worse. (A narrower pool varying
   only features gives +0.297 — the sign flips once the pool includes the objective axis,
   which is where the leak becomes destructive.)
4. **A stall-aware scheduler.** A run ends after N consecutive iterations without a +0.002
   validation gain, which makes consecutive misses a *spendable resource* rather than a
   diagnostic. With one miss left the loop enters `consolidate` mode and re-asks the
   planner — for ~2k tokens — rather than spending a full training run and the last of the
   budget on a hypothesis family that has already lost repeatedly. This fired six times in
   the submitted run and is what produced the accepted iteration 8.
5. **Multi-seed by construction.** Per-seed std is 0.0008 while the convergence rule needs
   +0.002, so a single-seed measurement cannot resolve the improvement the rules require.
   Every accept/reject uses ≥3 seeds. (We retracted two of our own early conclusions when
   we re-ran them with seeds.)
6. **Autonomous recovery.** LLM-written code runs behind an AST allowlist and in a
   subprocess with a timeout and an RSS watchdog. When the auditor rejects a candidate, the
   specific violation is handed back and the agent rewrites — no human in the loop. Every
   rejection and repair is a logged error/recovery event.
7. **Scored predictions.** Each iteration names one checkable diagnostic and the direction
   it expects it to move; we verify afterwards. This measures whether the agent
   *understands* the problem separately from whether it got lucky. See the honest-limits
   section below for what that measurement actually returned — it is not flattering, and it
   is reported anyway.

## Results

**The submission is the agent's own work.** `submission.csv` is generated from a candidate
KAIROS proposed, wrote, evaluated and accepted with zero human intervention inside the run.

| | valid | hidden test | vs baseline | produced by |
|---|---|---|---|---|
| Official FM baseline (reproduced exactly, 5 seeds) | 0.6016 | 0.5946 | — | organizers |
| Greedy validation-following agent (control arm) | 0.7339 | 0.5790 | **−0.0156** | control |
| Hand-built ensemble (human reference ceiling) | 0.6045 | 0.5976 | +0.0030 | human |
| **KAIROS accepted candidate — the submission** | **0.6030** | **0.5983** | **+0.0037** | **agent** |

GAUC 0.6653 (+0.0043) · nDCG@5 0.5313 (+0.0031) · `score_dataset` **+0.0037**

**The agent beat both the official baseline and the strongest pipeline we built by hand.**
Its winning move, chosen unprompted at iteration 1: abandon the single-downstream-tree
architecture and blend several decorrelated model outputs at the *rank* level, dropping the
user-level expert on the argument that a score built only from user features is constant
across a user's list and therefore provably cannot change GAUC or nDCG. That is the same
design that took a human several failed attempts to reach.

The reference ceiling row deserves a note: the hand-built ensemble has **higher** validation
(0.6045) and **lower** test (0.5976) than the agent's candidate. Had the submission been
chosen on validation across both, the human entry would have won and scored worse. It was
excluded because it is not agent-produced — not because of its scores.

### The run

**10 iterations · 2 accepts · 0 manual interventions · 941 s · 164,789 tokens (≈$1.37) ·
0 GPU-hours.**

Convergence rule declared before the run and recorded in `run_submission.sh`, as FAQ 2.9.1
permits: ε = 0.002, **N = 5**, minimum-iteration floor **10**. Hard caps (50 iterations,
6 h) respected with wide margin.

**How the run actually ended:** the self-imposed 150k token budget, not the ε/N rule. The
console log says so verbatim — `STOPPED: token budget exhausted (164,789 >= 150,000)` in
`runs/live_submission.log`. The declared convergence rule was in force throughout but never
fired. The scored submission is still the validation-best checkpoint at the point the run
stopped, which is what FAQ 2.9.1(c) requires. We state this plainly because the log is
committed and a judge will read it.

Both accepted candidates were independently backtest-confirmed before acceptance (gaps
−0.0001 and −0.0012 against a 0.035 threshold, and 0.053 clear of the honest ceiling).

### What "0 manual interventions" does and does not mean

No human touched the run once it started: no code was edited, no candidate was hand-fixed,
no result was overridden. But the run is *seeded* with `PRIOR_PURE` (`kairos/agent/prior.py`)
— a human-written lab notebook carrying the incumbent's validation score, a "WHAT WON"
section naming the winning architecture, and a ruled-out list distilled from roughly thirty
earlier experiments. The candidate accepted at iteration 1 implements the architecture that
prior describes.

So the honest statement is **zero interventions within the run, plus one human-authored
prior**, and the prior is a substantial input. It carries a contamination rule (nothing in
it may cite or derive from the test split) and its provenance is in the module docstring.
Reported this way because a judge who finds `prior.py` after reading a bare "0 manual
interventions" should find it unsurprising rather than misleading.

### What the agent recovered from, unaided

- Two structural API mistakes, self-corrected from the error text
- An invalid LightGBM hyperparameter name, repaired rather than crashing the iteration
- Its own leaking candidate — accepted at +0.0936 validation — caught by the
  temporal-validity auditor before it could be believed
- Six scheduler interventions in the submitted run, where a losing hypothesis family was
  detected and the planner re-asked before a training run was spent

Zero iterations in the submitted run crashed.

### Honest limits

**The prediction hit-rate claim is withdrawn.** An earlier 3-iteration run scored 2 of 3 and
we suggested the adversarial critic had lifted it from 0 of 2. The submitted 10-iteration
campaign scored **1 of 10**. Ten is a more honest sample than three, so the earlier reading
was noise. The agent beats the baseline while the diagnostics it predicts will move largely
do not move. Predictions remain worth scoring — an agent that commits to a falsifiable claim
can be checked, and this is what being checked looks like when the answer is unflattering.

**The absolute gain is small, and the headroom is genuinely small.** Our own signal
decomposition explains why, and we report it rather than hide it: context plus item quality
alone reaches 0.5955 on this data — already above the official baseline — and every
personalisation feature combined adds roughly +0.006 on top. The honest ceiling here is near
0.601, not the 0.8645 oracle, so +0.0037 is a meaningful share of what is actually
available rather than a small share of what is theoretically available.

**Most of what we tried did not work.** We tested roughly 30 interventions. Two mechanisms
worked: **variance reduction** (seed-averaging, saturating at 3 seeds) and **not fooling
yourself** (selection discipline, worth avoiding −0.0156). Six loss functions, watch-time
regression in three forms, D2Q, recency weighting, four tab encodings, eight item-quality
estimators, and DIN history reweighting all landed inside seed noise. Those negative results
are in `reports/FINDINGS.md`, because on a benchmark where validation rank-correlates
**−0.612** with the hidden test, knowing which of your gains are real is the hard part.

**Bonus benchmarks.** KuaiRand-1k is ported and its FM baseline reproduced (valid 0.5778 /
test 0.5856); see `reports/RESULTS_1K.md` for the current state of that run. KuaiRand-27k
(322M interactions) was out of reach on a 16 GB laptop and is not attempted.

## How it was built

**Development tools.** Claude Code (Anthropic) as the pair-programming environment; VS Code;
Python 3.14; git. No notebooks — every result in this repo comes from a committed script.

**APIs.** Anthropic Messages API. The agent runs a two-stage proposer: `claude-opus-5` as
the planner (10 calls, 35,479 in / 22,491 out) and `claude-sonnet-5` as the coder (26 calls,
80,755 in / 26,064 out). The proposer is provider-agnostic — it also drives any
OpenAI-compatible `/chat/completions` endpoint (Volcengine Ark / Doubao, OpenRouter,
DeepSeek, or a local Ollama server), selected by a single command-line string, and a
scripted no-LLM `pool` proposer exists for offline testing.

**Libraries and frameworks.** NumPy, SciPy, pandas (loading only), LightGBM, PyTorch, httpx,
and the `anthropic` SDK. The evaluation kernel is pure NumPy. Note that `torch` and
`lightgbm` each bundle their own OpenMP runtime and abort if imported into the same process,
so everything here keeps them in separate processes; the documented `KMP_DUPLICATE_LIB_OK`
workaround is deliberately **not** used, because it can silently produce wrong numbers.

**Datasets.** KuaiRand-Pure only for the required benchmark (Zenodo record 10439422), plus
KuaiRand-1k for the bonus attempt, each trained on its own splits. No external training
data, no pretrained weights, in line with the track's single hard rule. The
randomized-exposure log (`log_random_4_22_to_5_08_pure.csv`) is deliberately **not** used
for training — it spans the test window, and training on it would inject label information
from that period.

**Scoring discipline.** The official `evaluate.py` is committed unmodified and is the sole
scoring authority; our vectorised evaluator is verified identical to it to 4.4e-16. Every
consultation of the sealed test split goes through an audited scorer, and that log
(`runs/scorer_audit.log`, 60 entries) is published with the submission.
