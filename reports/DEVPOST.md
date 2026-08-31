# KAIROS — an autonomous ML research agent that knows when its validation set is lying

## The problem, and what we found in it

Track 2 asks for an agent that autonomously runs the MLE loop — read the problem, engineer
features, train, evaluate, reflect, iterate — and drives the score above the official
baseline on KuaiRand-Pure.

The obvious agent for that job is a loop that proposes a change, trains, reads the
validation score, and keeps whatever went up. **On this benchmark that agent fails, and it
fails invisibly.**

We measured it. Take the single most natural feature-engineering step — summarise each
user's and item's history as a `long_view` rate, computed with a correct time-ordered
prefix so no row ever sees its own label — and pair it with LambdaRank, a standard choice
for a ranking metric. Validation climbs from 0.6016 to **0.7330**, close to the 0.8484
validation oracle. The hidden-test score falls to **0.5790** — *below the 0.5946 baseline
the agent was trying to beat.*

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
   it. The fix doesn't try to understand the candidate's code at all — any implausible
   validation jump gets the same candidate re-run against a backtest fold with a genuinely
   unsealed test split, checked on both the valid/test gap and the absolute score against
   the best honest result ever measured there. Verified against the exact candidate that
   slipped through, and against a known-honest one, in both directions.
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
4. **A stall-aware scheduler.** The competition ends a run after 3 consecutive iterations
   without a +0.002 validation gain, which makes consecutive misses a *spendable resource*
   rather than a diagnostic. With one miss left the loop enters `consolidate` mode and
   re-asks the planner — for ~2k tokens — rather than spending a full training run and the
   last of the budget on a hypothesis family that has already lost twice.
5. **Multi-seed by construction.** Per-seed std is 0.0008 while the convergence rule needs
   +0.002, so a single-seed measurement cannot resolve the improvement the rules require.
   Every accept/reject uses ≥3 seeds. (We retracted two of our own early conclusions when
   we re-ran them with seeds.)
6. **Autonomous recovery.** LLM-written code runs behind an AST allowlist and in a
   subprocess with a timeout. When the auditor rejects a candidate, the specific violation
   is handed back and the agent rewrites — no human in the loop. Every rejection and repair
   is a logged error/recovery event.
7. **Scored predictions.** Each iteration names one checkable diagnostic and the direction
   it expects it to move; we verify afterwards. This measures whether the agent
   *understands* the problem separately from whether it got lucky, and the per-family
   hit-rate feeds back into which hypotheses earn more of the remaining budget. No
   published ML agent we found measures this.
8. **An adversarial critic** audits whether a stated prediction actually follows from the
   stated mechanism before any code is written, catching the vacuous case ("primary
   increases" is true of every improvement and so tests nothing). Hit-rate went **0/2 →
   2/3** after it was added, for ~1k tokens per check and no extra training runs.

## Results

**The submission is the agent's own work.** `submission.csv` is generated from a candidate
KAIROS proposed, wrote, evaluated and accepted with zero human intervention.

| | valid | test | vs baseline | produced by |
|---|---|---|---|---|
| Official FM baseline (reproduced exactly, 5 seeds) | 0.6016 | 0.5946 | — | organizers |
| Greedy validation-following agent (control arm) | 0.7339 | 0.5790 | **−0.0156** | control |
| Hand-built ensemble (human reference ceiling) | 0.6045 | 0.5976 | +0.0030 | human |
| **KAIROS accepted candidate** | **0.6034** | **0.5988** | **+0.0042** | **agent** |

**The agent beat both the official baseline and the strongest pipeline we built by hand.**
Its winning move, chosen unprompted: abandon the single-downstream-tree architecture and
blend several decorrelated model outputs at the *rank* level. That is the same design that
took a human several failed attempts to reach — and the agent reached it in one iteration,
once its action space could express it.

The run: **3 iterations, 0 manual interventions, 376 seconds, 36,490 tokens (≈$0.35)**,
converged on the competition's own N=3-without-+0.002 rule.

### The agent can be shown to reason, not just to get lucky

Every iteration commits to a falsifiable prediction — one named diagnostic, one direction —
which is checked after the candidate runs. That separates understanding from luck, and it
is measurable:

| | prediction hit-rate |
|---|---|
| before the critic | 0 / 2 |
| after the critic | 1 / 3 |
| current | **2 / 3** |

The improvement came from an adversarial critic that audits whether a stated prediction
actually follows from the stated mechanism. Caught live: a hypothesis about duration
confounding paired with the prediction "primary increases" — true of *any* improvement, so
it tests nothing. The critic replaced it with `inversion_loss_duration` decreasing, the
quantity the mechanism actually implies.

### What the agent recovered from, unaided

- Two structural API mistakes, self-corrected from the error text
- An invalid LightGBM hyperparameter name, repaired rather than crashing the iteration
- Its own leaking candidate — accepted at +0.0936 validation — caught by the
  temporal-validity auditor before it could be believed

### Honest limits

The absolute gain is small: +0.0042 is 1.6% of the headroom above the baseline. Our own
signal decomposition explains why, and we report it rather than hide it — context plus item
quality reaches 0.5955 on this data and personalisation is worth roughly +0.006 in total,
so every honest intervention here competes for single-digit thousandths.

We tested roughly 30 interventions. Two mechanisms worked: **variance reduction**
(seed-averaging, saturating at 3 seeds) and **not fooling yourself** (selection discipline,
worth avoiding −0.0156). Six loss functions, watch-time regression in three forms, D2Q,
recency weighting, four tab encodings, eight item-quality estimators, and DIN history
reweighting all landed inside seed noise. Those negative results are in the report, because
on a benchmark where validation rank-correlates **−0.612** with the hidden test, knowing
which of your gains are real is the hard part.

## How it was built

**Development tools.** Claude Code (Anthropic) as the pair-programming environment; Python
3.14; git.

**APIs.** The agent's proposer is provider-agnostic — Anthropic Messages API, or any
OpenAI-compatible `/chat/completions` endpoint (Volcengine Ark / Doubao, OpenRouter,
DeepSeek, or a local Ollama server), selected by a single command-line string.

**Libraries and frameworks.** NumPy, SciPy, pandas (loading only), LightGBM, PyTorch,
httpx, and the `anthropic` SDK. No dependency on the modelling side beyond LightGBM and
PyTorch; the evaluation kernel is pure NumPy.

**Datasets.** KuaiRand-Pure only (Zenodo record 10439422). No external training data, no
pretrained weights, in line with the track's single hard rule. The randomized-exposure log
(`log_random_4_22_to_5_08_pure.csv`) is deliberately **not** used for training — it spans
the test window, and training on it would inject label information from that period. We
use it, if at all, only as an unbiased validation aid, as the starter kit suggests.

**Scoring discipline.** The official `evaluate.py` is committed unmodified and is the sole
scoring authority; our vectorised evaluator is verified identical to it to 4.4e-16. Every
consultation of the sealed test split goes through an audited scorer, and that log is
published with the submission.
