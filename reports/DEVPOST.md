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
   read. Across a candidate pool, official validation correlates **+0.297** with hidden-test
   rank; backtest transfer correlates **+0.685**.
4. **A stall-aware scheduler.** The competition ends a run after 3 consecutive iterations
   without a +0.002 validation gain, which makes consecutive misses a *spendable resource*
   rather than a diagnostic. The agent budgets them explicitly.
5. **Multi-seed by construction.** Per-seed std is 0.0008 while the convergence rule needs
   +0.002, so a single-seed measurement cannot resolve the improvement the rules require.
   Every accept/reject uses ≥3 seeds. (We retracted two of our own early conclusions when
   we re-ran them with seeds.)
6. **Autonomous recovery.** LLM-written code runs behind an AST allowlist and in a
   subprocess with a timeout. When the auditor rejects a candidate, the specific violation
   is handed back and the agent rewrites — no human in the loop. Every rejection and repair
   is a logged error/recovery event.

## Results

Two different things, kept scrupulously separate, since this track is judged on autonomy:

**The submission** (`submission.csv`) is a research pipeline we built by hand, using the
same primitives the agent has — an ensemble of the FM baseline, a frozen-window GBDT, and
a behavioural-feedback signal, rank-fused. It beats the baseline and defines the ceiling
we know is reachable with this feature set.

| | valid | test | vs baseline |
|---|---|---|---|
| Official FM baseline (reproduced exactly, 5 seeds) | 0.6016 | 0.5946 | — |
| Greedy validation-following agent (control arm) | 0.7330 | 0.5790 | **−0.0156** |
| **Hand-built ensemble (submitted)** | 0.6045 | **0.5976** | **+0.0030** |

**The live agent** — Claude Opus 5 planning, Claude Sonnet 5 coding, zero manual
intervention in any accept/reject decision — got close but did not cross the baseline
within its allotted budget, and correctly retained the baseline rather than ship something
short of it. Across a sequence of live runs (each independently governed by the
competition's own N=3-without-+0.002 stopping rule), it: caught and self-corrected out of
two structural API mistakes without help; got its own accepted candidate blocked by our
temporal-validity auditor after that candidate slipped a leak neither of us had
anticipated (a hand-rolled aggregate over a user×item cross, closed via a backtest
confirmation gate added specifically because of this); and, once its action space was
extended to let it blend separately-trained models (rather than concatenate everything
into one feature matrix for a single tree — the architecture every earlier attempt lost
with), independently rediscovered the FM + frozen-history rank-fusion recipe and closed
the gap to baseline monotonically across three consecutive iterations (delta vs.
incumbent: −0.0024 → −0.0017 → −0.0011) before stalling out under budget.

We think that trajectory — reasoning correctly about *why* an architecture fails, adapting
across constrained budgets, catching its own mistakes, and refusing to ship an unproven
gain — is a more informative signal about the agent than the final number, on a benchmark
where the honest headroom above the baseline is on the order of a few thousandths of
primary in the first place.

The absolute gain is modest, and we are explicit about that: several well-motivated
directions produced nothing measurable, and we report them all — objective alignment
across six losses, recency weighting, watch-time regression, and the D2Q
duration-deconfounded target. On a benchmark where the noise floor (0.0008) sits under the
decision threshold (0.002), knowing which of your gains are real *is* the hard part.

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
