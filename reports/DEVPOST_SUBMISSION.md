# Devpost submission — copy-paste content

## Project name  (50 char limit)

```
KAIROS — Autonomous ML Research Agent
```
*(37 chars)*

Alternative if you want the hook in the name:
```
KAIROS: the agent that audits its own results
```
*(45 chars)*

---

## Elevator pitch  (200 char limit)

```
An autonomous ML research agent that knows when its validation set is lying. It audits its own results before believing them — and beats the baseline where a greedy agent lands 0.0156 below it.
```
*(193 chars)*

Shorter alternative:
```
An autonomous ML research agent that knows when its validation set is lying. It verifies its own results instead of trusting them, and beats the baseline where a greedy agent scores below it.
```
*(190 chars)*

---

## About the project  (Markdown)

## Inspiration

The standard autonomous ML agent runs a loop: propose a change, train, read the validation
score, keep it if the number went up. We built that agent first, as a control, and it did
something worth stopping for.

It reached **0.7330 on validation** — a number that looks like the benchmark had been
solved — and scored **0.5790 on the hidden test set**, which is **0.0156 *below* the
official baseline it set out to beat**.

It had discovered a leak. On this benchmark it is remarkably easy to build a feature that
lets a validation row see labels from inside its own evaluation window. The agent could not
tell the difference between a discovery and a leak, because the only instrument it had was
the number that the leak inflates.

Worse, the damage was not just lost points — it **inverted the ranking of design choices**:

| pipeline | valid | hidden test |
|---|---|---|
| causal features + LambdaRank | **0.7330** ← best on validation | **0.5790** ← worst on test |
| frozen features + LambdaRank | 0.5973 | **0.5921** ← best on test |

With leaky features, LambdaRank looks like the best training objective in the pool and is
actually the worst. With honest features it genuinely is the best. So a greedy agent does
not merely pick a bad candidate — it learns the **opposite lesson about its own objective**
and carries that into every later iteration.

That reframed the problem. The hard part of an autonomous ML researcher is not generating
ideas. It is **knowing which of its own results to believe**.

## What it does

KAIROS runs the full MLE loop autonomously — reads the problem, engineers features, writes
and trains models, evaluates, reflects, iterates — with one structural difference: **no
result is believed until it survives an independent check.**

On KuaiRand-Pure it produced the submitted result in **10 iterations with 0 manual
interventions**, in **941 seconds** and **164,789 tokens (~$1.37)** on a laptop CPU:

| metric | official baseline | KAIROS | delta |
|---|---|---|---|
| GAUC | 0.6610 | 0.6653 | **+0.0043** |
| nDCG@5 | 0.5282 | 0.5313 | **+0.0031** |
| primary | 0.5946 | 0.5983 | **+0.0037** |

## How we built it

Seven stages per iteration, ~300 lines of controller:

1. **Propose** — Claude Opus 5 plans a hypothesis, its *mechanism*, and one **falsifiable
   prediction** (a named diagnostic and a direction). Claude Sonnet 5 writes the code. A
   critic pass audits whether the prediction actually follows from the stated mechanism.
2. **Audit statically** — allowlisted sandbox; outcome columns blocked at the data API; a
   leakage probe requires any user-level statistic from a new primitive to have exactly zero
   within-user variance.
3. **Run isolated** — subprocess bounded in time *and* memory, with a parent-side RSS
   watchdog so an over-allocating candidate cannot kill the run.
4. **Evaluate** — vectorised GAUC/nDCG@5, verified identical to the organizers' scorer to
   4.4e-16 across seven stress cases including heavy ties, and 13× faster.
5. **Confirm** — every accepted candidate is re-run on a temporal backtest fold and checked
   with two independent leak signals, because each catches a shape the other misses.
6. **Score the prediction** — did the diagnostic move as promised? HIT, miss, or
   *unverifiable* — never silently counted as a miss.
7. **Decide** — accept only on validation improvement **plus** confirmation.

The selection rule is the core of it. We measured how well each candidate signal predicts
hidden-test performance:

| selection signal | rank correlation with hidden test |
|---|---|
| official validation | +0.297 |
| **backtest-fold transfer** | **+0.685** |

So KAIROS selects on transfer across three temporal backtest folds, never argmax on one
validation set.

## What we learned

**The honest headroom on this benchmark is tiny, and we can show why.** A nested
decomposition on validation:

| signal | standalone | cumulative | marginal |
|---|---|---|---|
| context (tab) | 0.5399 | 0.5405 | — |
| + item quality | 0.5807 | **0.5955** | **+0.0550** |
| + item × context | 0.5877 | 0.5966 | +0.0011 |
| + affinity (user × item) | 0.4818 | 0.5940 | +0.0002 |

Context plus item quality alone reaches **0.5955 — already above the official baseline**.
Every personalisation feature after that is worth approximately nothing: the FM's ID
embeddings buy about **+0.006** over a model with no personalisation at all. On this
benchmark the "recommender" part of the recommender system is worth ~0.006 of primary.
That is the context for reading +0.0037.

**Roughly 30 things did not work**, and the negative results are the substance: all six
training objectives within seed noise; watch-time regression far below the binary target;
eight item-quality estimators worth at most +0.0002; seed averaging saturating by three
seeds; recency weighting a single-seed phantom we retracted on re-run.

**We reproduced our own central finding on our own work.** Adding power exponents to the
rank fusion raised validation 0.6031 → 0.6035 and *lowered* test 0.5985 → 0.5982. Extra
parameters fitted against an unreliable signal buy validation and cost reality.

**And we had to withdraw a claim.** An early 3-iteration run scored 2 of 3 on its
predictions and we suggested our adversarial critic had improved the agent's reasoning. The
10-iteration run scored **1 of 10**. Three iterations was too small a sample; the claim is
withdrawn in the README. Scoring predictions is only worth doing if the unflattering answer
gets reported too.

## Challenges we ran into

**The bugs that produce wrong answers instead of errors.** Porting to KuaiRand-1k found
**five latent defects in our own code**, every one invisible on the primary benchmark:
side tables indexed by position rather than id (fine on Pure, silently wrong on 1k, where
32 ids are missing); a memory guard using `RLIMIT_AS`, which *raises* on macOS and whose
error we swallowed — a guard that looked present and capped nothing; caches shared across
datasets, loading a 1.4M-row signal into an 11.7M-row problem.

The first test we wrote for the memory guard passed against the broken version, because the
allocation bomb tripped a row-count assertion before it ever touched memory. **A test that
passes for the wrong reason certifies a guard that does nothing.**

**Making "we never selected on test" checkable rather than promised.** We had 60 audited
consultations of the test split, all post-hoc measurement. That position is unfalsifiable
from outside, so we added a sealed mode that makes the test split *raise* on access, and we
pre-committed to which of two campaigns to submit **before** scoring either. The one we
chose scored 0.0005 *lower*. We shipped it anyway, because a commitment you abandon when
the number disagrees was never a commitment.

## What's next

KuaiRand-1k, the bonus benchmark, is where the finding gets a real test: 117× more history
per user than Pure. If personalisation is worth ~0.006 on Pure because 44 rows per user is
too few, then 1k should behave differently — and so far it does, with a backtest-confirmed
**+0.0744** on validation against its own baseline. Same agent, same code, one environment
variable.

---

## Built with  (tags)

```
python, numpy, pytorch, lightgbm, pandas, anthropic-claude, claude-opus-5,
claude-sonnet-5, anthropic-api, recommender-systems, factorization-machines,
gradient-boosting, learning-to-rank, deep-interest-network, matrix-factorization,
collaborative-filtering, kuairand, llm-agents, autonomous-agents, structured-outputs
```

---

## "Try it out" links

```
https://github.com/ahsiwt101/kairos-automl-research-agent
```
