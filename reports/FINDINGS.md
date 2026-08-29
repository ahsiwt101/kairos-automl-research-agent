# What we found on KuaiRand-Pure

All numbers below are reproducible from `experiments/`. The official `evaluate.py` is the
sole scoring authority and is committed unmodified. Our vectorised evaluator is verified
identical to it to 4.4e-16 across seven stress cases including heavy ties.

## 0. Reproduction, verified rather than asserted

Seeds 0–4, the published protocol:

| | ours | published |
|---|---|---|
| valid primary | 0.6016 | 0.6016 |
| test primary | 0.5946 | 0.5946 |
| test std (5 seeds) | 0.0008 | 0.0008 |
| test GAUC / nDCG@5 | 0.6610 / 0.5282 | 0.6610 / 0.5282 |
| random baseline | 0.4753 | 0.4753 |

Row order is byte-identical to `data.load()` on all three splits — checked because
submission alignment is positional and `(user_id, video_id)` is not unique.

## 1. The noise floor sits under the decision threshold

Per-seed std is **0.0008**; observed 5-seed spread on test is 0.5931–0.5953. The
competition's convergence rule fires on a **+0.002** validation improvement. A single-seed
measurement therefore cannot reliably resolve the very improvement the rules require
(z≈2.5). Three seeds put the standard error at 0.00046 (z≈4.3).

Consequence for agent design: **≥3 seeds per candidate is mandatory, not a nicety.** A
single-seed agent will both accept noise and stall out on real gains. We retracted two of
our own conclusions when re-run with seeds (recency weighting; part of the loss ablation).

## 2. The logging density collapses 5× mid-window

| period | rows/day | impressions/user/day |
|---|---|---|
| Apr 10–12 | 228k–279k | 11–13 |
| Apr 18 onward | 14k–20k | ~2 |

59% of training rows sit in three days that look nothing like the evaluation regime, and
both valid and test are entirely in the sparse regime. Positive rate is stable (0.31–0.34),
so this is exposure drift, not label drift. Median evaluation list size is 4–5.

## 3. There is no failure pocket — the model is uniformly mediocre

Diagnosing the baseline FM by slice, per-user AUC is flat everywhere:

| slice | AUC |
|---|---|
| least-active user decile | 0.675 |
| most-active user decile | 0.685 |
| list size 2 | 0.636 |
| list size 12 | 0.675 |
| cold items | contribute **0.0000** GAUC loss (every eval item was seen in training) |

Inversion attribution is near-uniform across duration and popularity deciles.
Spearman(FM, item popularity) = 0.69 — FM is largely a popularity model with a thin
personalisation layer. This explains why the organizers' feature and capacity ablations
found nothing: those interventions reshape how you fit signal the feature set lacks.

## 4. Objective alignment does nothing here

Identical features, capacity, optimiser; only the loss changes (single seed, so only
differences >0.002 are meaningful):

| loss | valid | test |
|---|---|---|
| bce (baseline objective) | 0.6010 | 0.5948 |
| bpr_gauc (metric-exact) | 0.6010 | 0.5950 |
| primary (GAUC+nDCG surrogate) | 0.6009 | 0.5954 |
| listnet | 0.5973 | 0.5917 |
| lambda_ndcg | 0.5936 | 0.5874 |

`bpr_gauc` is derived: expanding GAUC shows the per-user positive-count weight cancels, so
the metric-exact pairwise weight is `1/n_neg(g)`, not uniform and not `1/(n_pos·n_neg)`.
It ties plain BCE. The organizers' ranked hypothesis #1 does not hold on this benchmark.

## 5. Validation is systematically corrupted — the central finding

Build history features the natural way — a user's/item's `long_view` rate via a correct
time-ordered prefix that never sees its own label — and:

| pipeline | valid | test | gap |
|---|---|---|---|
| FM baseline | 0.6016 | 0.5946 | +0.0070 |
| lgb + causal features, binary | 0.6158 | 0.5904 | +0.0254 |
| lgb + causal features, lambdarank/user_day | **0.7158** | **0.5749** | **+0.1409** |

Validation reaches 0.7158 against a validation oracle of 0.8484, while the hidden-test
score falls **below the baseline being attacked**.

**Mechanism.** Evaluation ranks each user's impressions *as a set*. A time-ordered history
feature lets a validation row see the labels of its own list-mates — exactly the quantity
the metric asks the model to predict, and unavailable for test rows, whose labels stop at
the horizon. Streaming-causality is correct for a live system and is the *wrong model of
this task*.

**Detection, structurally.** Within-user ranking is invariant to any quantity constant
across a user's list, so a *user-level* statistic must have exactly zero within-user
variance. Non-zero variance is a proof of label feedback, not a hint. Under the naive
construction `max|user_rate − mean(user_rate)|` is 1.24e-01; under the fix it is
**0.000e+00**.

**Fix.** Freeze every aggregate at the start of its evaluation window. The gap collapses
from +0.1409 to +0.0070, matching the baseline's own gap.

## 6. How much the corruption costs depends on the objective

This is where our first framing was wrong, and the correction matters.

Over a 10-pipeline pool varying only features (binary objective throughout), greedy
argmax-validation and transfer-corrected selection pick **the same hidden-test score**
(0.5907, regret 0.0005 each). A corrupted measurement costs nothing when every candidate
is genuinely equivalent — and that pool's test scores span only 0.5885–0.5913.

What the same pool *does* show decisively:

| selection signal | rank correlation with hidden test |
|---|---|
| official validation | **+0.297** |
| backtest-fold transfer | **+0.685** |

So validation is a poor ranking signal and backtest transfer is a much better one. The
score cost only materialises when the pool contains candidates whose true scores differ —
which happens once the objective axis is included, since lambdarank over per-day groups is
where the leak turns destructive (test 0.5749) rather than merely misleading.

## 7. What actually improved the score

| | valid | test | vs baseline |
|---|---|---|---|
| FM baseline | 0.6016 | 0.5946 | — |
| FM, 3-seed rank-averaged | 0.6026 | 0.5963 | +0.0017 |
| **ensemble (FM variants + frozen GBDT + watch-time binary)** | **0.6045** | **0.5976** | **+0.0030** |

Rank handling is easy to get wrong: convert each *seed's* raw prediction to within-user
percentile ranks, average those per model, then blend across models. Re-ranking an
already-averaged rank vector discards what seed-averaging created (this bug silently
degenerated one of our ensembles to "FM alone").

## 8. Directions that did not work

- **Watch-time regression.** `long_view` is empirically a duration-dependent threshold on
  watch ratio, cleanly separable in 9 of 10 duration deciles, and a *perfect* watch-ratio
  predictor would score 0.8023 against the label oracle's 0.8484. Despite that, regressing
  log play-time and ranking by margin against the threshold scores 0.5605 (L2) / 0.5754
  (Huber) versus 0.5917 for the binary target on identical features. The D2Q-style
  duration-deconfounded quantile target also loses (0.5846). The information is present;
  predicting magnitude well enough to rank threshold-crossings is harder than predicting
  the crossing directly.
- **Recency weighting**, despite the density collapse: 0.6017 → 0.6020 over 3 seeds.
- **Static features and capacity** — organizer-tested, confirmed not worth re-running.

## 9. Open, and honestly untried by us

Sequence modelling (DIN/SIM-style target attention over user history) is the largest
untouched lever and the organizers flag it as blank space. Our behavioural features are
aggregates, not sequences.

## Hidden-test discipline

Every consultation of the sealed test split goes through an audited `Scorer`. As of this
report that log contains **31 calls**, all research probes. The agent's own selection never
reads it.
