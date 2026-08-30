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

## 6b. The leak inverts the ranking of design choices

This is the sharpest consequence, and it is worse than losing points. On the official fold,
over a pool that varies features *and* objective:

| pipeline | valid | test |
|---|---|---|
| causal features + LambdaRank | **0.7330** (best on validation) | **0.5790** (worst on test) |
| causal features + binary | 0.7170 | 0.5916 |
| frozen features + binary | 0.5987 | 0.5910 |
| frozen features + LambdaRank | 0.5973 | **0.5921** (best on test) |

With leaky features, LambdaRank looks like the strongest idea in the pool and is in fact
the weakest. With honest features, LambdaRank genuinely *is* the strongest. So a greedy
agent does not merely pick a worse candidate — it draws the **opposite conclusion about
its own objective**, and carries that wrong lesson into every subsequent iteration.

Greedy argmax-validation obtains 0.5790 where 0.5921 was available: a regret of **0.0131**,
and a submission 0.0156 *below* the baseline it set out to beat.

Note also that this reverses our own earlier finding (§4) that objective choice does not
matter. It does not matter for the ID-embedding FM on the baseline feature set; it matters
once behavioural features are present, and its sign depends on whether those features are
constructed honestly.

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

## 8b. Declining a gain we could not verify

Refitting the final model on train+validation is standard practice and well motivated here:
test sits 8-17 days past the end of training, and validation is 7 more days of data lying
closer to it. But you cannot check it directly - training on validation destroys the only
signal you would check it with.

So we validated the *procedure* on backtest folds instead, using the boosting round count
chosen by the train-only run (the only honest way to pick it):

| fold | train-only | refit on train+valid | delta |
|---|---|---|---|
| backtest_a | 0.5935 | 0.5936 | +0.0000 |
| backtest_c | 0.5698 | 0.5698 | +0.0000 |
| official (reference only) | 0.5910 | 0.5925 | **+0.0015** |

The official fold dangles +0.0015. Both backtests say the procedure is worth exactly zero,
so we **declined it**. This is the same rule that governs everything else here: do not bank
a gain that is only visible in the place you have already shown you cannot trust. Had we
taken it, we would have been doing precisely what §6 warns against.

## 8c. The live agent, end to end

Everything above was measured with scripted or hand-directed pipelines. The actual
deliverable is autonomous: Claude Opus 5 plans, Claude Sonnet 5 codes, against the
kernel described here, with zero manual intervention in any accept/reject decision.

**A leak got through, and the fix generalises beyond this one shape.** One live run's
accepted candidate hand-rolled its own streaming aggregate directly over `ctx.data.y_raw`/
`time_ms` (never touching the flagged-dangerous `causal_prefix`), keyed on user x author /
user x tag / user x duration. It scored +0.0936 on validation and was accepted - the
existing structural check (`check_user_constancy`) only inspects columns *named*
`user_rate`/`user_logn` and, even name-blind, cannot flag a cross feature, since a cross is
*supposed* to vary within a user's list. The fix does not try to understand the
candidate's code at all: any implausible validation jump gets the same candidate re-run
against a backtest fold with a genuinely unsealed test split, checked on two independent
signals - the valid/test **gap** (catches the earlier-characterised per-fold-horizon leak:
`causal_all` on backtest_a, gap +0.124) and the **absolute level** against the best honest
score ever measured on that fold (catches this leak specifically: it inflates valid *and*
test roughly equally, gap only -0.0016, but +0.037 over the empirical ceiling - a leak that
is blind to fold boundaries entirely evades a gap-only check). Verified both ways: the
actual leaked candidate is rejected, a known-honest candidate is not.

**The action space had to be extended to let the agent execute its own best idea.** Every
early live attempt converged on the same losing architecture - concatenate
`baseline_score`, `cf_score`, `auxiliary_signal`, `mf_factors` into one feature matrix for
a single downstream tree - which the agent itself correctly diagnosed as the
"tree-on-a-calibrated-score pathology" (a tree shatters a smooth continuous score into
step functions) without ever finding a fix within that architecture. The harness could not
express the winning alternative (train separate models, blend their *output ranks*) at
all - `build(ctx)` only ever returned a feature matrix for the harness's own single GBDT.
Adding `train_cfg={'mode': 'scores'}` - a candidate may return already-blended final
scores, having trained and combined its own model(s) inside `build()` - unlocked it. This
required closing a latent risk before it could be trusted: `baseline_score`/`auxiliary_signal`
lazily import torch, and a scores-mode candidate is free to `import lightgbm`; the two
crash if loaded into the same process (each bundles its own OpenMP runtime). Fixed by
pre-computing every torch-backed primitive in the trusted parent process before any
candidate runs, so a candidate's `ctx` access is always a cache read.

**Cross-run memory mattered more than any single prompt change.** Each `Kairos` run starts
a blank ledger, so across three clean runs the agent kept independently re-deriving and
re-losing the same feature-concatenation idea, never once trying `mode='scores'` live
despite it being documented as the winning strategy. Seeding the next run's first digest
with a factual one-paragraph summary of what earlier runs tried and how it went - the way
a lab notebook would - changed this immediately: iteration 1 of the next run opened with
"switch architecture entirely: use train_cfg mode='scores'… this is the documented
hand-win architecture that has never been run live."

**Result.** Once unblocked, the agent independently rediscovered the FM + frozen-history
model rank-fusion architecture - the exact recipe that won by hand - across three
consecutive live iterations, closing the gap to the baseline monotonically each time
(delta vs. incumbent: -0.0024, -0.0017, -0.0011) before correctly stalling out at the
competition's own N=3 rule. It never crossed 0.6016 on validation within that budget, so
per the harness's own discipline (never ship what does not beat the incumbent) it retained
the FM baseline rather than accept a candidate that was still short. `submission.csv` is
unaffected by any of this - it was last written from the hand-built ensemble and no
live-agent candidate ever beat the incumbent it would need to beat to replace it.

**Two real bugs the live runs caught in the harness itself**, beyond the leak above:
- `Ledger.stall_counter` seeded its "best score" tracker at -inf instead of the actual
  incumbent, letting a run earn false progress credit for beating its own early bad
  guesses - it would have continued well past where the stated rule should have ended it.
  Fixed and pinned with a regression test; the already-published control-arm ablation
  (§6) was re-run under the correction (conclusion unchanged, iteration count corrected).
- An invalid `train_cfg.hparams` key raised inside the *scoring* subprocess, which has no
  repair loop wired to it, silently discarding an entire iteration - including a `build()`
  that had run correctly - over what should have been a one-line, repairable fix (the
  coder used `n_estimators`, an XGBoost/sklearn name, not LightGBM's). Moved the
  validation into the same subprocess and retry loop as the candidate's own code.

## 9. Open, and honestly untried by us

Sequence modelling (DIN/SIM-style target attention over user history) is the largest
untouched lever and the organizers flag it as blank space. Our behavioural features are
aggregates, not sequences.

## Hidden-test discipline

Every consultation of the sealed test split goes through an audited `Scorer`. As of this
report that log contains **31 calls**, all research probes. The agent's own selection never
reads it.
