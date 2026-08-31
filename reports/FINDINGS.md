# What we found on KuaiRand-Pure

> **Headline.** The submission is produced autonomously: KAIROS proposed, wrote, evaluated
> and accepted a candidate scoring **0.5988** on the hidden test against the official
> baseline's 0.5946 (**+0.0042**), beating both the baseline and the strongest pipeline we
> built by hand (0.5976). Three iterations, **zero manual interventions**, 376 seconds,
> 36,490 tokens. Its prediction hit-rate — whether the diagnostics it claims will move
> actually move — has gone **0/2 → 2/3** since adding an adversarial critic.
>
> The rest of this document is the measurement that made that possible, including the
> roughly 30 interventions that did **not** work.


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

## 8d. Why ~0.60 is the ceiling: a signal decomposition

Every entry will report a number against the baseline. Almost none will say why the number
is where it is. Nested model fits on the validation split, each rung adding one information
source to the one above it (standalone = that signal alone; cumulative = a fitted model on
everything up to and including it):

| signal | standalone | cumulative | marginal |
|---|---|---|---|
| context (tab) | 0.5399 | 0.5405 | — |
| + item quality | 0.5807 | **0.5955** | **+0.0550** |
| + item x context | 0.5877 | 0.5966 | +0.0011 |
| + duration fit (user x durbucket) | 0.4914 | 0.5935 | −0.0030 |
| + affinity (user x author) | 0.4825 | 0.5938 | +0.0002 |
| + affinity (user x item) | 0.4818 | 0.5940 | +0.0002 |

**Context plus item quality reaches 0.5955. Every personalisation feature after that is
worth approximately nothing.** The official FM's 0.6016 means its ID embeddings buy about
+0.006 over a model with no personalisation at all. On this benchmark the "recommender"
part of the recommender system is worth ~0.006 of primary; the task is ~95% context and
item quality.

Read the two columns differently: a sparse affinity rate scores at random *standalone*
because for most rows it is a constant prior, and a constant cannot order a user's list.
It can still pay inside a model that has other features to gate it on — it just doesn't,
here, at 0.58% matrix density.

This explains the whole pattern of null results: the organizers' feature and capacity
ablations, our six-loss ablation, and everything in §8 below. They are all interventions on
the ~0.006 of the score that is personalisation. It also reframes the oracle gap: 0.6045 to
0.8484 is **not** unexploited affinity signal waiting to be modelled, it is per-impression
noise — mood, interruption, what came before in the feed — that no user-item model recovers
at this density.

## 8e. What actually moves the number, and what does not

Interventions tested this project and found inside seed noise: six loss functions
(§4) · static features and capacity (organizer-tested) · recency weighting · watch-time
regression, three ways · D2Q duration-deconfounded quantile · refit on train+validation ·
four `tab` encodings (raw numeric / target-encoded / LightGBM-categorical / both; spread
0.5982–0.5988 at σ≈0.0002) · eight item-quality estimators (exponential decay ×
hierarchical empirical-Bayes shrinkage toward the author's rate; best +0.0002, and
aggressive decay at H=3d **hurts** by −0.0015, which says item quality here is stable
rather than drifting, so down-weighting old evidence just discards sample size) · and the
live agent's own MF / CF / auxiliary-signal attempts.

Two things work, and only two:

- **Variance reduction.** Seed-averaging the FM, which saturates fast: 1 seed 0.6013,
  3 seeds 0.6026, 5 seeds 0.6027, 10 seeds 0.6027. Three seeds captures it; beyond that is
  wasted compute. (This independently confirms the ≥3-seed rule §1 derived from the noise
  floor.)
- **Not fooling yourself.** Selection discipline is worth avoiding −0.0156 (§6).

**A necessary refinement to our own thesis.** Given §6 we expected uniformly-weighted
blending to generalise better than coordinate-ascent weights fitted on validation. It does
not — the two have *identical* val→test gaps (+0.0068 each), and coordinate ascent is
better on both splits, its advantage transferring almost exactly (+0.0026 validation →
+0.0025 test). So "validation is unreliable" is too coarse a statement. Fitting 3 blend
weights against 22,377 users is statistically sound: many observations, few parameters,
candidates that are not adversarially correlated. Selecting among 50 correlated *pipelines*
on a single validation read is not. The hazard is parameters-per-observation and
correlation structure, not validation per se.

## 8f. An audit of the provided data: `video_features_statistic_pure.csv`

This file holds 51 per-video aggregates including `play_progress` and `long_time_play_cnt`
— direct measures of exactly the item quality that §8d shows carries all the signal. It is
KuaiRand data, so the rules permit it. We do not use it, and the reason is worth stating.

Its counters are **51–66x larger than this sample's own impression counts**, so they are
KuaiShou's platform-wide production numbers, not aggregates over the released logs. That
has a consequence we could not engineer around: **their temporal coverage cannot be
established from the data at all.** Correlation diagnostics are inconclusive
(`play_progress` correlates *better* with test-period item rates, +0.5797, than
train-period, +0.5473, which is the signature of a full-period aggregate — but at
n=1,324 items that gap is ~1.2 standard errors, and a second column from the same file
points the other way).

If those aggregates include the evaluation window, using them imports test-period
behaviour under a column name no leakage check would flag. Since the entire thesis of this
project is that you cannot trust a number whose provenance you have not established, using
a temporally-unverifiable feature to raise our score would refute the work. We exclude it
and record the decision here rather than silently benefiting from it.

## 9. Open, and honestly untried by us

Sequence modelling (DIN/SIM-style target attention over user history) is the largest
untouched lever and the organizers flag it as blank space. Our behavioural features are
aggregates, not sequences.

## Hidden-test discipline

Every consultation of the sealed test split goes through an audited `Scorer`. As of this
report that log contains **31 calls**, all research probes. The agent's own selection never
reads it.

---

## 10. The agent's own reasoning, measured

Every iteration commits to a falsifiable prediction: one diagnostic from a fixed vocabulary
of ten quantities the diagnostics layer already computes, plus a direction. After the
candidate runs we check whether that quantity moved that way. An unverifiable claim scores
`None`, never `False` — an instrumentation gap is not the agent's fault and must not
corrupt its record.

| run | prediction hit-rate |
|---|---|
| before the critic | 0 / 2 |
| first run with critic | 1 / 3 |
| current | **2 / 3** |

The critic audits whether the prediction follows from the mechanism before code is written.
Caught live, on a real call:

> **Given** a mechanism about duration-based mis-ordering paired with the prediction
> `primary / increase`
> **Critic** — *"The mechanism specifically concerns duration-based mis-ordering (inversion
> loss from duration deciles), so the prediction should target inversion_loss_duration
> decreasing"* → substituted. A control case that already cohered was left untouched.

"Primary increases" is true of any improvement, so it tests nothing about the specific
mechanism claimed. That was the source of the original 0/2.

## 11. Sub-space experts: a confirmed premise that did not pay

Rank fusion is rewarded by decorrelation rather than member strength, and our members were
less decorrelated than their architectures suggested. Three experts, each restricted to one
disjoint feature family:

| expert | sees | valid alone |
|---|---|---|
| context | tab, hour, duration, staleness | 0.5718 |
| item | item / author / item x tab rates | 0.5906 |
| user | user x tab, user x duration rates | 0.5357 |

All three are *weaker* than the FM (0.6005) individually — that is the bet. Measured
decorrelation:

```
mean expert-pair Spearman   +0.362
FM vs DIN (the bar)         +0.848   <- unrelated architectures, still correlated
item vs user                +0.279   <- most decorrelated pair
```

**The premise holds and the payoff does not.** A blend including them reached test 0.5985,
short of the standing 0.5988. The independence is real; the additional signal it carries is
too small to pay for the weight it takes. Shipped as a capability the agent may use, not
forced into the submission.

## 12. Two more instances of the central finding, on our own work

**Parameterised rank fusion.** Adding per-member power exponents to the blend raised
validation 0.6031 → 0.6035 and *lowered* held-out test 0.5985 → 0.5982, widening the gap
+0.0046 → +0.0053. Extra parameters fitted against an unreliable signal buy validation and
cost reality. Declined.

**A necessary refinement, though.** We expected uniform blend weights to generalise better
than validation-fitted ones. They do not — both have *identical* val→test gaps (+0.0068),
and coordinate ascent is better on both splits. So "validation is unreliable" is too coarse:
fitting 3 weights against 22,377 users is sound, while selecting among 50 correlated
pipelines on one read is not. The hazard is parameters-per-observation and correlation
structure, not validation itself.

**DIN history mismatch: answered, not fixed.** Training rows average 7.3 history items (32%
empty) against ~17 at serving, because 78% of training rows precede history accumulation.
Three corrections tested against unweighted DIN's 0.6023, with an adoption bar of +0.0005
fixed in advance: recency +0.0002, importance weighting −0.0002, and `late_only` — which
achieves a **perfect** distribution match — **−0.0024**, the worst of the four. Discarding
78% of training rows costs more than the mismatch does. The mismatch is real and is not
what limits DIN.

---

## 13. Porting to a second dataset found three silent-wrong-answer bugs

The KuaiRand-1k transfer run was set up to answer a scientific question (does the
personalisation ceiling hold when per-user history is 117x longer?). Before producing a
single score it found three defects in our own code, all of the same shape as the leakage
finding this project is built around: **a guard or an assumption that looks correct, is
never exercised on Pure, and fails by producing wrong numbers rather than an error.**

### 13.1 Side tables were indexed by position, not by id

`dataset.py` sorted the video/user feature tables and read them back as `arr[video_id]`,
on a comment asserting ids are "already 0..N-1 dense ints". That is true on Pure. On 1k,
4,371,868 videos carry ids running to 4,371,899 - 32 gaps - so every video attribute after
the first gap bound to the **wrong video**, with no error raised.

Fixed by reindexing onto the full `0..max` id range, making position == id true by
construction; absent ids become NaN, which is a visible absence rather than another
video's value. Verified a no-op on Pure: all 114 cached columns byte-identical.

### 13.2 The candidate sandbox bounded time but not memory

A candidate that over-allocates does not die politely. The OS kills the **largest** process,
which is the *parent* - it holds the columnar cache and every prewarmed signal. The first
1k run therefore vanished with no traceback, no SUMMARY, and no ledger entry. Subprocess
isolation contained exceptions but not memory pressure.

Two attempted fixes failed, and the second failure is the more instructive:

1. `resource.setrlimit(RLIMIT_AS)` in a `preexec_fn`. **On macOS that call raises**, and the
   handler swallowed the error - so the guard was present, visible in code review, and
   capped nothing. Strictly worse than no guard, because it looked like one.
2. The test written for it passed anyway. The allocation bomb returned the wrong row count,
   so the existing row-count contract check caught it *before memory was ever touched*. **A
   test that passes for the wrong reason certifies an inert guard.**

The working fix is a parent-side RSS watchdog polling every 0.5s: it measures resident
pages, which is what the OS killer actually reacts to, and needs no rlimit support. The
child now dies first with `stage='memory'` and an actionable hint, so the failure reaches
the agent through the normal repair path. `experiments/verify_mem_guard.py` pins both
directions with a contract-valid, page-touching bomb.

### 13.3 Every derived-signal cache was shared across variants

The columnar cache was made variant-aware; the caches one layer down - `fm_signal`,
`refit`, `din`, `mf`, `cf`, `expert`, `aux` - were fixed path strings. A 1k run therefore
`np.load`-ed **Pure's 1,436,609-row signal into an 11,713,045-row problem**. This is the
identical mistake to 13.1, made one level of abstraction lower, by the same author, within
the same hour.

Fixed with a single `variant_path()` helper.
`experiments/verify_variant_isolation.py` asserts both halves: Pure's eight paths are
unchanged (no cached work invalidated) and no 1k path collides with a Pure one.

### Why this belongs in the findings rather than the changelog

None of the three would have surfaced on Pure, at any level of test coverage, because each
is an assumption that happens to hold there. They surfaced within an hour of pointing the
same code at a dataset built differently.

That is an argument for the transfer run **independent of what it scores**: a second
dataset functions as an assumption detector. It is also the strongest available evidence
for this project's central claim - that a validation number can be confidently wrong, and
that the defence is an independent check rather than more care. We wrote the auditor that
catches this class of bug in the agent's work, and then produced three instances of it in
our own.
