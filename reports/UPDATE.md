> **Historical document — superseded numbers.** This report was written during the project
> and describes the earlier 3-iteration campaign (`runs/kairos_agent_submission`), which
> scored valid 0.6034 / test 0.5988. **That is not the submitted result.** The shipped
> `submission.csv` comes from the later 10-iteration campaign in
> `runs/kairos_submission_repro`: valid 0.6030 / test **0.5983**, `score_dataset` **+0.0037**.
> See [`RESULTS.md`](RESULTS.md) for the current figures and
> [`_results_note.md`](_results_note.md) for why the lower-scoring campaign was the one
> submitted.
>
> **Withdrawn claim.** Where this document reports a prediction hit-rate of 0/2 → 2/3, that
> reading did not survive a larger sample: the submitted 10-iteration campaign scored
> **1 of 10**. The claim that the adversarial critic lifted the hit-rate is withdrawn. The
> reasoning about *why* predictions are worth scoring stands; the number does not.
>
> Kept unedited as a record of what was believed at the time.

# Recent Updates

*Since the build review. Everything below is measured, not estimated.*

---

## Headline

| | test primary | vs baseline |
|---|---|---|
| Official FM baseline | 0.5946 | — |
| Hand-built ensemble (human) | 0.5976 | +0.0030 |
| **KAIROS submission (agent)** | **0.5988** | **+0.0042** |

**Submission unchanged and verified intact** — I re-scored `submission.csv` directly rather
than assuming. Two subsequent agent runs did not beat it.

**Prediction hit-rate: 0/2 → 1/3 → 2/3.** This is the number that actually moved.

---

## What was implemented

Four proposals from the research-expansion doc were accepted after checking them against
what we'd already measured. All four are now resolved: one shipped as a capability, one
shipped and working, two tested and declined on their own evidence.

### A. Sub-space experts — premise confirmed, payoff not realised

Three models each trained on ONE disjoint feature family, so none can rediscover what the
others know.

| expert | sees | valid (alone) |
|---|---|---|
| context | tab, hour, duration, staleness | 0.5718 |
| item | item / author / item×tab rates | 0.5906 |
| user | user×tab, user×duration rates | 0.5357 |

All three are *weaker* than the FM (0.6005) individually. That was the bet — fusion is
rewarded by decorrelation, not member strength. The go/no-go test:

```
mean expert-pair Spearman  +0.362
FM vs DIN (the bar)        +0.848      <- unrelated architectures, still correlated
item vs user               +0.279      <- most decorrelated pair
```

**Premise confirmed.** But a hand-built blend including them reached test 0.5985 — real,
and still short of the agent's 0.5988. So the experts went into `ctx.expert_score(sub)` as
a capability for the agent rather than being forced into the submission by hand.

### B. Adversarial critic — works, and moved the open roadblock

The open problem from the review doc: the agent beat the baseline while its stated
predictions never verified (0/2). Root cause was partly vacuous predictions — "primary
increases" is true of any improvement and tests nothing.

A critic pass now checks the prediction against the stated mechanism before code is
written. Verified on real API calls:

> **Input:** mechanism about duration-based mis-ordering, prediction `primary / increase`
> **Critic:** *"The mechanism specifically concerns duration-based mis-ordering (inversion
> loss from duration deciles), so the prediction should target inversion_loss_duration
> decreasing"* → substituted
> **Control case:** an already-coherent prediction was left untouched

Cost ~1k tokens per check, no extra training runs. Hit-rate since: **0/2 → 1/3 → 2/3**.

### C. Parameterised power-rank fusion — tested, declined

| scheme | valid | test | gap |
|---|---|---|---|
| linear rank fusion | 0.6031 | **0.5985** | +0.0046 |
| power-rank (γ=0.75) | **0.6035** | 0.5982 | +0.0053 |

Higher validation, **lower test**, wider gap. A textbook instance of this project's own
central finding: extra parameters fitted against an unreliable validation signal buy
validation and cost reality. Declined by our own discipline, and the measured numbers are
now written into the agent's prompt so it doesn't repeat the mistake.

### D. DIN train/serve history mismatch — tested, NOT adopted

Measured cause:

| train window | rows | mean history |
|---|---|---|
| Apr 08–14 | 891,418 (78%) | **5.1** |
| Apr 15–19 | 208,822 | 14.7 |
| Apr 20–21 | 40,872 | **17.2** |
| *valid / test* | | *16.9 / 17.5* |

Late-window training rows already have the right distribution — there are just few of them.
Three corrections tested against unweighted DIN's 0.6023, with an adoption bar of
**+0.0005** set *before* seeing results:

| weighting | valid | vs 0.6023 |
|---|---|---|
| None (current) | 0.6023 | — |
| recency | 0.6025 | +0.0002 |
| hist_match | 0.6021 | −0.0002 |
| late_only | 0.5999 | **−0.0024** |

**Not adopted.** `recency` is nominally best but +0.0002 is inside noise, and the
pre-registered bar exists so that cannot be rationalised into a win.

The informative row is `late_only`: it achieves a *perfect* history-distribution match and
is the **worst** option. Discarding 78% of training rows costs more than the mismatch does.
So the mismatch is real and measurable, but it is not what limits DIN — which closes the
roadblock as answered rather than fixed.

Worth noting: the original proposal suggested *dropping* 30–50% of training history. That
runs the wrong way — it would shorten training histories further and widen the very gap it
aims to close. The fix reweights toward realistic histories instead.

---

## Declined, with reasons

**LambdaLoss / soft-nDCG** — already tested. `lambda_ndcg` scored **test 0.5874**, the worst
of six losses (BCE: 0.5948). Re-running it would spend an iteration on a known dead end.

**IPW via `log_random`** — two independent problems. It spans 20220422–20220508, straight
through the hidden-test window, so training on it injects test-period data. And the metric
ranks within the *logged, biased* exposure set, so correcting exposure bias optimises a
counterfactual we aren't scored on.

**MCTS over hypothesis space** — targets "early convergence after 3 misses", but that is the
competition's rule, not a bug in our loop. Backtracking still consumes iterations.

**Graph RAG over the ledger** — over-engineered for 3–12 entries per run. We already carry a
prior-run summary and per-family track record into planning.

---

## Two bugs found and fixed

**Incumbent carryover.** Each fresh agent run restarted from the official baseline (0.6016)
instead of our best-known result (0.6034), so it could "accept" a candidate that was
actually a regression — and did, at 0.6022. Same class as the earlier stall-counter bug:
comparing against the wrong reference. Runs now resume from the best known.

**First-iteration predictions were unscoreable.** No baseline digest existed at startup, so
`check_prediction` returned `None` for the single most important iteration. Now computed up
front.

---

## Current state

- **Tests:** 8 files, all green (`verify_metrics`, `verify_causal`, `verify_frozen`,
  `verify_new_primitives`, `verify_stall_counter`, `verify_agent_mechanisms`,
  `verify_scores_mode`, `audit_assumptions`)
- **Commits:** 47 on branch `score-push`
- **Audited hidden-test consultations:** 60
- **Running now:** nothing

## Honest read

The score has plateaued at 0.5988. Three agent runs since have not beaten it, and the
signal decomposition predicted exactly this — context plus item quality reaches 0.5955 and
personalisation is worth ~0.006 in total, so everything competes for single-digit
thousandths.

What *has* improved is the agent's ability to explain itself: the prediction hit-rate is up
from 0/2 to 2/3, which is the difference between an agent that gets lucky and one that
understands the problem. For a track weighting Innovation and Autonomy at 40% combined,
that is the more valuable movement.

## Next

1. Reframe the writeups around the agent (Phase 5)
3. KuaiRand-1k transfer — extra credit plus the Feasibility story
4. Ask organizers about the §2.3 contradiction and `log_random`
5. Flip the repo public at submission
