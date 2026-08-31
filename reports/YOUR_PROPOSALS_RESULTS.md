# Results: the research-expansion proposals

*Every one of the 12 proposals, checked against measurement. Four were implemented and
tested, four declined on existing evidence, four judged marginal and not scheduled.*

**Bottom line: 1 of 12 produced a measurable improvement — and it was the agent-architecture
one, not any of the modelling ones.**

---

## Scorecard

| # | Proposal | Verdict | Outcome |
|---|---|---|---|
| 4.2 | Adversarial LLM auditor | ✅ **WORKED** | prediction hit-rate 0/2 → **2/3** |
| 2.3 | De-correlated sub-space models | ⚠️ premise right, no gain | decorrelation confirmed (+0.362 vs +0.848); blend 0.5985 < incumbent 0.5988 |
| 3.1 | Parameterised power-rank fusion | ❌ **HURT** | valid ↑ 0.6035, test ↓ 0.5982 |
| 1.3 | HistDrop sequence masking | ❌ backwards + null | direction inverted; corrected version still null |
| 2.1 | LambdaLoss / soft-nDCG | ❌ already tested | test 0.5874 — worst of six losses |
| 1.2 | IPW via `log_random` | ❌ rules + method | 76% of that file is inside the test window |
| 4.1 | MCTS over hypothesis space | ❌ wrong target | early stop is the *rules*, not a bug |
| 4.3 | Graph RAG over ledger | ❌ over-engineered | ledger holds 3–12 entries per run |
| 1.1 | OOF target encoding + noise | ➖ mostly already done | we use frozen-window encoding w/ beta smoothing |
| 2.2 | Hard negative mining | ➖ not scheduled | every loss/sampling change so far has been null |
| 3.2 | Monotone stacking | ➖ not scheduled | marginal over coordinate ascent |
| 5 | KuaiRand-1k transfer | ⏳ not attempted | still the best remaining extra-credit item |

---

## Your roadmap predictions vs measured reality

You gave expected impacts. Here's how they landed:

| Priority | Your estimate | Measured |
|---|---|---|
| 1 · Parameterised rank fusion | +0.0005 to +0.0015 | **−0.0003** on test |
| 2 · De-correlated sub-space models | +0.0010 to +0.0020 | **0.0000** (premise confirmed, no score gain) |
| 3 · HistDrop in DIN | "address mismatch" | mismatch real; **fixing it doesn't help** |
| 4 · Adversarial auditor | "improve hit-rate" | **hit-rate 0/2 → 2/3** ✅ |
| 5 · KuaiRand-1k | extra credit | not attempted |

The modelling estimates were optimistic by roughly the amount the signal decomposition
predicts they would be. The agent-side estimate was correct.

---

## What worked

### 4.2 · Adversarial LLM auditor ✅

This targeted a real open roadblock: the agent was beating the baseline while its stated
predictions never verified (0/2). Part of the cause was *vacuous* predictions — "primary
increases" is true of any improvement and therefore tests nothing about the mechanism
claimed.

A critic pass now checks the prediction against the stated mechanism before any code is
written. Verified on live API calls:

> **Input** — mechanism about duration-based mis-ordering, prediction `primary / increase`
> **Critic** — *"The mechanism specifically concerns duration-based mis-ordering (inversion
> loss from duration deciles), so the prediction should target inversion_loss_duration
> decreasing"* → substituted
> **Control** — an already-coherent prediction was left untouched

**Result: hit-rate 0/2 → 1/3 → 2/3 across successive runs.** Cost ~1k tokens per check, no
extra training runs.

One change from your version: you had Sonnet proposing and Opus critiquing. Ours is Opus
planning, so the critic is the *cheaper* model auditing the more expensive one's
reasoning — the right way round for a consistency check, and cheaper.

---

## What was right in principle but didn't pay

### 2.3 · De-correlated sub-space models ⚠️

**Your premise was correct and worth confirming.** Three experts, each on a disjoint feature
family:

| expert | sees | valid alone |
|---|---|---|
| context | tab, hour, duration, staleness | 0.5718 |
| item | item / author / item×tab rates | 0.5906 |
| user | user×tab, user×duration rates | 0.5357 |

All *weaker* than the FM (0.6005) individually — which is the bet, since fusion rewards
decorrelation rather than strength. The go/no-go:

```
mean expert-pair Spearman   +0.362
FM vs DIN (the bar)         +0.848   <- unrelated architectures, still correlated
item vs user                +0.279   <- most decorrelated pair
```

**The decorrelation is real and large.** But a blend including them reached test **0.5985**,
short of the standing **0.5988**. So the independence exists and doesn't convert to score —
the experts' errors are decorrelated, but the extra signal they carry is small enough that
weighting them in costs as much as it adds.

Shipped as `ctx.expert_score(sub)` for the agent to use, rather than forced into the
submission by hand.

---

## What actively hurt

### 3.1 · Parameterised power-rank fusion ❌

| scheme | valid | test | gap |
|---|---|---|---|
| linear rank fusion | 0.6031 | **0.5985** | +0.0046 |
| power-rank (γ=0.75) | **0.6035** | 0.5982 | +0.0053 |

Higher validation, **lower test**, wider gap. This is a textbook instance of the project's
own central finding — extra parameters fitted against an unreliable validation signal buy
validation and cost reality. Declined by our own discipline, and the measured numbers are
now in the agent's prompt so it doesn't repeat the mistake.

### 1.3 · HistDrop sequence masking ❌

**The diagnosis was right; the prescription ran backwards.** You proposed dropping 30–50% of
training history. But our mismatch is train-**short** / serve-**long**:

| train window | rows | mean history |
|---|---|---|
| Apr 08–14 | 891,418 (78%) | **5.1** |
| Apr 15–19 | 208,822 | 14.7 |
| Apr 20–21 | 40,872 | **17.2** |
| *valid / test* | | *16.9 / 17.5* |

Dropping history in training would shorten it further and **widen** the gap it aims to
close. So we tested the corrected direction — reweighting *toward* realistic histories:

| weighting | valid | vs 0.6023 |
|---|---|---|
| None (current) | 0.6023 | — |
| recency | 0.6025 | +0.0002 |
| hist_match (importance weighting) | 0.6021 | −0.0002 |
| late_only (perfect distribution match) | 0.5999 | **−0.0024** |

Not adopted — the adoption bar of +0.0005 was set *before* seeing results, so +0.0002
couldn't be rationalised into a win.

The informative row is `late_only`: it achieves a **perfect** history-distribution match and
is the **worst** option. Discarding 78% of training rows costs more than the mismatch does.
**The mismatch is real but is not what limits DIN** — which answers the roadblock rather
than fixing it.

---

## Declined before testing, with reasons

### 2.1 · LambdaLoss / soft-nDCG — already tested, and it lost

From `exp02_loss_ablation.json`:

| loss | valid | test |
|---|---|---|
| bce (baseline objective) | 0.6010 | 0.5948 |
| bpr_gauc (metric-exact) | 0.6010 | 0.5950 |
| **lambda_ndcg** | 0.5936 | **0.5874** |

Worst of six losses. Re-running it would have spent an iteration on a known dead end.

### 1.2 · IPW via `log_random` — two independent problems

**Rules.** The file spans 20220422–20220508, straight through the hidden-test window:

| period | rows | |
|---|---|---|
| validation window (Apr 22–28) | 288,338 | 24% |
| **test window (Apr 29–May 8)** | **897,721** | **76%** |

For scale, our whole training split is 1,141,112 rows — this would roughly double it, with
three-quarters coming from the period we're scored on.

**Method.** More fundamentally, the metric ranks within the **logged, biased** exposure set.
Correcting exposure bias optimises a counterfactual uniform-exposure world we are not
scored on — so IPW is as likely to hurt as help even where it's legal.

*(The 24% inside the validation window may genuinely be trainable, since we're allowed
everything up to Apr 28. That's one of two questions worth putting to the organizers.)*

### 4.1 · MCTS over hypothesis space — targets a rule, not a bug

It aims at "early convergence after 3 misses." But that isn't a flaw in our loop — it's the
competition's stated rule: three consecutive iterations without +0.002 ends the run.
Backtracking to a parent node still consumes an iteration and still increments the stall
counter. MCTS would reallocate exploration without buying a single extra iteration.

### 4.3 · Graph RAG over the ledger — over-engineered for the scale

A run's ledger holds 3–12 entries. We already carry a prior-run summary into the digest and
a per-family track record into planning — the same idea at the right weight for the data
volume.

---

## Two bugs your proposals surfaced indirectly

Testing these turned up two real defects, both the same shape — comparing against the wrong
reference:

1. **Incumbent carryover.** Each fresh agent run restarted from the official baseline
   (0.6016) rather than our best-known result (0.6034), so it could "accept" a candidate
   that was actually a regression — and did, at 0.6022. Fixed.
2. **First-iteration predictions were unscoreable.** No baseline digest existed at startup,
   so the single most important prediction always scored `None`. Fixed.

---

## The meta-lesson

Your four modelling proposals (1.1, 1.3, 2.1, 2.3, 3.1) produced **zero** measurable score
gain between them. Your one agent-architecture proposal (4.2) produced the only improvement.

That is exactly what the signal decomposition predicts: context plus item quality reaches
0.5955, personalisation is worth ~0.006 in total, and everything else competes for
single-digit thousandths. **The modelling surface is saturated; the agent is where the
remaining value is** — which is also where 65% of the rubric sits.

---

## Current state

- **Submission:** 0.5988 (+0.0042 over baseline), agent-produced, verified intact
- **Prediction hit-rate:** 0/2 → 1/3 → **2/3**
- **Tests:** 8 files, all green
- **Commits:** 48 on `score-push`
- **Running:** nothing

## Still open

1. Reframe the writeups around the agent
2. **KuaiRand-1k transfer** — your #5, and the best remaining extra-credit item
3. Ask organizers: the §2.3 metric contradiction, and whether the validation-window slice of
   `log_random` is trainable
4. Flip the repo public at submission
