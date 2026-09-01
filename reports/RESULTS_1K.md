# Results — KuaiRand-1k (bonus benchmark)

A **transfer probe**: the same agent, the same code, pointed at a dataset it was never tuned
on. Only `KAIROS_VARIANT=1k` changed. 11.7M rows, 4.37M items, and **117× more history per
user** than Pure (5,143 rows/user vs 44).

## Result

| metric | official FM baseline (reproduced) | KAIROS | absolute delta |
|---|---|---|---|
| GAUC | 0.6428 | 0.6841 | **+0.0413** |
| nDCG@5 | 0.5285 | 0.6232 | **+0.0947** |
| **primary** | **0.5856** | **0.6536** | **+0.0680** |

`score_dataset` = mean of per-metric deltas = **+0.0680**.

**Validation 0.6522 → test 0.6536.** The test score is *higher* than validation, so the
large validation gain did not evaporate on held-out data. On a project whose central finding
is that validation gains often do evaporate, that is the number that matters most here.

The accepted candidate also passed backtest confirmation before acceptance:
`backtest_a valid 0.6414 / test 0.6440, gap −0.0026` against a 0.035 threshold.

## The run

**6 iterations · 1 accept · 0 manual interventions · 18,905 s · 147,256 tokens.**
Stopped on the self-imposed 120k token budget (`STOPPED: token budget exhausted
(147,256 >= 120,000)`), before the declared floor of 10 iterations was reached. The declared
rule (ε = 0.002, N = 5, floor 10) was in force throughout but never fired.

Iterations run ~50× slower than on Pure — 11.7M rows versus 1.4M, with each accept adding a
full confirmation pass.

## What it says about the Pure finding

Pure's decomposition says personalisation is worth only ~0.006 there, and we attributed that
to sparsity: 44 rows per user is too few to estimate a user's preferences. 1k removes
exactly that constraint, and the ceiling does move — a +0.0680 gain against 1k's own
baseline, against +0.0037 on Pure.

**But the mechanism is not the one we predicted.** The agent found the gain in the *fusion
architecture*, not in personalisation features. Iterations 3, 4 and 6 all proposed
user-conditional features — duration-affinity, popularity-preference, duration-decile
crosses — and all failed, the one that ran scoring −0.0356. So per-user history being 117×
longer did not make user-conditional features pay. The honest reading is that 1k has more
headroom, and that it is still not personalisation headroom.

## Caveat: the leak detector does not fully transfer

`_backtest_confirm` uses two signals. The **gap check** transferred cleanly (−0.0026 against
a 0.035 threshold). The **absolute-ceiling check** did not: `HONEST_CEILING` is calibrated on
Pure's backtest folds, and this candidate cleared it by only 0.0060. A threshold with no
calibration on this dataset nearly rejected an honest candidate. The gap check is what
carries the verdict, and the val→test result above is the independent confirmation.

## What the port itself found

Before producing any score, porting to 1k exposed **five latent defects in our own code** —
positional side-table indexing, an inert `RLIMIT_AS` guard, cross-variant cache collisions,
an unguarded evaluation subprocess, and a prewarm deadlock. Every one was invisible on Pure
and would have produced wrong numbers rather than an error. See `FINDINGS.md` §13.
