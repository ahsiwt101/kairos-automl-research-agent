# Results — KuaiRand-1k (bonus benchmark)

A **transfer probe**: the same agent, the same code, the same declared convergence rule,
with only `KAIROS_VARIANT=1k` changed. No retuning, and a prior
(`PRIOR_1K` in `kairos/agent/prior.py`) deliberately stripped of every empirical conclusion
measured on Pure — it carries only mechanism, a theorem about the metric, the leakage rule,
and which primitives are wired. Feeding Pure's conclusions in would have answered "do Pure's
findings hold?" while looking like "does the agent generalise?".

## Result

| metric | official FM baseline | KAIROS | absolute delta |
|---|---|---|---|
| GAUC | 0.6428 | 0.6841 | **+0.0413** |
| nDCG@5 | 0.5285 | 0.6232 | **+0.0947** |
| **primary** | **0.5856** | **0.6536** | **+0.0680** |

Validation 0.5778 → **0.6522** (+0.0744). Accepted at iteration 2, backtest-confirmed.

**6 iterations · 1 accept · 0 manual interventions · 18,905 s · 147,256 tokens.**
Stopped on its 120k token budget, before the declared 10-iteration floor.

## Why this is not a leak

A +0.0680 gain is exactly the shape that should trigger suspicion, so it was checked three
ways rather than assumed:

1. **Backtest confirmation** — re-running the candidate's own code on `backtest_a`:
   valid 0.6414 / test 0.6440, **gap −0.0026** against a 0.035 threshold. A leak inflates
   validation relative to test; this gap is *negative*.
2. **The official fold agrees** — validation 0.6522, test **0.6536**. Test is *higher* than
   validation. Within-window label feedback cannot produce that.
3. **The mechanism predicts it.** Our Pure decomposition found personalisation worth only
   ~0.006 because 44 rows per user is too few to estimate a user's preferences. 1k carries
   **5,143 rows per user** — 117× more history. The constraint the Pure finding blamed is
   the one 1k removes.

**Caveat, stated because it nearly mattered.** The absolute-ceiling half of the leak
detector is calibrated on *Pure's* backtest folds (`HONEST_CEILING = 0.60` for
`backtest_a`). On 1k it was cleared by only **0.0060**. That threshold has no 1k calibration
behind it, so the *gap* check is what carries the verdict here. The detector's gap check
transfers; its ceiling check does not.

## What the agent found, and what it did not

The accepted move was architectural: abandon the single downstream tree and fuse
decorrelated model outputs at rank level — the same family that won on Pure, rediscovered
without being told, since `PRIOR_1K` contains no Pure results.

What did **not** work is as informative. Iterations 3, 4 and 6 all proposed *personalisation*
features — user × duration-decile affinity, user duration-preference, popularity-preference
— and all three failed (the one that ran scored −0.0356). So even on a dataset with 117×
more history per user, hand-specified personalisation features did not pay; the gain came
from the fusion architecture instead.

That is a more interesting result than a clean "personalisation works when you have data",
and it is the honest reading of this trajectory.

## What the port itself found

Before producing any score, porting to 1k exposed **five latent defects in our own code**,
every one invisible on Pure and silently wrong rather than loud: side tables indexed by
position rather than id; an inert `RLIMIT_AS` memory guard whose error we swallowed;
signal caches shared across variants; an unguarded evaluation subprocess; and a prewarm
deadlock from loading torch and LightGBM into one process. See `FINDINGS.md` §13.

A second dataset is an assumption detector, independent of what it scores.
