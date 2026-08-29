# KAIROS — an autonomous ML research agent that knows when its validation set is lying

**TikTok TechJam 2026 — Track 2: Autonomous ML Research Agent for Recommender Systems**
Benchmark: KuaiRand-Pure, within-user ranking on `long_view`, primary = mean(GAUC, nDCG@5).

---

## Overview

Most agents for this task are a loop: propose a change, train, read the validation score,
keep it if it went up. On this benchmark that loop is actively harmful, and we can measure
how harmful.

A pipeline built the natural way — summarise each user's and item's history as a
long_view rate, computed with a correct time-ordered prefix so no row sees its own label —
reaches **validation 0.7158**, close to the 0.8484 validation oracle, while its **hidden-test
score falls to 0.5749**, below the 0.5946 baseline it was trying to beat. Validation gain
and test gain are *anti-correlated* in this regime.

The cause is structural. Evaluation ranks each user's impression list **as a set**. A
time-ordered history feature lets a validation row see the labels of its own list-mates —
which is exactly the quantity the metric asks the model to predict, and which does not
exist for test rows, whose labels stop at the horizon. The fix is to freeze every
aggregate at the **start of its evaluation window**, which collapses the validation→test
gap from +0.1409 to +0.0070.

KAIROS is built around that finding. Its distinguishing component is not its search but
its **temporal-validity auditor**, which vetoes a candidate before its score is believed,
and its **selection rule**, which chooses on transfer across backtest folds rather than
argmax over a single validation set.

## What is in here

| path | what it is |
|---|---|
| `kairos/kernel/fastmetrics.py` | vectorised GAUC / nDCG@5, exact to 4.4e-16 vs the official `evaluate.py`, 12x faster |
| `kairos/kernel/dataset.py` | cached columnar loader; test labels sealed behind an audited `Scorer` |
| `kairos/kernel/causal.py` | streaming prefix aggregates **and** window-frozen aggregates |
| `kairos/kernel/frozenfeat.py` | deployment-faithful feature matrix + per-fold window schedules |
| `kairos/kernel/diagnostics.py` | per-slice headroom attribution and pairwise inversion attribution |
| `kairos/kernel/candidates.py` | pipeline pool spanning the leaky↔honest axis |
| `kairos/agent/auditor.py` | temporal-validity checks; BLOCK vetoes a candidate |
| `kairos/agent/sandbox.py` | AST allowlist + subprocess isolation for LLM-authored code |
| `kairos/agent/selection.py` | transfer / stability / shrinkage corrections to argmax |
| `kairos/agent/ledger.py` | per-iteration run log; also the stall-budget accounting |
| `kairos/agent/loop.py` | the controller |
| `experiments/` | every experiment and every contract test |

The official starter kit (`evaluate.py`, `data.py`, `baseline.py`, `submit.py`) is
committed **unmodified**; `evaluate.py` is the sole authority on scoring.

## Setup

```bash
# dataset (194 MB, no registration)
curl -LO https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar xzf KuaiRand-Pure.tar.gz

python3 -m venv .venv
./.venv/bin/pip install numpy scipy pandas pyarrow lightgbm torch anthropic
# macOS only: lightgbm needs libomp
brew install libomp
```

Note: `torch` and `lightgbm` each bundle their own OpenMP runtime and abort if imported
into the same process. Everything here keeps them in separate processes. The documented
`KMP_DUPLICATE_LIB_OK` workaround is deliberately **not** used — it can silently produce
wrong numbers, which is not acceptable on a benchmark.

## Reproducing

```bash
# contract tests first - these are the guarantees everything else rests on
./.venv/bin/python experiments/verify_metrics.py       # fast metrics == official evaluate.py
./.venv/bin/python experiments/verify_causal.py        # prefix aggregates vs brute force
./.venv/bin/python experiments/verify_frozen.py        # frozen aggregates vs brute force
./.venv/bin/python experiments/audit_assumptions.py    # silent-failure assumptions
./.venv/bin/python experiments/verify_baseline.py      # 5-seed reproduction of the baseline

# findings
./.venv/bin/python experiments/exp02_loss_ablation.py  # objective alignment: no effect
./.venv/bin/python experiments/exp03_diagnose.py       # where the metric is actually lost
./.venv/bin/python experiments/exp06_frozen.py         # the leak, and the fix
./.venv/bin/python experiments/exp11_selection.py      # selection-rule comparison

# the agent
./.venv/bin/python experiments/exp08_agent_smoke.py    # end-to-end, incl. catch-and-recover
```

## Verified reproduction of the official baseline

Seeds 0–4, matching the published protocol:

| | ours | published |
|---|---|---|
| valid primary | 0.6016 | 0.6016 |
| test primary | 0.5946 | 0.5946 |
| test std (5 seeds) | 0.0008 | 0.0008 |
| test GAUC / nDCG@5 | 0.6610 / 0.5282 | 0.6610 / 0.5282 |

Row order is also byte-identical to `data.load()` on all three splits — checked because
submission alignment is positional and `(user_id, video_id)` is not unique (3.06% of test
rows are repeated pairs).

## Limitations, honestly

- **Absolute gains are modest.** The scored improvement comes mostly from ensembling and
  disciplined selection, not from a stronger model. Several well-motivated directions
  produced nothing measurable: objective alignment (all six losses within seed noise),
  recency weighting, and watch-time regression as a ranking target.
- **The noise floor is close to the decision threshold.** Per-seed std is 0.0008 while the
  competition's convergence rule needs +0.002, so single-seed conclusions are unsafe here.
  Every claim above uses ≥3 seeds; two of our own early conclusions were retracted when
  re-run with seeds.
- **Sequence modelling is unexplored by us too.** DIN/SIM-style target attention over user
  history is the largest untried direction, and the organizers flag it as blank space.
- **Backtest folds are imperfect analogues.** They live inside the public-label region, so
  their test windows are 6–7 days against the official 10, and their training windows are
  shorter.
- **Bonus benchmarks (KuaiRand-1k / 27k) are not attempted** in the current state.

## Contributions

Solo entry. All code in `kairos/` and `experiments/` written for this submission; the
files listed as the official starter kit are the organizers' and are unmodified.
