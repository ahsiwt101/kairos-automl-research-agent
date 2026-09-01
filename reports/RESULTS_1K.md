# Results — KuaiRand-1k (bonus benchmark)

> **Status: run in flight.** This document describes a campaign that is still executing at
> the time of writing (`runs/kairos_1k/`, started 2026-09-01). Iterations completed so far
> are reported below with their real numbers; the final row is deliberately empty rather
> than estimated. Nothing here is scored on the 1k hidden test yet.

KuaiRand-1k is a **bonus** benchmark. Per §2.6, skipping it does not reduce the
KuaiRand-Pure score, and KuaiRand-Pure alone determines 100% of the Primary metric.

## What this run is testing

The same agent, prompt-for-prompt and code-for-code, pointed at a dataset it was never
tuned on. Only `KAIROS_VARIANT` changes. 1k is 8× the rows of Pure, 577× the item space,
and 117× the per-user history — so this is a transfer probe on the *agent*, not a fresh
tuning exercise.

Per FAQ 2.9.2, 1k is trained on its own splits only. It is never used as auxiliary training
data for Pure, and Pure is never used for it. This is enforced structurally rather than by
convention: every signal cache is keyed by dataset variant, and a cached array of the wrong
length is refused outright (`experiments/verify_variant_isolation.py`).

## Official FM baseline on 1k, reproduced

Reproduced with the same pipeline, on 1k's own splits:

| split | GAUC | nDCG@5 | primary | users |
|---|---|---|---|---|
| valid | 0.6406 | 0.5151 | 0.5778 | 978 |
| test | 0.6428 | 0.5285 | 0.5856 | 997 |

The agent's `baseline_valid` is set to 1k's own 0.5778, not Pure's — carrying Pure's number
across would make every 1k candidate look like a regression against a figure from a
different dataset.

## Run configuration

Declared before the run, in `run_1k.sh`:

- Convergence rule (FAQ 2.9.1): ε = 0.002, N = 5, minimum-iteration floor = 10
- Hard caps: 50 iterations, 6 h wall-clock
- Token budget: 120,000
- **Seeds: 1, not 3.** Every candidate is re-run in full for backtest confirmation, so seed
  count multiplies the *cost of verification* on 11.7M rows. Seed averaging was measured to
  saturate by 3 seeds on Pure and is worth ~0.001; being able to *check* a large claim is
  worth more. This is a deliberate trade and it is why the 1k numbers below carry
  `±0.0000` — that is a single seed, not a stability claim.

## Iterations so far

| # | decision | valid primary | GAUC | nDCG@5 | Δ vs 1k baseline | wall-clock |
|---|---|---|---|---|---|---|
| 1 | rollback | — (no score) | — | — | — | 126.7 s |
| 2 | **accept** | **0.6522** | 0.6810 | 0.6235 | **+0.0744** | 727.8 s |
| … | *run in progress* | | | | | |

Iteration 1 rolled back rather than crashed: the candidate was produced and rejected before
its score was believed, which counts toward the 50-iteration cap but does not advance or
reset the convergence window (FAQ 2.9.1).

## What is not yet claimed

- **No hidden-test score.** The 1k test split has not been consulted for this run.
- **The +0.0744 is a single-seed validation figure**, backtest-confirmed but not
  seed-averaged. It is far larger than any gain seen on Pure, which on this project's own
  argument is a reason for suspicion rather than celebration until it is confirmed on the
  test split.
- **No submission file for 1k has been generated.**

## Why an earlier 1k attempt failed, and what fixed it

A previous 1k campaign rejected every large-gain candidate. The rejections were misread at
the time as the verifier being intrinsically too expensive on 11.7M rows. The actual cause
was narrower: prewarming did not cover the *confirmation* fold, so every backtest
confirmation rebuilt two windowed FMs over 11.7M rows inside the candidate sandbox and blew
its timeout. Prewarm now covers the confirmation fold, and confirmations complete.

The port itself found five latent defects in our own code — documented in
`reports/FINDINGS.md`. That is the more durable result from the 1k work so far: running the
same agent on a second dataset surfaced bugs that a single-dataset run never would have.

## Updating this document

When the run stops, regenerate the per-iteration log and this table from the ledger:

```bash
./.venv/bin/python experiments/export_run_log.py runs/kairos_1k reports/ITERATION_LOG_1K.md
```

The stop reason is read from `runs/live_1k.log`; report it verbatim, as the Pure campaign
does, rather than assuming the convergence rule fired.
