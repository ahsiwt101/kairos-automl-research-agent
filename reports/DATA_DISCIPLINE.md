# What we train on, and what we never touch

A short answer to a question worth asking of any leaderboard result: **which rows did the
model actually see?** Everything below is verifiable from the code paths named, not just
asserted.

## The three splits (KuaiRand-Pure, the submitted result)

| split | rows | dates | what it is used for |
|---|---|---|---|
| train | 1,141,112 | 20220409-20220421 | the only rows any model fits on |
| valid | 124,909 | 20220422-20220428 | early stopping, blend weights, candidate selection |
| test | 170,588 | 20220429-20220508 | **scored only, never trained on, never selected on** |

## Models fit on training rows only

The downstream model is constructed as `lgb.Dataset(Xtr, label=ytr)` where `Xtr = X[fold.idx['train']]`
(`kairos/agent/evaluate_candidate.py`). Validation rows enter only through the early-stopping
callback, which picks the boosting iteration that maximises the validation metric. Test rows
are scored by the fitted model and never contribute a gradient.

## The one place validation labels legitimately train a model

`ctx.refit_score()` (`kairos/kernel/refit_signal.py`) is deliberately asymmetric:

    train / valid rows -> prediction from a model trained on TRAIN ONLY
    test rows          -> prediction from a model trained on TRAIN + VALIDATION

This is not a leak, and the asymmetry is the point. A validation row must be scored by a
model that has not seen validation labels, or every weight fitted against it is optimistic.
A test row, at submission time, may legitimately be scored by a model that used everything
available before the test window opened - that is simply using the data one actually has.
Collapsing the two cases in either direction is what causes trouble, so the distinction is
enforced inside the primitive rather than left to each candidate to remember.

## Feature construction is horizon-bounded, not merely split-bounded

Splitting by date is not enough: a feature computed over the whole log leaks future
information into training rows even when the *rows* are correctly partitioned. Every rate,
count and embedding in this project comes from `frozen_prefix(keys, date, y, mask, horizon)`,
which aggregates labels only from rows dated at or before an explicit per-window horizon.
`LabelVault` enforces this at the data layer: labels past the horizon return the sentinel
`-1` rather than a real value, so a feature builder that ignores the horizon produces an
obviously poisoned feature instead of a silent leak.

## The sealed test split, and how often we looked

KuaiRand-Pure is a public dataset, so the labels for the competition's hidden-test window
exist on disk. We treat them as sealed anyway, and route every consultation through an
audited `Scorer` that appends to `runs/scorer_audit.log`.

**Audited consultations of the hidden-test split to date: 60.** Every one is timestamped
with the reason it was made. They are used to *report* results, never to choose between
candidates - selection runs on validation plus three temporal backtest folds that live
entirely inside the public-label region (`FOLDS` in `kairos/kernel/dataset.py`).

That distinction is the whole reason our val->test gap is small. An agent that selects on
the test split will beat us on validation and lose on test; we measured exactly that in the
greedy control arm (valid 0.7339, test 0.5790).

## On KuaiRand-1k

1k's labels are fully public for the whole period, so `FOLDS['official']['sealed']` is False
there and the test window is scored directly. This is a deliberate difference: the 1k run is
a transfer probe, not a submission, and being able to measure the true val->test gap with no
submission budget is precisely what makes it useful as a check on the Pure result.
