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

## Training data: the train split only (with one flag, and the argument for it)

Organizer FAQ 2.9.2 states:

> "training data is the train split only: date 20220408-20220421"

Every model in this project therefore fits on train-split rows only. That was **not**
originally true, and the bug is worth recording because it is subtle: three signals
(`baseline_score`, `cf_score`, `mf_factors`) used a frozen window's HORIZON as their fit
set. Horizon and training cut-off are different quantities. A window covering the test
period legitimately aggregates labels to 20220428 - FAQ 2.2 permits developing on "the
training split and the public validation feedback", so validation *feature statistics* are
fine - but a model scoring that window may still only FIT on rows to 20220421. The
unclamped horizon admitted **124,909 validation rows** into training.
`experiments/verify_train_split_only.py` pins the separation and proves the clamp is
load-bearing rather than decorative.

### The one genuinely ambiguous case, exposed as a flag

`ctx.refit_score()` can refit on train+validation to score TEST rows - worth a replicated
+0.0020 / +0.0022 on two independent backtest folds (exp22). Whether the rules permit it is
not obvious:

| supports permitting it | supports forbidding it |
|---|---|
| 2.3 out-of-scope: "no hidden-test access during development **(train + validation only)**" | 2.9.2: "training data is the train split only" |
| 2.4: "teams **develop on train + validation only**" | judging is by code review, and the call is visible in one function |
| 2.9.3 disqualifies "a pipeline that touches **test** labels" - this touches none | |
| 2.9.2's own rationale forbids `log_random` because it "covers both the validation and the test window ... injects in-period information about the scored rows and breaks the temporal split". Validation closes before the test window opens, so neither clause reaches it. | |

Rather than decide quietly, this is `STRICT_TRAIN_SPLIT` in `kairos/kernel/dataset.py`,
**defaulting to the strict reading**. The permissive setting is one environment variable
away, the two cache separately, and the run log records which produced any given
submission.

One property makes this safe to leave open: the flag changes **only test-row predictions**.
The clamp binds only where a window's horizon exceeds 20220421, which is true of exactly
one window - the test window. No validation number moves, so the agent's search, its
accept/reject decisions and its validation-best checkpoint are identical either way.

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
