"""Hard prior knowledge carried into a fresh agent run.

Each Kairos run starts with an empty ledger, so without this the agent re-derives the same
dead ends every time - we watched it independently reinvent the same losing architecture
across three separate runs. This is the lab notebook a returning researcher would read.

Kept in its own module rather than inline in run_live.sh: it was previously embedded in a
shell heredoc wrapping a python -c string, where two successive edits silently failed to
apply and the agent ran without capabilities it had been told about.
"""

PRIOR_PURE = (
    "PRIOR RUNS ON THIS BENCHMARK. Early runs repeatedly tried the SAME losing "
    "architecture: concatenate ctx.baseline_score, ctx.cf_score, ctx.auxiliary_signal and "
    "an MF dot-product into ONE feature matrix for a single downstream LightGBM, in raw "
    "then within-user-normalised then rank-fused form. Every attempt scored BELOW the FM "
    "baseline. That is the tree-on-a-calibrated-score pathology: a tree shatters a smooth, "
    "already-good continuous score into step functions. Adding within-user normalisation "
    "to that same architecture does not fix it. "

    "WHAT WON: train_cfg mode='scores' - train your own model(s) inside build() and blend "
    "their FINAL OUTPUTS by within-user rank fusion, bypassing the single-tree "
    "architecture. An accepted candidate blending refit/din/baseline/cf/mf reached "
    "validation 0.6034 (hidden test 0.5988). That is the incumbent to beat. "

    "AVAILABLE, MEASURED: ctx.refit_score() is the FM fit with the best data per split "
    "(~+0.002 over ctx.baseline_score, confirmed on two backtest folds) - prefer it. "
    "ctx.din_score() is a sequence model at 0.6023 standalone, ABOVE the FM baseline, from "
    "an unrelated architecture. ctx.expert_score(sub) for sub in {context,item,user} gives "
    "three models each trained on ONE disjoint feature family: individually WEAK "
    "(0.5718/0.5906/0.5357, all below the FM) but mutually decorrelated at mean Spearman "
    "+0.362 versus +0.848 between FM and DIN. Fusion is rewarded by decorrelation, not "
    "member strength. A hand-built linear blend including the experts reached validation "
    "0.6031 / test 0.5985 - close to but NOT beating the incumbent - so a better "
    "combination of these members plausibly exists. "

    "RULED OUT BY MEASUREMENT - do not re-propose: LambdaRank/soft-nDCG loss (test 0.5874, "
    "worst of six losses); watch-time regression targets L2/Huber/D2Q "
    "(0.5605/0.5754/0.5846, far below binary); recency weighting of training rows (noise); "
    "refitting the GBDT on train+valid (+0.0000, though refitting the FM DOES help and is "
    "already inside ctx.refit_score); four tab encodings (all within noise); eight "
    "item-quality estimators combining time-decay and hierarchical shrinkage (best "
    "+0.0002); seed-averaging beyond 3 seeds (saturates - 1/3/5/10 seeds give "
    "0.6013/0.6026/0.6027/0.6027); power/gamma exponents in rank fusion (raised validation "
    "0.6031->0.6035 but LOWERED test 0.5985->0.5982); DIN history reweighting (all modes "
    "within noise, and a PERFECT train/serve distribution match was the worst option). "

    "PROVABLE NO-OP - monotone post-processing of the final score (temperature scaling, "
    "per-user percentile transforms) cannot change GAUC or nDCG, which depend only on "
    "within-user ORDER; a monotone map leaves order unchanged. Within-user ties are worth "
    "~5e-05 in total. "

    "UNTRIED AND PLAUSIBLE: checkpoint/SWA averaging across training epochs - a different "
    "variance-reduction axis from seed averaging, which is already saturated. "

    "LEAKAGE: one earlier accepted candidate was later found to be a leak - a hand-rolled "
    "streaming aggregate over user x author / user x tag / user x duration crosses, "
    "reading ctx.data.time_ms and ctx.data.y_raw directly with no per-fold horizon. The "
    "harness now catches this via backtest confirmation, but do not re-derive unhorizoned "
    "streaming aggregates over raw ctx.data.y_raw; use ctx.frozen_prefix, which takes an "
    "explicit horizon."
)


# ---------------------------------------------------------------------------- transfer
# The 1k prior is deliberately NOT PRIOR_PURE with the numbers swapped.
#
# PRIOR_PURE carries empirical conclusions measured in Pure's regime - an incumbent score,
# a ruled-out list, per-primitive standalone strengths. 1k is a materially different
# regime (5,143 rows/user vs 44; 4.37M items vs 7,583; no train->serve density collapse),
# so those conclusions are exactly the things whose transfer is in question. Feeding them
# in would answer "do Pure's findings hold?" while looking like "does the agent
# generalise?", and a skeptic would rightly say the answer was handed over.
#
# What survives is only what is true independent of the dataset: one architectural
# pathology with a stated mechanism, one theorem about the metric, the leakage rule, and
# an honest statement of which primitives are actually wired on this variant.
PRIOR_1K = (
    "NEW BENCHMARK. You have not seen this dataset before and no prior run has been made "
    "on it. There is no incumbent beyond the official FM baseline. Nothing below is a "
    "measurement on THIS data - treat every claim as a hypothesis to test, not a result. "

    "ARCHITECTURAL PATHOLOGY (mechanism, not measurement). Feeding an already-calibrated "
    "continuous score (ctx.baseline_score) into a downstream tree as one feature among "
    "many tends to LOSE information: a tree shatters a smooth monotone score into step "
    "functions. This is a property of trees, not of any one dataset, so expect it here "
    "too - but it is still a hypothesis you may test. "

    "CAPABILITY: train_cfg mode='scores' lets you train your own model(s) inside build() "
    "and blend their FINAL OUTPUTS by within-user rank fusion, instead of concatenating "
    "everything into one feature matrix for a single tree. Fusion is rewarded by "
    "DECORRELATION between members, not by member strength - a weak but uncorrelated "
    "member can pay more than a strong correlated one. "

    "PROVABLE NO-OP (theorem, holds on any dataset). GAUC and nDCG depend only on "
    "within-user ORDER. Any monotone map applied to the final score - temperature "
    "scaling, per-user percentile transforms, sigmoid rescaling - leaves that order "
    "unchanged and therefore cannot move either metric. Do not spend an iteration on one. "
    "For the same reason, any feature that is CONSTANT within a user contributes nothing "
    "to the ranking, however predictive it looks globally. "

    "LEAKAGE. Never build a streaming aggregate over raw ctx.data.y_raw or ctx.data.time_ms "
    "without a per-fold horizon: a validation row must not see labels from inside its own "
    "evaluation window. Use ctx.frozen_prefix, which takes an explicit horizon. The "
    "harness independently backtest-confirms every accepted candidate, so a leak costs you "
    "the iteration rather than reaching the submission. "

    "PRIMITIVES WIRED ON THIS VARIANT: ctx.baseline_score (the official FM's own "
    "out-of-sample score) and ctx.refit_score() (the same FM given the best data per "
    "split). ctx.din_score / ctx.mf_factors / ctx.cf_score / ctx.expert_score / "
    "ctx.auxiliary_signal are NOT available on this run and will raise - do not plan "
    "around them. You still have ctx.col, ctx.video_attr, ctx.user_attr, ctx.frozen_prefix "
    "and ctx.check, which is enough to build strong frozen-window features."
)
