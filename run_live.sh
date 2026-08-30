#!/usr/bin/env bash
set -uo pipefail
cd /Users/twishamehta/tiktok/kuairand-starter-kit
source "/private/tmp/claude-501/-Users-twishamehta-tiktok-kuairand-starter-kit/a9d09e32-9cbd-4d9a-a593-8000cae399d0/scratchpad/anthropic_env.sh"
export PYTHONWARNINGS=ignore
exec ./.venv/bin/python -u -c "
import sys; sys.path.insert(0,'.')
from kairos.agent.loop import Kairos
from kairos.agent.proposer import TwoStageProposer

PRIOR = (
    'Three prior live runs (9 iterations total) all tried variants of the SAME strategy: '
    'concatenate ctx.baseline_score, ctx.cf_score, ctx.auxiliary_signal, and an MF '
    'dot-product from ctx.mf_factors into ONE feature matrix (raw, then within-user '
    'rank-normalised, then rank-fusion columns) for a single downstream LightGBM. Every '
    'attempt scored BELOW the FM baseline (deltas -0.002 to -0.015 vs incumbent). This is '
    'the documented tree-on-a-calibrated-score pathology: a decision tree shatters a '
    'smooth, already-good continuous score into step-function splits, which degrades it - '
    'adding within-user normalisation to that same architecture does not fix the '
    'underlying problem. This strategy family should not be tried again without a '
    'genuinely new idea for WHY it would work differently this time. train_cfg mode='
    \"'scores'\"' (train your own model(s) inside build() and blend their FINAL OUTPUTS '
    'via within-user rank fusion, bypassing the single-downstream-tree architecture '
    'entirely) has never been attempted live despite being documented as the strategy '
    'that won by hand (+0.0030 primary). One earlier run also had an ACCEPTED candidate '
    'later found to be a leak (a hand-rolled streaming aggregate over user x author / '
    'user x tag / user x duration crosses, reading ctx.data.time_ms/y_raw directly '
    'without any per-fold horizon) - the harness now catches this via backtest '
    'confirmation, but avoid re-deriving unhorizoned streaming aggregates over raw '
    'ctx.data.y_raw; use ctx.frozen_prefix (which takes an explicit horizon) instead.'
)

p = TwoStageProposer(planner='claude-opus-5', coder='claude-sonnet-5')
# Resume from the best result any run has produced, not from the official baseline -
# otherwise a fresh run can 'improve' on 0.6016 while regressing against what we have.
k = Kairos(p, max_iters=12, seeds=(0,1,2), workdir='runs/kairos_live',
           max_tokens_total=300000, prior_summary=PRIOR, baseline_valid=0.6034)
s = k.run()
import json
s['tokens_by_model'] = p.by_model
print('SUMMARY'); print(json.dumps(s, indent=2, default=str))
"
