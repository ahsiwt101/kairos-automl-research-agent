#!/usr/bin/env bash
# Transfer probe: the SAME agent, prompt-for-prompt and code-for-code, pointed at
# KuaiRand-1k - a dataset it was never tuned on, 8x the rows, 577x the item space, and
# 117x the per-user history. Nothing is retuned; only KAIROS_VARIANT changes.
set -uo pipefail
cd "$(dirname "$0")"
: "${ANTHROPIC_API_KEY:?set ANTHROPIC_API_KEY before running}"
export PYTHONWARNINGS=ignore
export KAIROS_VARIANT=1k
BASE="${1:?usage: run_1k.sh <1k_fm_baseline_valid_primary>}"
exec ./.venv/bin/python -u -c "
import sys; sys.path.insert(0,'.')
from kairos.agent.loop import Kairos
from kairos.agent.proposer import TwoStageProposer
from kairos.agent.prior import PRIOR_1K

p = TwoStageProposer(planner='claude-opus-5', coder='claude-sonnet-5')
# baseline_valid is 1k's OWN FM baseline - carrying Pure's 0.6034 would make every
# candidate look like a regression against a number from a different dataset.
# One seed, not three: every candidate is re-run in full for backtest confirmation, so
# seed count multiplies the COST OF VERIFICATION on 11.7M rows. Both iterations of the
# first attempt were rejected not because they failed the check but because the check
# could not finish inside its budget - the verifier, not the hypothesis, was the
# bottleneck. Seed averaging was measured to saturate by 3 seeds on Pure and is worth
# ~0.001; being able to CHECK a +0.07 claim is worth more.
# Same declared rule as the Pure campaign (FAQ 2.9.1): eps=0.002, N=5, floor=10.
# prewarm now also covers the CONFIRMATION fold. Its omission is what killed earlier
# attempts: every confirmation rebuilt two windowed FMs over 11.7M rows inside the
# candidate sandbox and blew its timeout, which was misread as the verifier being
# intrinsically too expensive on this variant.
k = Kairos(p, max_iters=50, max_seconds=6*3600, seeds=(0,), workdir='runs/kairos_1k',
           eps=0.002, stall_limit=5, min_iters=10,
           max_tokens_total=120000, prior_summary=PRIOR_1K,
           baseline_valid=float('$BASE'), prewarm=('refit',))
s = k.run()
import json
s['tokens_by_model'] = p.by_model
print('SUMMARY'); print(json.dumps(s, indent=2, default=str))
"
