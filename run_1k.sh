#!/usr/bin/env bash
# Transfer probe: the SAME agent, prompt-for-prompt and code-for-code, pointed at
# KuaiRand-1k - a dataset it was never tuned on, 8x the rows, 577x the item space, and
# 117x the per-user history. Nothing is retuned; only KAIROS_VARIANT changes.
set -uo pipefail
cd /Users/twishamehta/tiktok/kuairand-starter-kit
source "/private/tmp/claude-501/-Users-twishamehta-tiktok-kuairand-starter-kit/a9d09e32-9cbd-4d9a-a593-8000cae399d0/scratchpad/anthropic_env.sh"
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
k = Kairos(p, max_iters=8, seeds=(0,1,2), workdir='runs/kairos_1k',
           max_tokens_total=200000, prior_summary=PRIOR_1K,
           baseline_valid=float('$BASE'), prewarm=('refit',))
s = k.run()
import json
s['tokens_by_model'] = p.by_model
print('SUMMARY'); print(json.dumps(s, indent=2, default=str))
"
