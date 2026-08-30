#!/usr/bin/env bash
set -uo pipefail
cd /Users/twishamehta/tiktok/kuairand-starter-kit
source "/private/tmp/claude-501/-Users-twishamehta-tiktok-kuairand-starter-kit/a9d09e32-9cbd-4d9a-a593-8000cae399d0/scratchpad/anthropic_env.sh"
export PYTHONWARNINGS=ignore
exec ./.venv/bin/python -u -c "
import sys; sys.path.insert(0,'.')
from kairos.agent.loop import Kairos
from kairos.agent.proposer import TwoStageProposer
p = TwoStageProposer(planner='claude-opus-5', coder='claude-sonnet-5')
k = Kairos(p, max_iters=10, seeds=(0,1), workdir='runs/kairos_live',
           max_tokens_total=250000)
s = k.run()
import json
s['tokens_by_model'] = p.by_model
print('SUMMARY'); print(json.dumps(s, indent=2, default=str))
"
