#!/usr/bin/env bash
set -uo pipefail
cd /Users/twishamehta/tiktok/kuairand-starter-kit
source "/private/tmp/claude-501/-Users-twishamehta-tiktok-kuairand-starter-kit/a9d09e32-9cbd-4d9a-a593-8000cae399d0/scratchpad/anthropic_env.sh"
until ! pgrep -f exp14_control_arm >/dev/null 2>&1; do sleep 20; done
echo "##### control arm finished #####"
tail -14 runs/exp14.log | grep -v Warning
echo
echo "##### LIVE AGENT RUN - claude-opus-5 writing real code #####"
./.venv/bin/python -u -c "
import sys; sys.path.insert(0,'.')
from kairos.agent.loop import Kairos
from kairos.agent.proposer import AnthropicProposer
k = Kairos(AnthropicProposer(model='claude-opus-5'), max_iters=10, seeds=(0,1),
           workdir='runs/kairos_live', max_tokens_total=250000)
s = k.run()
import json; print(json.dumps(s, indent=2, default=str))
" 2>&1 | grep -v Warning
