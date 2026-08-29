#!/usr/bin/env bash
# Launch a KAIROS run against any provider.
#
#   ./run_agent.sh pool                                  # scripted proposals, no key
#   ./run_agent.sh 'anthropic:claude-sonnet-5'           # needs ANTHROPIC_WORKSPACE_ID too
#   ./run_agent.sh 'https://ark.cn-beijing.volces.com/api/v3|doubao-pro-32k'   # Ark/Doubao
#   ./run_agent.sh 'http://localhost:11434/v1|qwen2.5-coder'                   # local
#
# Credentials come from the environment, never from a file in this repo:
#   LLM_API_KEY (OpenAI-compatible)  |  ANTHROPIC_API_KEY + ANTHROPIC_WORKSPACE_ID
set -euo pipefail
SPEC="${1:-pool}"; ITERS="${2:-12}"
exec ./.venv/bin/python -u -c "
import sys; sys.path.insert(0,'.')
from kairos.agent.loop import Kairos
from kairos.agent.proposer import make_proposer
k = Kairos(make_proposer('$SPEC'), max_iters=$ITERS, seeds=(0,1,2),
           workdir='runs/kairos_live')
s = k.run()
import json; print(json.dumps(s, indent=2, default=str))
"
