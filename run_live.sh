#!/usr/bin/env bash
# DEVELOPMENT script, kept for provenance. This is NOT the submitted configuration -
# it resumes from baseline_valid=0.6034 with max_iters=12 and no declared eps/N/floor.
# The submitted campaign is run_submission.sh; reproduce that one.
set -uo pipefail
cd "$(dirname "$0")"
: "${ANTHROPIC_API_KEY:?set ANTHROPIC_API_KEY before running}"
export PYTHONWARNINGS=ignore
exec ./.venv/bin/python -u -c "
import sys; sys.path.insert(0,'.')
from kairos.agent.loop import Kairos
from kairos.agent.proposer import TwoStageProposer

from kairos.agent.prior import PRIOR_PURE as PRIOR

p = TwoStageProposer(planner='claude-opus-5', coder='claude-sonnet-5')
# Resume from the best result any run has produced, not from the official baseline -
# otherwise a fresh run can 'improve' on 0.6016 while regressing against what we have.
# prewarm every torch-backed primitive in the trusted parent: candidates run in
# subprocesses that also import lightgbm, and the two OpenMP runtimes abort if a
# torch-backed cache is built alongside one.
k = Kairos(p, max_iters=12, seeds=(0,1,2), workdir='runs/kairos_live',
           max_tokens_total=300000, prior_summary=PRIOR, baseline_valid=0.6034,
           prewarm=('refit','din','expert','mf','cf','aux'))
s = k.run()
import json
s['tokens_by_model'] = p.by_model
print('SUMMARY'); print(json.dumps(s, indent=2, default=str))
"
