#!/usr/bin/env bash
# Reproduce the SUBMITTED KuaiRand-Pure result.
#
# Paths are relative to the repo root and the API key is read from the environment, so this
# runs on any machine. An earlier script hardcoded an absolute home directory and sourced a
# key file from a private scratchpad - which meant a judge could not run it at all.
#
#   export ANTHROPIC_API_KEY=sk-ant-...
#   ./run_submission.sh
#
# runs/kairos_submission_repro/ is the ARCHIVED campaign that produced the shipped
# submission.csv - it is Deliverable 3 evidence and this script must never overwrite it.
# A reproduction goes to a fresh directory, and we refuse to start if it already exists
# rather than silently merging two runs' candidates into one ledger.
#
#   WORKDIR=runs/my_run ./run_submission.sh    # override the destination
set -uo pipefail
cd "$(dirname "$0")"

: "${ANTHROPIC_API_KEY:?set ANTHROPIC_API_KEY before running}"
PY="${PY:-./.venv/bin/python}"
[ -x "$PY" ] || PY=python3
export PYTHONWARNINGS=ignore

ARCHIVE=runs/kairos_submission_repro
WORKDIR="${WORKDIR:-runs/kairos_submission_rerun}"
if [ "$WORKDIR" = "$ARCHIVE" ]; then
  echo "refusing to write to $ARCHIVE - that is the archived submitted campaign." >&2
  echo "set WORKDIR to something else." >&2
  exit 1
fi
if [ -e "$WORKDIR" ]; then
  echo "$WORKDIR already exists; move it aside or set WORKDIR to a fresh path." >&2
  exit 1
fi
echo "reproducing the submitted campaign into $WORKDIR"

# Convergence rule, declared BEFORE the run and recorded here, as FAQ 2.9.1 permits:
#   eps = 0.002 (organizer default)
#   N   = 5     (a miss is cheap at ~$0.12/iteration; 3 ends a run before a trajectory
#                is visible, and the trajectory is what Innovation and Impact are scored on)
#   floor = 10  (minimum scored iterations before convergence may trigger)
# Hard caps respected: 50 iterations, 6h wall-clock.
#
# Token cap 150k, costed rather than guessed: the archived 3-iteration campaign used 36,490
# tokens, so 150k covers 12-15 iterations with margin. An earlier 400k cap was picked
# arbitrarily and was nearly 2x the project's entire spend to that point.
exec "$PY" -u -c "
import sys; sys.path.insert(0,'.')
from kairos.agent.loop import Kairos
from kairos.agent.proposer import TwoStageProposer
from kairos.agent.prior import PRIOR_PURE

p = TwoStageProposer(planner='claude-opus-5', coder='claude-sonnet-5')
k = Kairos(p, max_iters=50, max_seconds=6*3600, seeds=(0,1,2),
           workdir='$WORKDIR',
           eps=0.002, stall_limit=5, min_iters=10,
           max_tokens_total=150000, prior_summary=PRIOR_PURE,
           baseline_valid=0.6016,
           prewarm=('refit','din','expert','mf','cf','aux'))
s = k.run()
import json
s['tokens_by_model'] = p.by_model
print('SUMMARY'); print(json.dumps(s, indent=2, default=str))
"
