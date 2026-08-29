"""End-to-end agent test with the scripted proposer.

The point is not the score - it is the control flow.  Iteration 1 proposes the natural,
WRONG feature construction (streaming-causal history, which lets a validation row see its
own list-mates' labels).  The auditor must reject it before it is ever believed, hand the
specific violation back, and the agent must recover on its own with the frozen construction.
That catch-and-recover path is what Robustness and Autonomy are scored on.
"""
import sys; sys.path.insert(0,'.')
from kairos.agent.loop import Kairos
from kairos.agent.proposer import MockProposer

k = Kairos(MockProposer(), max_iters=3, seeds=(0,1), workdir='runs/kairos_smoke')
summary = k.run()
print("\n=== run summary ===")
for key in ('iterations','wall_clock_s','tokens_in','tokens_out','manual_interventions',
            'stall','converged','reason','best_valid'):
    print(f"  {key:22s} {summary[key]}")
print("\n=== error / recovery events ===")
import json, os
p = 'runs/kairos_smoke/ledger_errors.jsonl'
if os.path.exists(p):
    for line in open(p):
        r = json.loads(line)
        print(f"  iter {r['iteration']} [{r['kind']}]")
        print(f"    detail:   {r['detail'][:150]}")
        print(f"    recovery: {r['recovery']}")
else:
    print("  (none)")
