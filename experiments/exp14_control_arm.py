"""Ablation of the agent's JUDGEMENT, holding its proposals fixed.

Two agents, identical in every respect except one: the control has its temporal-validity
auditor switched off, which is what a conventional propose-train-follow-validation agent
does.  Both consume the same proposal stream in the same order.
"""
import sys, json; sys.path.insert(0,'.')
import numpy as np
from kairos.agent.loop import Kairos
from kairos.agent.proposer_pool import PoolProposer

ORDER = ['causal_all', 'causal_user', 'frozen_all', 'frozen_user', 'causal_ui', 'frozen_ui']
out = {}
for tag, audit in (('control (no auditor)', False), ('KAIROS (auditor on)', True)):
    k = Kairos(PoolProposer(ORDER), max_iters=6, seeds=(0,1),
               workdir=f"runs/arm_{'audit' if audit else 'greedy'}",
               audit_enabled=audit, repair_attempts=1)
    print(f"\n########## {tag} ##########")
    s = k.run()
    inc = k.incumbent
    out[tag] = {'summary': s, 'chosen_valid': inc['valid_primary'] if inc else None,
                'chosen_X': inc['X_path'] if inc else None}
    print(f"  chose: validation {inc['valid_primary']:.4f}" if inc else "  chose: nothing")
json.dump(out, open('runs/exp14_control_arm.json','w'), indent=2, default=str)
print("\nNOTE: the hidden-test score of each agent's choice is graded separately in "
      "exp12 using the audited scorer - neither agent ever sees it.")
