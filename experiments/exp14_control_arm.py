"""Ablation of the agent's JUDGEMENT, with its proposals held fixed.

Two agents, identical except that the control's temporal-validity auditor is switched off
- which is what a conventional propose/train/follow-validation agent does. Both consume
the same proposal stream in the same order, so any difference is attributable to judgement.

The proposal order is a plausible exploration path: try behavioural history features, then
try a ranking objective (LambdaRank is the natural choice for a ranking metric), then the
frozen variants. Each agent's FINAL choice is then scored once on the hidden test through
the audited scorer - neither agent ever sees that number.
"""
import sys, json, subprocess; sys.path.insert(0,'.')
import numpy as np
from kairos.agent.loop import Kairos
from kairos.agent.proposer_pool import PoolProposer

ORDER = ['causal_all_bin', 'causal_all_lmr', 'frozen_all_bin', 'frozen_all_lmr']
POOL_TEST = json.load(open('runs/exp15_selection_v2.json'))   # hidden-test scores, for grading

out = {}
for tag, audit in (('control (no auditor)', False), ('KAIROS (auditor on)', True)):
    print(f"\n########## {tag} ##########")
    k = Kairos(PoolProposer(ORDER), max_iters=4, seeds=(0,1),
               workdir=f"runs/arm_{'audit' if audit else 'greedy'}",
               audit_enabled=audit, repair_attempts=1)
    s = k.run()
    chosen = None
    for e in reversed(k.ledger.entries):
        if e.decision == 'accept':
            chosen = e.hypothesis.statement.split(']')[0].lstrip('['); break
    test = POOL_TEST.get(chosen, {}).get('official', {}).get('test') if chosen else None
    out[tag] = {'chosen': chosen,
                'valid': k.incumbent['valid_primary'] if k.incumbent else None,
                'hidden_test': test,
                'iterations': s['iterations'], 'interventions': s['manual_interventions'],
                'tokens': s['tokens_in'] + s['tokens_out'], 'reason': s['reason']}
    print(f"  -> chose {chosen}  validation "
          f"{k.incumbent['valid_primary']:.4f}" if k.incumbent else "  -> chose nothing")

print("\n" + "="*74)
print(f"{'agent':<24} {'chose':<18} {'validation':>11} {'HIDDEN TEST':>12}")
print("-"*74)
for tag, r in out.items():
    v = f"{r['valid']:.4f}" if r['valid'] else "n/a"
    t = f"{r['hidden_test']:.4f}" if r['hidden_test'] else "n/a"
    print(f"{tag:<24} {str(r['chosen']):<18} {v:>11} {t:>12}")
print(f"{'baseline FM':<24} {'-':<18} {'0.6016':>11} {'0.5946':>12}")
ts = [r['hidden_test'] for r in out.values() if r['hidden_test']]
if len(ts) == 2:
    print(f"\nauditor is worth {ts[1]-ts[0]:+.4f} of hidden-test primary")
json.dump(out, open('runs/exp14_control_arm.json','w'), indent=2, default=str)
