"""Pins the stall-counter bug found while the Phase-3 live run was in flight.

Seeding best_so_far at -inf let a run earn "progress" credit for beating its own bad early
guesses, so it kept going well past the point the stated rule (no +eps improvement in N
consecutive iterations, measured against the incumbent) would actually end it. This
reproduces the exact numbers from the live run that exposed it: three iterations below the
FM baseline (0.6016) should converge the run at N=3, not keep it going.
"""
import sys; sys.path.insert(0,'.')
from kairos.agent.ledger import Ledger, Entry, Hypothesis, Outcome

BASELINE = 0.6016
led = Ledger(path='runs/_stall_probe.jsonl', baseline=BASELINE)
for i, v in enumerate([0.5950, 0.5957, 0.5994], start=1):
    led.add(Entry(iteration=i,
                  hypothesis=Hypothesis('x', 'x', 'x', 0.0, 'test'),
                  action_kind='patch', code_diff='',
                  outcome=Outcome(valid_primary=v, delta_vs_incumbent=v - BASELINE),
                  decision='reject', reason=''))

s = led.stall_counter(eps=0.002)
print(f"stall counter after 3 sub-baseline iterations: {s} (expected 3)")
assert s == 3, f"expected stall=3, got {s} - the bug is back"
conv, why = led.converged(eps=0.002, n=3, max_iters=50)
print(f"converged: {conv} ({why})")
assert conv and 'stalled' in why
print("PASS: run correctly ends at the stated N=3 rule, not later")

import os
os.remove('runs/_stall_probe.jsonl')
