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

# --------------------------------------------------------------------------------------
# FAQ 2.9.1: a crashed iteration (no validation score) must NOT advance or reset the
# convergence window. It still counts toward the 50-iteration and 6h caps.
import math
led = Ledger(path='runs/_stall_crash_probe.jsonl', baseline=0.6016)
def _miss(i):
    return Entry(iteration=i, hypothesis=Hypothesis(family='f', statement='s', mechanism='m',
                                       predicted_effect='e', predicted_gain=0.0),
                 action_kind='patch', code_diff='',
                 outcome=Outcome(valid_primary=0.6000), decision='reject', reason='')
def _crash(i):
    return Entry(iteration=i, hypothesis=Hypothesis(family='f', statement='s', mechanism='m',
                                       predicted_effect='e', predicted_gain=0.0),
                 action_kind='patch', code_diff='',
                 outcome=Outcome(valid_primary=float('nan')), decision='crash', reason='')
led.add(_miss(1)); led.add(_crash(2)); led.add(_miss(3))
assert led.stall_counter(0.002) == 2, \
    f'crashed iteration must not advance the window, got {led.stall_counter(0.002)}'
led.add(_crash(4))
assert led.stall_counter(0.002) == 2, 'a trailing crash must not advance the window either'
led.add(_miss(5))
assert led.stall_counter(0.002) == 3, 'real misses must still advance the window'
import os as _os; _os.remove('runs/_stall_crash_probe.jsonl')
print("  [PASS] crashed iterations neither advance nor reset the convergence window")
