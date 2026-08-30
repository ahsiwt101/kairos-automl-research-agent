"""Replay the EXACT leaked candidate from the live run this fix responds to.

That candidate hand-rolled its own time-decayed prefix over user x author / user x tag /
user x duration crosses, reading ctx.data.time_ms and ctx.data.y_raw directly - never
calling ctx.causal_prefix, so the auditor's structural checks (which only inspect columns
named 'user_rate'/'user_logn', and only pure user-level aggregates at that) never looked at
it. It scored +0.0936 on validation and was ACCEPTED. This is the check that should have
stopped it: does the SAME code also show a small, honest valid-test gap on an unsealed
backtest fold?
"""
import sys, json; sys.path.insert(0,'.')
from kairos.agent.loop import Kairos
from kairos.agent.proposer import Proposal

src = open('runs/kairos_live/cand2/candidate.py').read()
hyp = {'statement': 'replay of the accepted leaky candidate from the live run',
      'mechanism': 'x', 'predicted_effect': 'x', 'predicted_gain': 0.0, 'family': 'history'}
prop = Proposal(hyp, src, '<replay>')

k = Kairos.__new__(Kairos)   # avoid re-running the full ~100s prewarm for this check
import kairos.agent.loop as loop_mod
Kairos.__init__(k, proposer=None, workdir='runs/_replay_confirm', max_iters=1)

ok, detail = k._backtest_confirm(prop)
print(f"backtest confirmation: {detail}")
print(f"would accept: {ok}")
assert not ok, "the leaked candidate must FAIL backtest confirmation"
print("\nPASS: the fix catches the exact candidate that slipped through live")
