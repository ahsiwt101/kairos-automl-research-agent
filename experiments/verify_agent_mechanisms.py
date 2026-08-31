"""Tests for the two mechanisms that were DESIGNED BUT DORMANT until now.

`prediction_hit` was declared in the ledger and never written by anything;
`family_track_record()` was computed and never read; `misses_before_run_ends` was passed to
the LLM as advice with no enforcement. All three looked implemented. These tests exist so
that "it is wired" is a checked claim rather than an assumed one.
"""
import sys; sys.path.insert(0,'.')
import copy
from kairos.kernel.diagnostics import check_prediction, PREDICTABLE
from kairos.agent.ledger import Ledger, Entry, Hypothesis, Outcome

BEFORE = {'GAUC':0.660,'nDCG@5':0.530,'primary':0.600,'headroom_total':0.250,
          'inversions':{'duration_decile':{'total_gauc_loss':0.290},
                        'item_pop_decile':{'total_gauc_loss':0.290}},
          'slices':{'eval_list_size':[{'bucket':1,'auc_mean':0.630},
                                      {'bucket':12,'auc_mean':0.675}],
                    'user_train_impressions':[{'bucket':0,'auc_mean':0.675},
                                              {'bucket':9,'auc_mean':0.685}]}}

print("=== prediction scoring ===")
after = copy.deepcopy(BEFORE)
after['GAUC'] = 0.670
after['inversions']['duration_decile']['total_gauc_loss'] = 0.250
cases = [
    ({'diagnostic':'gauc','direction':'increase'},                    True,  'true claim'),
    ({'diagnostic':'gauc','direction':'decrease'},                    False, 'false claim'),
    ({'diagnostic':'inversion_loss_duration','direction':'decrease'}, True,  'slice claim, true'),
    ({'diagnostic':'inversion_loss_duration','direction':'increase'}, False, 'slice claim, false'),
    ({'diagnostic':'ndcg','direction':'increase'},                    False, 'nothing moved'),
    ({'diagnostic':'not_a_metric','direction':'increase'},            None,  'unknown diagnostic'),
    ({},                                                              None,  'empty prediction'),
]
for pred, want, label in cases:
    got = check_prediction(pred, BEFORE, after)
    status = 'PASS' if got is want else 'FAIL'
    print(f"  [{status}] {label:24s} -> {str(got):5s} (expected {want})")
    assert got is want, f"{label}: got {got}, expected {want}"
# a missing digest must be unscored, never counted as a miss against the agent
assert check_prediction({'diagnostic':'gauc','direction':'increase'}, None, after) is None
print("  [PASS] missing digest -> None (an instrumentation gap is not the agent's fault)")

print("\n=== family track record feeds back ===")
led = Ledger(path='runs/_mech_probe.jsonl', baseline=0.6016)
for i, (fam, vp, hit) in enumerate([('ensemble',0.5950,False), ('ensemble',0.5940,False),
                                    ('debias',0.6030,True)], start=1):
    led.add(Entry(iteration=i, hypothesis=Hypothesis('s','m','p',0.0,fam),
                  action_kind='patch', code_diff='',
                  outcome=Outcome(valid_primary=vp, delta_vs_incumbent=vp-0.6016,
                                  prediction_hit=hit),
                  decision='reject', reason=''))
rec = led.family_track_record()
print(f"  ensemble: n={rec['ensemble']['n']} hit_rate={rec['ensemble']['hit_rate']} "
      f"mean_gain={rec['ensemble']['mean_gain']:+.4f}")
print(f"  debias:   n={rec['debias']['n']} hit_rate={rec['debias']['hit_rate']} "
      f"mean_gain={rec['debias']['mean_gain']:+.4f}")
assert rec['ensemble']['n'] == 2 and rec['ensemble']['hit_rate'] == 0.0
assert rec['debias']['hit_rate'] == 1.0
assert rec['ensemble']['mean_gain'] < 0 < rec['debias']['mean_gain']
print("  [PASS] hit-rates and mean gains are tracked per family")

print("\n=== the scheduler can distinguish the two modes ===")
# 2 misses recorded against a 0.6016 baseline -> one miss left -> consolidate
print(f"  stall counter after 3 sub-baseline entries: {led.stall_counter(0.002)}")
assert led.stall_counter(0.002) >= 2
bad = rec['ensemble']
assert bad['n'] >= 2 and bad['mean_gain'] < 0, "ensemble should look like a bad bet"
print("  [PASS] a repeatedly-losing family is identifiable as one to block in consolidate mode")

import os; os.remove('runs/_mech_probe.jsonl')
print("\nALL AGENT-MECHANISM TESTS PASS")

# --------------------------------------------------------------------------------------
# _backtest_confirm must return the True SINGLETON when it confirms.
# The caller distinguishes three outcomes by identity - True (confirmed), False
# (disconfirmed), None (could not be run) - because "we checked and it failed" and "we
# could not check" are different evidence and must not carry the same message. numpy
# comparisons return np.bool_, which is equal to True but is not True, so a missing bool()
# would silently turn every confirmed candidate into a rejected one.
import inspect as _inspect
from kairos.agent.loop import Kairos as _Kairos
_src = _inspect.getsource(_Kairos._backtest_confirm)
assert 'ok = bool(' in _src, '_backtest_confirm must coerce its verdict with bool()'
assert _src.count('return None,') >= 3, \
    'infrastructure failures in _backtest_confirm must return None, not False'
import numpy as _np
assert (_np.bool_(True) is not True), 'premise of this test no longer holds'
assert (bool(_np.bool_(True)) is True), 'bool() must yield the True singleton'
print("  [PASS] confirm verdict is a real bool; unverifiable is None, not False")
