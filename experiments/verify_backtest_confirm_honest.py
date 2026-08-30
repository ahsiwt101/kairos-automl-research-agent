"""False-positive check: a genuinely honest candidate (frozen_prefix, properly horizoned)
must NOT be rejected by the backtest confirmation gate.
"""
import sys, textwrap; sys.path.insert(0,'.')
from kairos.agent.loop import Kairos
from kairos.agent.proposer import Proposal

src = textwrap.dedent('''
import numpy as np
def build(ctx):
    d = ctx.data
    y = d.y_raw.astype(np.float64)
    hz = ctx.window_horizons(d.date.astype(np.int64), ctx.windows)
    labeled = np.ones(d.n, dtype=bool)
    cols, names = [], []
    for nm, keys in (('item', d.video_id.astype(np.int64)),
                     ('user', d.user_id.astype(np.int64))):
        l_, p_ = ctx.frozen_prefix(keys, d.date.astype(np.int64), y, labeled, hz)
        cols.append(ctx.smoothed_rate(p_, l_, 0.33, 20.0)); names.append(nm+'_rate')
        cols.append(np.log1p(l_)); names.append(nm+'_logn')
    cols.append(ctx.baseline_score); names.append('baseline_score')
    X = np.stack(cols, 1).astype(np.float32)
    ctx.check(X, names)
    return X, names
''').strip()

hyp = {'statement': 'honest frozen-prefix candidate (false-positive check)',
      'mechanism': 'x', 'predicted_effect': 'x', 'predicted_gain': 0.0, 'family': 'history'}
prop = Proposal(hyp, src, '<honest-replay>')

k = Kairos.__new__(Kairos)
Kairos.__init__(k, proposer=None, workdir='runs/_replay_honest', max_iters=1)
ok, detail = k._backtest_confirm(prop)
print(f"backtest confirmation: {detail}")
print(f"would accept: {ok}")
assert ok, "an honest candidate must NOT be rejected by the confirmation gate"
print("\nPASS: honest candidate correctly passes")
