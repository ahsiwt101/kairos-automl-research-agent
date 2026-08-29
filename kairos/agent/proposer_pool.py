"""A proposer that walks the exp11 candidate pool as real generated code.

Used for the control-arm ablation: both agents see the SAME proposal stream, so any
difference in outcome is attributable to what the agent does with the proposals, not to
one of them getting luckier suggestions.
"""
import textwrap
from kairos.agent.proposer import Proposal
from kairos.kernel.candidates import POOL

TEMPLATE = '''
import numpy as np
def build(ctx):
    from kairos.kernel.candidates import build_candidate_matrix
    X, names, hz = build_candidate_matrix(ctx.data, ctx.fold.spec, {mode!r}, {fams!r})
    return X, names
'''

HYP = {
    'causal': dict(
        statement="Summarise each user's and item's history as a long_view rate using a "
                  "time-ordered prefix so no row sees its own label",
        mechanism="The ID model has no behavioural summary; an explicit historical rate "
                  "should add personalisation it cannot represent",
        predicted_effect="GAUC rises across user-activity deciles",
        predicted_gain=0.02, family='history'),
    'frozen': dict(
        statement="Same aggregates, but frozen at the start of each evaluation window",
        mechanism="Evaluation ranks a user's list as a set, so list-mate labels are not "
                  "available at scoring time and must not enter the features",
        predicted_effect="validation falls toward test; the val-test gap collapses",
        predicted_gain=0.003, family='debias'),
}


class PoolProposer:
    def __init__(self, order=None):
        self.pool = [p for p in POOL if p[0] in order] if order else list(POOL)
        if order:
            self.pool.sort(key=lambda p: order.index(p[0]))
        self.i = -1
        self.tokens_in = self.tokens_out = 0

    def propose(self, digest, ledger_summary, budget, last_failure=None):
        self.i += 1
        if self.i >= len(self.pool):
            self.i = len(self.pool) - 1
        name, mode, fams = self.pool[self.i]
        self.tokens_in += 1400; self.tokens_out += 500
        h = dict(HYP[mode]); h['statement'] = f"[{name}] " + h['statement']
        return Proposal(h, TEMPLATE.format(mode=mode, fams=tuple(fams)).strip(), '<pool>')
