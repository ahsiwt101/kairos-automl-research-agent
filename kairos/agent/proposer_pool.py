"""A proposer that walks the exp11 candidate pool as real generated code.

Used for the control-arm ablation: both agents see the SAME proposal stream, so any
difference in outcome is attributable to what the agent does with the proposals, not to
one of them getting luckier suggestions.
"""
import textwrap
from kairos.agent.proposer import Proposal
from kairos.kernel.candidates import POOL, POOL_V2

TEMPLATE = '''
import numpy as np
def build(ctx):
    from kairos.kernel.candidates import build_candidate_matrix
    X, names, hz = build_candidate_matrix(ctx.data, ctx.fold.spec, {mode!r}, {fams!r})
    return X, names, {cfg!r}
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
    def __init__(self, order=None, pool=None):
        src = pool if pool is not None else POOL_V2
        src = [(p[0], p[1], p[2], p[3] if len(p) > 3 else 'binary',
                p[4] if len(p) > 4 else None) for p in src]
        self.pool = [p for p in src if p[0] in order] if order else list(src)
        if order:
            missing = [n for n in order if n not in {p[0] for p in src}]
            if missing:
                raise ValueError(f"unknown candidate names in order: {missing}; "
                                 f"available: {sorted(p[0] for p in src)}")
            self.pool.sort(key=lambda p: order.index(p[0]))
        if not self.pool:
            raise ValueError("PoolProposer built an empty pool")
        self.i = -1
        self.tokens_in = self.tokens_out = 0

    def propose(self, digest, ledger_summary, budget, last_failure=None):
        self.i += 1
        if self.i >= len(self.pool):
            self.i = len(self.pool) - 1
        name, mode, fams, obj, grp = self.pool[self.i]
        self.tokens_in += 1400; self.tokens_out += 500
        h = dict(HYP[mode]); h['statement'] = f"[{name}] " + h['statement']
        if obj == 'lambdarank':
            h['statement'] += ", optimised with LambdaRank over per-day lists"
            h['mechanism'] += "; the metric is a ranking metric, so a ranking objective "\
                              "should align the loss with what is scored"
        cfg = {'objective': obj, 'group': grp or 'user_day'}
        return Proposal(h, TEMPLATE.format(mode=mode, fams=tuple(fams), cfg=cfg).strip(),
                        '<pool>')
