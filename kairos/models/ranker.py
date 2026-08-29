"""Factorization Machine with pluggable objectives, trained on within-user groups.

The point of this module is a controlled ablation: identical features, identical capacity,
identical optimiser - only the loss changes.  The official baseline is `loss='bce'`, which
this reproduces, so every other number here is attributable to the objective alone.

Losses
------
bce         pointwise logloss. The official baseline's objective, and mismatched to the
            metric: it spends capacity on calibration, which within-user ranking discards.
bpr         pairwise, every (pos,neg) pair in a group weighted equally.
bpr_gauc    pairwise with pair weight 1/n_neg(g).  Expanding GAUC's definition,
                GAUC = (1/sum_g n_pos_g) * sum_g (1/n_neg_g) * sum_{(p,n)} 1[s_p > s_n]
            so the n_pos weighting cancels and the metric-exact pair weight is 1/n_neg.
listnet     listwise Plackett-Luce top-1: -log softmax over the group, summed over positives.
lambda_ndcg pairwise weighted by |delta nDCG@5| from swapping the pair (LambdaRank).
primary     0.5*bpr_gauc + 0.5*lambda_ndcg - a surrogate for the actual scored quantity,
            mean(GAUC, nDCG@5).

Groups with n_pos == 0 or n_neg == 0 are dropped from every ranking loss, mirroring the
metric: such users are excluded from GAUC and have a model-independent nDCG.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FM(nn.Module):
    def __init__(self, dim, k=16, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.V = nn.Parameter(torch.empty(dim, k).normal_(0, 0.01, generator=g))
        self.W = nn.Parameter(torch.zeros(dim))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, X):
        """X: (..., F) int64 field ids -> (...) scores"""
        E = self.V[X]                                   # (..., F, k)
        S = E.sum(-2)                                   # (..., k)
        inter = 0.5 * ((S ** 2).sum(-1) - (E ** 2).sum((-2, -1)))
        return self.b + self.W[X].sum(-1) + inter


# ------------------------------------------------------------------ group construction
def build_groups(user_id, date, key='user_week', max_len=32, seed=0, week0=20220408):
    """Partition rows into ranking groups.

    Evaluation groups a user's impressions over the WHOLE split window (median 4-5 rows),
    so 'user_window' is the faithful analogue.  'user_day' makes tighter, more numerous
    groups.  Groups longer than max_len are randomly subsampled, which also pulls the
    training group-size distribution towards the evaluation one.
    """
    if key == 'user_day':
        g = user_id.astype(np.int64) * 100000 + date.astype(np.int64) % 100000
    elif key == 'user_week':
        wk = (date.astype(np.int64) - week0) // 7
        g = user_id.astype(np.int64) * 100 + wk
    elif key == 'user_window':
        g = user_id.astype(np.int64)
    else:
        raise ValueError(key)
    order = np.argsort(g, kind='stable')
    gs = g[order]
    starts = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]])
    sizes = np.diff(np.r_[starts, len(gs)])
    rng = np.random.default_rng(seed)
    rows, keep = [], []
    for s, n in zip(starts, sizes):
        idx = order[s:s + n]
        if n > max_len:
            idx = rng.choice(idx, max_len, replace=False)
        rows.append(idx)
        keep.append(len(idx))
    return rows, np.array(keep)


def pad_groups(rows, X, y, max_len):
    """-> Xp (G,L,F) int64, yp (G,L) float32, mask (G,L) bool"""
    G, F_ = len(rows), X.shape[1]
    L = max_len
    Xp = np.zeros((G, L, F_), dtype=np.int64)
    yp = np.zeros((G, L), dtype=np.float32)
    mk = np.zeros((G, L), dtype=bool)
    for i, idx in enumerate(rows):
        n = len(idx)
        Xp[i, :n] = X[idx]
        yp[i, :n] = y[idx]
        mk[i, :n] = True
    return Xp, yp, mk


# ------------------------------------------------------------------ losses
NEG_INF = -1e9


def _pair_terms(s, y, mask):
    """diff (G,L,L) = s_p - s_n, and a boolean pair mask for (pos, neg) pairs."""
    valid = mask.float()
    pos = (y > 0.5).float() * valid
    neg = (y < 0.5).float() * valid
    pm = pos.unsqueeze(2) * neg.unsqueeze(1)            # (G,L,L)
    diff = s.unsqueeze(2) - s.unsqueeze(1)
    return diff, pm, pos.sum(1), neg.sum(1)


def loss_bpr(s, y, mask, weight='uniform'):
    diff, pm, npos, nneg = _pair_terms(s, y, mask)
    live = (npos > 0) & (nneg > 0)
    if live.sum() == 0:
        return s.sum() * 0.0
    lp = F.logsigmoid(diff) * pm
    per_group = -lp.sum((1, 2))
    if weight == 'gauc':                                # metric-exact: pair weight 1/n_neg
        w = per_group / nneg.clamp(min=1)
        return w[live].sum() / npos[live].sum().clamp(min=1)
    per_group = per_group / (npos * nneg).clamp(min=1)  # plain per-group mean over pairs
    return per_group[live].mean()


def loss_listnet(s, y, mask):
    npos = ((y > 0.5) & mask).sum(1)
    nneg = ((y < 0.5) & mask).sum(1)
    live = (npos > 0) & (nneg > 0)
    if live.sum() == 0:
        return s.sum() * 0.0
    sm = s.masked_fill(~mask, NEG_INF)
    logZ = torch.logsumexp(sm, dim=1, keepdim=True)
    logp = sm - logZ
    per_group = -(logp * (y > 0.5).float() * mask.float()).sum(1)
    return (per_group[live] / npos[live].clamp(min=1)).mean()


def loss_lambda_ndcg(s, y, mask, k=5):
    """LambdaRank: pair weight = |delta nDCG@5| from swapping the two items.

    Ranks are taken from the CURRENT scores (detached), which is the standard LambdaRank
    treatment - the weight is a coefficient on the gradient, not part of the graph.
    """
    diff, pm, npos, nneg = _pair_terms(s, y, mask)
    live = (npos > 0) & (nneg > 0)
    if live.sum() == 0:
        return s.sum() * 0.0
    with torch.no_grad():
        sm = s.masked_fill(~mask, NEG_INF)
        order = sm.argsort(dim=1, descending=True)
        rank = torch.empty_like(order)
        ar = torch.arange(s.shape[1], device=s.device).expand_as(order)
        rank.scatter_(1, order, ar)                     # 0-based rank of each slot
        disc = torch.where(rank < k, 1.0 / torch.log2(rank.float() + 2.0),
                           torch.zeros_like(s))
        # ideal DCG for binary labels: first min(npos,k) discount positions
        cum = torch.cumsum(1.0 / torch.log2(torch.arange(k, device=s.device).float() + 2.0), 0)
        ni = npos.clamp(max=k).long()
        idcg = torch.where(ni > 0, cum[(ni - 1).clamp(min=0)], torch.ones_like(npos))
        dd = (disc.unsqueeze(2) - disc.unsqueeze(1)).abs()
        w = dd / idcg.clamp(min=1e-9).view(-1, 1, 1)
    lp = F.logsigmoid(diff) * pm * w
    per_group = -lp.sum((1, 2))
    return (per_group[live] / (npos * nneg).clamp(min=1)[live]).mean()


def compute_loss(kind, s, y, mask):
    if kind == 'bce':
        return F.binary_cross_entropy_with_logits(s[mask], y[mask])
    if kind == 'bpr':
        return loss_bpr(s, y, mask, 'uniform')
    if kind == 'bpr_gauc':
        return loss_bpr(s, y, mask, 'gauc')
    if kind == 'listnet':
        return loss_listnet(s, y, mask)
    if kind == 'lambda_ndcg':
        return loss_lambda_ndcg(s, y, mask)
    if kind == 'primary':
        return 0.5 * loss_bpr(s, y, mask, 'gauc') + 0.5 * loss_lambda_ndcg(s, y, mask)
    raise ValueError(kind)
