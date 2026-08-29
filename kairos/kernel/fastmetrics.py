"""Vectorised GAUC / nDCG@5 — must be numerically identical to the official evaluate.py.

The official implementation is the sole authority on scoring (its own docstring says so).
This module exists only because the reference is a pure-Python per-user sort loop, and the
agent evaluates candidates thousands of times.  `verify_exact()` is the contract: if it
ever fails, the fast path is wrong and must not be used.

Tie semantics replicated deliberately:
  * nDCG   - reference does `lst.sort(key=lambda x: -x[0])`, a STABLE sort, so rows with
             equal scores keep their original relative order.  We reproduce this with a
             lexsort whose final key is the original row index.
  * GAUC   - reference uses Mann-Whitney with average ranks over tied score blocks.
"""
import numpy as np


def factorize(keys):
    """Map arbitrary group keys -> contiguous int32 ids. Group order does not affect any
    metric (nDCG averages over groups, GAUC is a weighted sum), so ordering is arbitrary."""
    uniq, inv = np.unique(np.asarray(keys), return_inverse=True)
    return inv.astype(np.int32), len(uniq)


def fast_evaluate(group_ids, labels, scores, k=5, n_groups=None):
    """group_ids: int array of pre-factorized user ids. labels: 0/1. scores: float.
    Returns dict matching evaluate.evaluate()."""
    g_all = np.asarray(group_ids)
    y_all = np.asarray(labels, dtype=np.float64)
    s_all = np.asarray(scores, dtype=np.float64)
    n = g_all.shape[0]
    if n == 0:
        return {'GAUC': 0.5, f'nDCG@{k}': 0.0, 'primary': 0.25, 'users': 0, 'rows': 0}

    # stable sort: primary group asc, then score DESC, ties broken by original row order
    order = np.lexsort((np.arange(n), -s_all, g_all))
    g, y, s = g_all[order], y_all[order], s_all[order]

    # ---- group boundaries -------------------------------------------------
    new_g = np.empty(n, dtype=bool)
    new_g[0] = True
    np.not_equal(g[1:], g[:-1], out=new_g[1:])
    starts = np.flatnonzero(new_g)
    G = starts.shape[0]
    sizes = np.diff(np.append(starts, n))
    gid = np.repeat(np.arange(G), sizes)              # group index per row
    pos_desc = np.arange(n) - starts[gid]             # 0-based rank, score-descending

    npos = np.bincount(gid, weights=y, minlength=G)
    nneg = sizes - npos

    # ---- nDCG@k -----------------------------------------------------------
    disc = 1.0 / np.log2(np.arange(k) + 2.0)          # gain 2^rel-1 == rel for binary
    topk = pos_desc < k
    dcg = np.bincount(gid[topk], weights=y[topk] * disc[pos_desc[topk]], minlength=G)
    cum = np.cumsum(disc)
    npos_i = np.minimum(npos.astype(np.int64), k)
    idcg = np.where(npos_i > 0, cum[np.maximum(npos_i - 1, 0)], 0.0)
    ndcg_per_group = np.where(idcg > 0, dcg / np.where(idcg > 0, idcg, 1.0), 0.0)
    ndcg = float(ndcg_per_group.mean())

    # ---- GAUC (Mann-Whitney, average ranks within tied score blocks) ------
    new_b = np.empty(n, dtype=bool)
    new_b[0] = True
    np.logical_or(g[1:] != g[:-1], s[1:] != s[:-1], out=new_b[1:])
    b_start = np.flatnonzero(new_b)
    b_size = np.diff(np.append(b_start, n))
    b_gid = gid[b_start]
    # ascending 1-based rank of a row at descending 0-based position p in a group of size m
    # is (m - p); a tied block spanning b_size positions therefore averages to:
    p0 = b_start - starts[b_gid]
    avg_rank_block = sizes[b_gid] - p0 - (b_size - 1) / 2.0
    avg_rank = np.repeat(avg_rank_block, b_size)

    srank = np.bincount(gid, weights=y * avg_rank, minlength=G)
    valid = (npos > 0) & (nneg > 0)
    denom = np.where(valid, npos * nneg, 1.0)
    auc_g = (srank - npos * (npos + 1) / 2.0) / denom
    w = npos[valid]
    gauc = float((w * auc_g[valid]).sum() / w.sum()) if w.sum() > 0 else 0.5

    return {'GAUC': gauc, f'nDCG@{k}': ndcg, 'primary': (gauc + ndcg) / 2.0,
            'users': int(G), 'rows': int(n)}


def per_group_metrics(group_ids, labels, scores, k=5):
    """Same computation, but returns the PER-USER vectors (auc, ndcg, npos, size).
    This is the substrate for the diagnostic layer: every slice metric the agent reasons
    about is an aggregation of these, so slicing never costs another full evaluation."""
    g_all = np.asarray(group_ids)
    y_all = np.asarray(labels, dtype=np.float64)
    s_all = np.asarray(scores, dtype=np.float64)
    n = g_all.shape[0]
    order = np.lexsort((np.arange(n), -s_all, g_all))
    g, y, s = g_all[order], y_all[order], s_all[order]

    new_g = np.empty(n, dtype=bool); new_g[0] = True
    np.not_equal(g[1:], g[:-1], out=new_g[1:])
    starts = np.flatnonzero(new_g); G = starts.shape[0]
    sizes = np.diff(np.append(starts, n))
    gid = np.repeat(np.arange(G), sizes)
    pos_desc = np.arange(n) - starts[gid]
    npos = np.bincount(gid, weights=y, minlength=G)
    nneg = sizes - npos

    disc = 1.0 / np.log2(np.arange(k) + 2.0)
    topk = pos_desc < k
    dcg = np.bincount(gid[topk], weights=y[topk] * disc[pos_desc[topk]], minlength=G)
    cum = np.cumsum(disc)
    npos_i = np.minimum(npos.astype(np.int64), k)
    idcg = np.where(npos_i > 0, cum[np.maximum(npos_i - 1, 0)], 0.0)
    ndcg_g = np.where(idcg > 0, dcg / np.where(idcg > 0, idcg, 1.0), 0.0)

    new_b = np.empty(n, dtype=bool); new_b[0] = True
    np.logical_or(g[1:] != g[:-1], s[1:] != s[:-1], out=new_b[1:])
    b_start = np.flatnonzero(new_b); b_size = np.diff(np.append(b_start, n))
    b_gid = gid[b_start]
    p0 = b_start - starts[b_gid]
    avg_rank = np.repeat(sizes[b_gid] - p0 - (b_size - 1) / 2.0, b_size)
    srank = np.bincount(gid, weights=y * avg_rank, minlength=G)
    valid = (npos > 0) & (nneg > 0)
    denom = np.where(valid, npos * nneg, 1.0)
    auc_g = np.where(valid, (srank - npos * (npos + 1) / 2.0) / denom, np.nan)

    # group_key[i] = the original group id for row-group i
    group_key = g[starts]
    return {'group_key': group_key, 'auc': auc_g, 'ndcg': ndcg_g,
            'npos': npos, 'size': sizes, 'valid_for_gauc': valid}
