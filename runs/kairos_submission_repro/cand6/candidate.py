def build(ctx):
    import numpy as np

    n = ctx.data.n
    user_id = ctx.data.user_id

    baseline = np.asarray(ctx.refit_score(), dtype=np.float32)
    din = np.asarray(ctx.din_score(), dtype=np.float32)
    cf_score, cf_hist = ctx.cf_score()
    cf_score = np.asarray(cf_score, dtype=np.float32)
    cf_hist = np.asarray(cf_hist, dtype=np.float32)
    U, V = ctx.mf_factors(dim=16)
    mf_score = np.sum(np.asarray(U, dtype=np.float32) * np.asarray(V, dtype=np.float32), axis=1).astype(np.float32)
    aux = np.asarray(ctx.auxiliary_signal('is_like'), dtype=np.float32)

    def per_user_pct_rank(x, uid):
        order = np.argsort(uid, kind='mergesort')
        uid_sorted = uid[order]
        x_sorted = x[order]
        n_ = len(x_sorted)
        ranks = np.empty(n_, dtype=np.float32)

        boundaries = np.flatnonzero(np.diff(uid_sorted)) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [n_]))

        for s, e in zip(starts, ends):
            seg = x_sorted[s:e]
            m = e - s
            if m == 1:
                ranks[s:e] = 0.5
            else:
                order_seg = np.argsort(seg, kind='mergesort')
                rank_seg = np.empty(m, dtype=np.float32)
                rank_seg[order_seg] = np.arange(m, dtype=np.float32)
                ranks[s:e] = rank_seg / (m - 1)

        out = np.empty(n_, dtype=np.float32)
        out[order] = ranks
        return out

    r_baseline = per_user_pct_rank(baseline, user_id)
    r_din = per_user_pct_rank(din, user_id)
    r_cf = per_user_pct_rank(cf_score, user_id)
    r_mf = per_user_pct_rank(mf_score, user_id)
    r_aux = per_user_pct_rank(aux, user_id)

    fused = (r_baseline + r_din + r_cf + r_mf + r_aux) / 5.0
    fused = fused.astype(np.float32)

    cols = [
        baseline, din, cf_score, cf_hist, mf_score, aux,
        r_baseline, r_din, r_cf, r_mf, r_aux,
        fused,
    ]
    names = [
        'baseline_score', 'din_score', 'cf_score', 'cf_hist', 'mf_score', 'aux_like',
        'r_baseline', 'r_din', 'r_cf', 'r_mf', 'r_aux',
        'fused_rank',
    ]

    X = np.stack(cols, axis=1).astype(np.float32)

    ctx.check(X, names)
    return X, names
