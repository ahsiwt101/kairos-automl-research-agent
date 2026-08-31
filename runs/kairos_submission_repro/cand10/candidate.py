def build(ctx):
    import numpy as np

    n = ctx.data.n
    uid = ctx.data.user_id

    fm = ctx.refit_score().astype(np.float64)
    fm_base = ctx.baseline_score.astype(np.float64)
    din = ctx.din_score().astype(np.float64)
    cf_score, cf_hist = ctx.cf_score()
    cf_score = cf_score.astype(np.float64)
    U, V = ctx.mf_factors(dim=16)
    mf = np.sum(U.astype(np.float64) * V.astype(np.float64), axis=1)
    try:
        aux = ctx.auxiliary_signal('is_like').astype(np.float64)
    except Exception:
        aux = np.zeros(n, dtype=np.float64)

    def within_user_rank(score, uid):
        order = np.argsort(uid, kind='stable')
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        sorted_uid = uid[order]
        sorted_score = score[order]

        n_ = len(score)
        ranks = np.empty(n_, dtype=np.float64)

        boundaries = np.nonzero(np.diff(sorted_uid))[0] + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [n_]))

        for s, e in zip(starts, ends):
            grp = sorted_score[s:e]
            m = e - s
            if m <= 1:
                ranks[s:e] = 0.5
                continue
            order_grp = np.argsort(grp, kind='stable')
            rank_grp = np.empty(m, dtype=np.float64)
            rank_grp[order_grp] = np.arange(m)
            ranks[s:e] = rank_grp / (m - 1)

        out = np.empty(n_, dtype=np.float64)
        out[order] = ranks
        return out

    r_fm = within_user_rank(fm, uid)
    r_refit = r_fm  # refit_score used as fm above, keep separate baseline too
    r_din = within_user_rank(din, uid)
    r_cf = within_user_rank(cf_score, uid)
    r_mf = within_user_rank(mf, uid)
    r_aux = within_user_rank(aux, uid)

    fuse = (0.34 * r_fm + 0.14 * r_refit + 0.12 * r_din +
            0.20 * r_cf + 0.13 * r_mf + 0.07 * r_aux)

    disagreement = fuse - r_fm
    stack = np.stack([r_fm, r_refit, r_din, r_cf, r_mf, r_aux], axis=1)
    member_std = np.std(stack, axis=1)

    X = np.column_stack([
        fm_base.astype(np.float32),
        fm.astype(np.float32),
        din.astype(np.float32),
        cf_score.astype(np.float32),
        cf_hist.astype(np.float32),
        mf.astype(np.float32),
        aux.astype(np.float32),
        r_fm.astype(np.float32),
        r_refit.astype(np.float32),
        r_din.astype(np.float32),
        r_cf.astype(np.float32),
        r_mf.astype(np.float32),
        r_aux.astype(np.float32),
        fuse.astype(np.float32),
        disagreement.astype(np.float32),
        member_std.astype(np.float32),
    ]).astype(np.float32)

    names = [
        'baseline_score', 'refit_score', 'din_score', 'cf_score', 'cf_hist',
        'mf_score', 'aux_score',
        'r_fm', 'r_refit', 'r_din', 'r_cf', 'r_mf', 'r_aux',
        'fuse', 'fuse_minus_fm', 'member_std'
    ]

    ctx.check(X, names)
    return X, names