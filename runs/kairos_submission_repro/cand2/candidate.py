def build(ctx):
    import numpy as np

    n = ctx.data.n
    user_id = ctx.data.user_id

    fm = np.asarray(ctx.refit_score(), dtype=np.float64)
    din = np.asarray(ctx.din_score(), dtype=np.float64)
    cf_score, cf_hist = ctx.cf_score()
    cf = np.asarray(cf_score, dtype=np.float64)
    U, V = ctx.mf_factors(dim=16)
    mf = np.sum(np.asarray(U, dtype=np.float64) * np.asarray(V, dtype=np.float64), axis=1)
    aux = np.asarray(ctx.auxiliary_signal('is_like'), dtype=np.float64)

    def within_user_rank(scores, uid):
        order = np.argsort(uid, kind='stable')
        sorted_uid = uid[order]
        sorted_scores = scores[order]
        n_ = len(uid)
        ranks = np.empty(n_, dtype=np.float64)

        # find group boundaries
        boundaries = np.nonzero(np.diff(sorted_uid))[0] + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [n_]))

        for s, e in zip(starts, ends):
            seg = sorted_scores[s:e]
            m = e - s
            if m == 1:
                ranks[s:e] = 0.5
                continue
            order2 = np.argsort(seg, kind='stable')
            r = np.empty(m, dtype=np.float64)
            # average ranks for ties
            sorted_seg = seg[order2]
            rank_vals = np.arange(m, dtype=np.float64)
            # handle ties: average rank positions for equal values
            i = 0
            while i < m:
                j = i
                while j + 1 < m and sorted_seg[j+1] == sorted_seg[i]:
                    j += 1
                avg_rank = (i + j) / 2.0
                rank_vals[i:j+1] = avg_rank
                i = j + 1
            r[order2] = rank_vals
            ranks[s:e] = r / (m - 1) if m > 1 else 0.5

        out = np.empty(n_, dtype=np.float64)
        out[order] = ranks
        return out

    r_fm = within_user_rank(fm, user_id)
    r_din = within_user_rank(din, user_id)
    r_cf = within_user_rank(cf, user_id)
    r_mf = within_user_rank(mf, user_id)
    r_aux = within_user_rank(aux, user_id)

    fused = 0.40 * r_fm + 0.20 * r_din + 0.15 * r_cf + 0.15 * r_mf + 0.10 * r_aux

    X = fused.reshape(-1, 1).astype(np.float32)
    names = ['fused_rank_blend']

    ctx.check(X, names)
    return X, names, {'mode': 'scores'}
