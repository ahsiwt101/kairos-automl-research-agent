def build(ctx):
    import numpy as np

    n = ctx.data.n
    uid = ctx.data.user_id

    s_fm = np.asarray(ctx.refit_score(), dtype=np.float64)
    s_din = np.asarray(ctx.din_score(), dtype=np.float64)
    cf_score, hist_count = ctx.cf_score()
    s_cf = np.asarray(cf_score, dtype=np.float64)
    U, V = ctx.mf_factors(dim=16)
    s_mf = np.sum(np.asarray(U, dtype=np.float64) * np.asarray(V, dtype=np.float64), axis=1)

    def within_user_pct_rank(scores, uid):
        order = np.argsort(uid, kind='stable')
        uid_sorted = uid[order]
        scores_sorted = scores[order]
        n = len(uid)
        result_sorted = np.empty(n, dtype=np.float64)

        boundaries = np.flatnonzero(np.diff(uid_sorted)) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [n]))

        for s, e in zip(starts, ends):
            grp = scores_sorted[s:e]
            m = e - s
            if m == 1:
                result_sorted[s:e] = 0.5
                continue
            ranks = np.argsort(np.argsort(grp, kind='stable'), kind='stable').astype(np.float64)
            # average ties
            order2 = np.argsort(grp, kind='stable')
            sorted_grp = grp[order2]
            rank_avg = np.empty(m, dtype=np.float64)
            i = 0
            while i < m:
                j = i
                while j < m and sorted_grp[j] == sorted_grp[i]:
                    j += 1
                avg_rank = (i + j - 1) / 2.0
                rank_avg[order2[i:j]] = avg_rank
                i = j
            result_sorted[s:e] = rank_avg / (m - 1)

        result = np.empty(n, dtype=np.float64)
        result[order] = result_sorted
        return result

    r_fm = within_user_pct_rank(s_fm, uid)
    r_din = within_user_pct_rank(s_din, uid)
    r_cf = within_user_pct_rank(s_cf, uid)
    r_mf = within_user_pct_rank(s_mf, uid)

    fuse = 0.35 * r_fm + 0.25 * r_din + 0.20 * r_cf + 0.20 * r_mf

    X = fuse.reshape(-1, 1).astype(np.float32)
    names = ['fused_score']

    ctx.check(X, names)

    return X, names, {'mode': 'scores'}
