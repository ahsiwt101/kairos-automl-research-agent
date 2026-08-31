def build(ctx):
    import numpy as np

    n = ctx.data.n
    uid = ctx.data.user_id

    def within_user_rank(score):
        score = np.asarray(score, dtype=np.float64)
        order = np.argsort(uid, kind='stable')
        inv = np.empty_like(order)
        inv[order] = np.arange(n)
        sorted_uid = uid[order]
        sorted_score = score[order]

        ranks = np.empty(n, dtype=np.float64)
        boundaries = np.nonzero(np.diff(sorted_uid))[0] + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [n]))

        for s, e in zip(starts, ends):
            seg = sorted_score[s:e]
            m = e - s
            if m == 1:
                ranks[s:e] = 0.5
                continue
            order2 = np.argsort(seg, kind='stable')
            r = np.empty(m, dtype=np.float64)
            r[order2] = np.arange(m)
            # average ties
            uniq_vals, inv_idx, counts = np.unique(seg, return_inverse=True, return_counts=True)
            sum_r = np.zeros(len(uniq_vals))
            np.add.at(sum_r, inv_idx, r)
            avg_r = sum_r / counts
            r_final = avg_r[inv_idx]
            ranks[s:e] = (r_final + 0.5) / m

        out = np.empty(n, dtype=np.float64)
        out[order] = ranks
        return out

    m1 = np.asarray(ctx.refit_score(), dtype=np.float64)
    m2 = np.asarray(ctx.din_score(), dtype=np.float64)
    m3 = np.asarray(ctx.baseline_score, dtype=np.float64)
    cf_score, hist_count = ctx.cf_score()
    m4 = np.asarray(cf_score, dtype=np.float64)

    U, V = ctx.mf_factors(dim=16)
    m5 = np.sum(np.asarray(U, dtype=np.float64) * np.asarray(V, dtype=np.float64), axis=1)

    r1 = within_user_rank(m1)
    r2 = within_user_rank(m2)
    r3 = within_user_rank(m3)
    r4 = within_user_rank(m4)
    r5 = within_user_rank(m5)

    s = 0.35 * r1 + 0.30 * r2 + 0.20 * r3 + 0.10 * r4 + 0.05 * r5

    X = s.reshape(-1, 1).astype(np.float32)
    names = ['fused_rank_score']
    train_cfg = {'mode': 'scores'}

    ctx.check(X, names)
    return X, names, train_cfg