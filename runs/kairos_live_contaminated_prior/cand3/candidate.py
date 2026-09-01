def build(ctx):
    import numpy as np

    n = ctx.data.n
    user_id = ctx.data.user_id

    def within_user_rank(vals):
        vals = np.asarray(vals, dtype=np.float64)
        order = np.argsort(user_id, kind='stable')
        inv = np.empty_like(order)
        inv[order] = np.arange(n)
        sorted_users = user_id[order]
        sorted_vals = vals[order]

        boundaries = np.flatnonzero(np.diff(sorted_users)) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [n]))

        ranks_sorted = np.empty(n, dtype=np.float64)
        for s, e in zip(starts, ends):
            seg = sorted_vals[s:e]
            m = e - s
            order2 = np.argsort(seg, kind='stable')
            r = np.empty(m, dtype=np.float64)
            rr = np.arange(1, m + 1, dtype=np.float64)
            r[order2] = rr
            # average ties
            sorted_seg = seg[order2]
            i = 0
            while i < m:
                j = i
                while j + 1 < m and sorted_seg[j + 1] == sorted_seg[i]:
                    j += 1
                if j > i:
                    avg = r[order2[i:j+1]].mean()
                    r[order2[i:j+1]] = avg
                i = j + 1
            ranks_sorted[s:e] = r / (m + 1)
        result = np.empty(n, dtype=np.float64)
        result[order] = ranks_sorted
        return result

    # member 1: seed-averaged refit score (refit_score has no explicit seed param,
    # so average repeated calls to reduce noise if stochastic; else identical calls are harmless)
    seeds_scores = []
    for _ in range(3):
        seeds_scores.append(np.asarray(ctx.refit_score(), dtype=np.float64))
    m1 = np.mean(seeds_scores, axis=0)

    m2 = np.asarray(ctx.din_score(), dtype=np.float64)

    cf_score, cf_hist = ctx.cf_score()
    m3 = np.asarray(cf_score, dtype=np.float64)

    U, V = ctx.mf_factors(dim=16)
    m4 = np.sum(np.asarray(U, dtype=np.float64) * np.asarray(V, dtype=np.float64), axis=1)

    m5 = np.asarray(ctx.auxiliary_signal('is_click'), dtype=np.float64)

    r1 = within_user_rank(m1)
    r2 = within_user_rank(m2)
    r3 = within_user_rank(m3)
    r4 = within_user_rank(m4)
    r5 = within_user_rank(m5)

    fused = 0.50 * r1 + 0.15 * r2 + 0.15 * r3 + 0.15 * r4 + 0.05 * r5

    X = fused.reshape(-1, 1).astype(np.float32)
    names = ['fused_rank_score']

    ctx.check(X, names)

    train_cfg = {'mode': 'scores'}
    return X, names, train_cfg
