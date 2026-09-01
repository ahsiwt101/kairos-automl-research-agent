def build(ctx):
    import numpy as np
    n = ctx.data.n
    uid = ctx.data.user_id

    def rankify(s):
        s = np.asarray(s, dtype=np.float64)
        order = np.argsort(uid, kind='stable')
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        sorted_uid = uid[order]
        sorted_s = s[order]
        r = np.zeros(len(s), dtype=np.float64)
        start = 0
        N = len(sorted_uid)
        while start < N:
            end = start + 1
            while end < N and sorted_uid[end] == sorted_uid[start]:
                end += 1
            grp = sorted_s[start:end]
            gsize = end - start
            if gsize == 1:
                r[start:end] = 0.5
            else:
                order2 = np.argsort(grp, kind='stable')
                ranks = np.empty(gsize, dtype=np.float64)
                ranks[order2] = np.arange(gsize)
                # average ties
                sorted_grp = grp[order2]
                i = 0
                while i < gsize:
                    j = i + 1
                    while j < gsize and sorted_grp[j] == sorted_grp[i]:
                        j += 1
                    avg_rank = (i + j - 1) / 2.0
                    ranks[order2[i:j]] = avg_rank
                    i = j
                r[start:end] = ranks / (gsize - 1)
            start = end
        out = np.empty(len(s), dtype=np.float64)
        out[order] = r
        return out

    s_fm = ctx.baseline_score
    s_refit = ctx.refit_score()
    s_din = ctx.din_score()
    s_cf, hist_count = ctx.cf_score()
    s_aux = ctx.auxiliary_signal('is_like')

    U, V = ctx.mf_factors(dim=16)
    s_mf = np.sum(U * V, axis=1)

    r_fm = rankify(s_fm)
    r_refit = rankify(s_refit)
    r_din = rankify(s_din)
    r_cf = rankify(s_cf)
    r_mf = rankify(s_mf)
    r_aux = rankify(s_aux)

    final = (0.30 * r_fm + 0.15 * r_refit + 0.10 * r_din +
             0.20 * r_cf + 0.15 * r_mf + 0.10 * r_aux)

    X = final.reshape(-1, 1).astype(np.float32)
    names = ['fused_rank_score']
    train_cfg = {'mode': 'scores'}
    ctx.check(X, names)
    return X, names, train_cfg
