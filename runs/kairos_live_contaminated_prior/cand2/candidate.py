def build(ctx):
    import numpy as np
    n = ctx.data.n
    uid = ctx.data.user_id

    def rank_within_user(score):
        score = np.asarray(score, dtype=np.float64)
        order = np.argsort(uid, kind='stable')
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        sorted_uid = uid[order]
        sorted_score = score[order]
        ranks = np.empty(len(order), dtype=np.float64)
        start = 0
        L = len(order)
        while start < L:
            end = start
            u = sorted_uid[start]
            while end < L and sorted_uid[end] == u:
                end += 1
            seg = sorted_score[start:end]
            m = end - start
            if m == 1:
                ranks[start:end] = 0.5
            else:
                order2 = np.argsort(seg, kind='stable')
                r = np.empty(m, dtype=np.float64)
                # average rank for ties
                sorted_seg = seg[order2]
                rr = np.arange(1, m + 1, dtype=np.float64)
                # handle ties by averaging
                i = 0
                while i < m:
                    j = i
                    while j < m and sorted_seg[j] == sorted_seg[i]:
                        j += 1
                    avg_rank = (rr[i] + rr[j-1]) / 2.0
                    r[order2[i:j]] = avg_rank
                    i = j
                ranks[start:end] = r / (m + 1.0)
            start = end
        out = np.empty(L, dtype=np.float64)
        out[order] = ranks
        return out

    m1 = ctx.baseline_score
    m2 = ctx.din_score()
    m3 = ctx.refit_score()
    cf_score, hist_count = ctx.cf_score()
    m4 = cf_score

    U, V = ctx.mf_factors(dim=16)
    m5 = np.sum(U * V, axis=1)

    try:
        m6 = ctx.auxiliary_signal('is_click')
    except Exception:
        m6 = None

    r1 = rank_within_user(m1)
    r2 = rank_within_user(m2)
    r3 = rank_within_user(m3)
    r4 = rank_within_user(m4)
    r5 = rank_within_user(m5)

    fused = 0.34 * r1 + 0.20 * r2 + 0.18 * r3 + 0.12 * r4 + 0.10 * r5
    weight_sum = 0.34 + 0.20 + 0.18 + 0.12 + 0.10

    if m6 is not None:
        r6 = rank_within_user(m6)
        fused = fused + 0.06 * r6
        weight_sum += 0.06

    fused = fused / weight_sum

    X = fused.reshape(-1, 1).astype(np.float32)
    names = ['fused_rank_score']
    train_cfg = {'mode': 'scores'}

    ctx.check(X, names)
    return X, names, train_cfg