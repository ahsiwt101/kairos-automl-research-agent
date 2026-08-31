def build(ctx):
    import numpy as np

    def rank_norm(score):
        score = np.asarray(score, dtype=np.float64)
        uid = ctx.data.user_id
        order = np.lexsort((score, uid))
        n = len(score)
        ranks = np.empty(n, dtype=np.float64)
        sorted_uid = uid[order]
        start = 0
        for i in range(1, n + 1):
            if i == n or sorted_uid[i] != sorted_uid[start]:
                grp = order[start:i]
                m = i - start
                sc = score[grp]
                sidx = np.argsort(sc, kind='mergesort')
                r = np.empty(m, dtype=np.float64)
                r[sidx] = np.arange(1, m + 1)
                sv = sc[sidx]
                j = 0
                while j < m:
                    k = j
                    while k + 1 < m and sv[k + 1] == sv[j]:
                        k += 1
                    if k > j:
                        avg = r[sidx[j:k + 1]].mean()
                        r[sidx[j:k + 1]] = avg
                    j = k + 1
                ranks[grp] = r / (m + 1)
                start = i
        return ranks

    n = ctx.data.n

    m_refit = np.zeros(n, dtype=np.float64)
    for s in range(3):
        m_refit += ctx.refit_score()
    m_refit /= 3.0

    m_din = np.zeros(n, dtype=np.float64)
    for s in range(3):
        m_din += ctx.din_score()
    m_din /= 3.0

    m_ctx = ctx.expert_score('context')
    m_item = ctx.expert_score('item')

    cf_score, hist_count = ctx.cf_score()
    m_cf = cf_score

    U, V = ctx.mf_factors(dim=16)
    m_mf = np.sum(U * V, axis=1)

    m_aux = ctx.auxiliary_signal('is_click')

    r_refit = rank_norm(m_refit)
    r_din = rank_norm(m_din)
    r_ctx = rank_norm(m_ctx)
    r_item = rank_norm(m_item)
    r_cf = rank_norm(m_cf)
    r_mf = rank_norm(m_mf)
    r_aux = rank_norm(m_aux)

    final = (0.42 * r_refit + 0.24 * r_din + 0.09 * r_ctx + 0.08 * r_item +
             0.07 * r_cf + 0.06 * r_mf + 0.04 * r_aux)

    X = final.reshape(-1, 1).astype(np.float32)
    names = ['fused_score']
    train_cfg = {'mode': 'scores'}
    ctx.check(X, names)
    return X, names, train_cfg
