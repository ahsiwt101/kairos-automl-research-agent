def build(ctx):
    import numpy as np

    n = ctx.data.n
    user_id = ctx.data.user_id

    def rank_pct(score):
        score = np.asarray(score, dtype=np.float64)
        order = np.argsort(user_id, kind='stable')
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        sorted_users = user_id[order]
        sorted_scores = score[order]
        boundaries = np.flatnonzero(np.diff(sorted_users)) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [len(sorted_users)]))
        ranks = np.empty(len(sorted_scores), dtype=np.float64)
        for s, e in zip(starts, ends):
            seg = sorted_scores[s:e]
            cnt = e - s
            if cnt <= 1:
                ranks[s:e] = 0.5
            else:
                order2 = np.argsort(seg, kind='stable')
                r = np.empty(cnt, dtype=np.float64)
                r[order2] = np.arange(cnt)
                ranks[s:e] = r / (cnt - 1)
        out = np.empty(len(sorted_scores), dtype=np.float64)
        out[order] = ranks
        return out

    m1 = np.asarray(ctx.baseline_score, dtype=np.float64)

    refit_seeds = []
    din_seeds = []
    for _ in range(3):
        refit_seeds.append(np.asarray(ctx.refit_score(), dtype=np.float64))
        din_seeds.append(np.asarray(ctx.din_score(), dtype=np.float64))
    m2 = np.mean(refit_seeds, axis=0)
    m3 = np.mean(din_seeds, axis=0)

    cf_score, hist_count = ctx.cf_score()
    m4 = np.asarray(cf_score, dtype=np.float64)

    U, V = ctx.mf_factors(dim=16)
    m5 = np.sum(np.asarray(U, dtype=np.float64) * np.asarray(V, dtype=np.float64), axis=1)

    r1 = rank_pct(m1)
    r2 = rank_pct(m2)
    r3 = rank_pct(m3)
    r4 = rank_pct(m4)
    r5 = rank_pct(m5)

    final = 0.40 * r1 + 0.25 * r3 + 0.15 * r2 + 0.10 * r4 + 0.10 * r5

    X = final.reshape(-1, 1).astype(np.float32)
    names = ['fused_rank_score']
    train_cfg = {'mode': 'scores'}

    ctx.check(X, names)
    return X, names, train_cfg
