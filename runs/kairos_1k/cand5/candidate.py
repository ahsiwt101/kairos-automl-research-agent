def build(ctx):
    import numpy as np

    n = ctx.data.n
    uid = ctx.data.user_id

    def percentile_rank(score, uid):
        score = np.asarray(score, dtype=np.float64)
        order = np.argsort(uid, kind='stable')
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        sorted_uid = uid[order]
        sorted_score = score[order]
        ranks = np.zeros(len(score), dtype=np.float64)
        start = 0
        L = len(sorted_uid)
        while start < L:
            end = start + 1
            while end < L and sorted_uid[end] == sorted_uid[start]:
                end += 1
            grp = sorted_score[start:end]
            order2 = np.argsort(grp, kind='stable')
            r = np.empty(len(grp))
            r[order2] = np.arange(len(grp))
            if len(grp) > 1:
                r = r / (len(grp) - 1)
            else:
                r[:] = 0.5
            ranks[start:end] = r
            start = end
        out = ranks[inv]
        return out

    baseline = np.asarray(ctx.baseline_score, dtype=np.float64)
    refit = np.asarray(ctx.refit_score(), dtype=np.float64)
    din = np.asarray(ctx.din_score(), dtype=np.float64)
    cf_score, cf_hist = ctx.cf_score()
    cf_score = np.asarray(cf_score, dtype=np.float64)

    U, V = ctx.mf_factors(dim=16)
    mf_dot = np.sum(np.asarray(U, dtype=np.float64) * np.asarray(V, dtype=np.float64), axis=1)

    aux = np.asarray(ctx.auxiliary_signal('is_like'), dtype=np.float64)

    r_fm = percentile_rank(baseline, uid)
    r_refit = percentile_rank(refit, uid)
    r_din = percentile_rank(din, uid)
    r_cf = percentile_rank(cf_score, uid)
    r_mf = percentile_rank(mf_dot, uid)
    r_aux = percentile_rank(aux, uid)

    miss_cf = (cf_hist <= 0).astype(np.float32)

    fused = 0.34*r_fm + 0.20*r_refit + 0.14*r_din + 0.13*r_cf + 0.11*r_mf + 0.08*r_aux
    fused_minus_fm = fused - r_fm

    feats = [
        baseline.astype(np.float32),
        refit.astype(np.float32),
        din.astype(np.float32),
        cf_score.astype(np.float32),
        mf_dot.astype(np.float32),
        aux.astype(np.float32),
        r_fm.astype(np.float32),
        r_refit.astype(np.float32),
        r_din.astype(np.float32),
        r_cf.astype(np.float32),
        r_mf.astype(np.float32),
        r_aux.astype(np.float32),
        fused.astype(np.float32),
        fused_minus_fm.astype(np.float32),
        miss_cf,
        cf_hist.astype(np.float32),
    ]
    names = [
        'baseline_score', 'refit_score', 'din_score', 'cf_score', 'mf_dot',
        'aux_is_like', 'r_fm', 'r_refit', 'r_din', 'r_cf', 'r_mf', 'r_aux',
        'fused_rank', 'fused_minus_fm', 'cf_hist_missing', 'cf_hist_count'
    ]

    X = np.stack(feats, axis=1).astype(np.float32)

    ctx.check(X, names)

    train_cfg = {
        'objective': 'binary',
        'group': 'user_day',
        'hparams': {
            'num_leaves': 31,
            'learning_rate': 0.05,
            'n_estimators': 300,
        },
        'mode': 'features',
    }

    return X, names, train_cfg