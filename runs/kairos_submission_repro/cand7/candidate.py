def build(ctx):
    import numpy as np

    def within_user_rank(score, uid):
        order = np.argsort(uid, kind='stable')
        sorted_uid = uid[order]
        sorted_score = score[order]
        n = len(score)
        ranks = np.empty(n, dtype=np.float64)
        start = 0
        for i in range(1, n + 1):
            if i == n or sorted_uid[i] != sorted_uid[start]:
                seg = sorted_score[start:i]
                order2 = np.argsort(np.argsort(seg, kind='stable'), kind='stable')
                ranks[start:i] = (order2 + 1) / (i - start + 1)
                start = i
        out = np.empty(n, dtype=np.float64)
        out[order] = ranks
        return out

    uid = ctx.data.user_id

    m1 = np.asarray(ctx.baseline_score, dtype=np.float64)
    m2 = np.asarray(ctx.din_score(), dtype=np.float64)
    m3 = np.asarray(ctx.refit_score(), dtype=np.float64)
    cf_score, hist_count = ctx.cf_score()
    m4 = np.asarray(cf_score, dtype=np.float64)

    U, V = ctx.mf_factors(dim=16)
    aux_names = ['is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward']
    aux_feats = [np.asarray(ctx.auxiliary_signal(a), dtype=np.float64) for a in aux_names]

    feat_list = [U, V] + [a.reshape(-1, 1) for a in aux_feats]
    Xm5 = np.concatenate(feat_list, axis=1).astype(np.float64)

    y = ctx.data.y_raw.astype(np.float64)

    try:
        import lightgbm as lgb
        train_idx = ctx.fold.idx['train']
        seeds = [11, 23, 37]
        preds = np.zeros(ctx.data.n, dtype=np.float64)
        for s in seeds:
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'verbosity': -1,
                'seed': s,
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_seed': s,
                'min_data_in_leaf': 50,
            }
            dtrain = lgb.Dataset(Xm5[train_idx], label=y[train_idx])
            model = lgb.train(params, dtrain, num_boost_round=200)
            preds += model.predict(Xm5)
        m5 = preds / len(seeds)
    except Exception:
        m5 = Xm5.mean(axis=1)

    r1 = within_user_rank(m1, uid)
    r2 = within_user_rank(m2, uid)
    r3 = within_user_rank(m3, uid)
    r4 = within_user_rank(m4, uid)
    r5 = within_user_rank(m5, uid)

    fused = 0.35 * r1 + 0.20 * r2 + 0.20 * r3 + 0.15 * r4 + 0.10 * r5

    X = fused.reshape(-1, 1).astype(np.float32)
    names = ['fused_rank_score']

    train_cfg = {'mode': 'scores'}

    ctx.check(X, names)
    return X, names, train_cfg
