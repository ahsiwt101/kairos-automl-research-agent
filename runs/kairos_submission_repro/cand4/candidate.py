def build(ctx):
    import numpy as np
    import lightgbm as lgb

    n = ctx.data.n
    uid = ctx.data.user_id

    def pct_rank(user_id, value):
        order = np.lexsort((value, user_id))
        sorted_uid = user_id[order]
        nn = len(sorted_uid)
        change = np.empty(nn, dtype=bool)
        change[0] = True
        if nn > 1:
            change[1:] = sorted_uid[1:] != sorted_uid[:-1]
        group_ids = np.cumsum(change) - 1
        first_idx = np.where(change)[0]
        rank0 = np.arange(nn) - first_idx[group_ids]
        group_size = np.bincount(group_ids)
        size_per_row = group_size[group_ids]
        pct = (rank0 + 0.5) / size_per_row
        out = np.empty(nn, dtype=np.float32)
        out[order] = pct.astype(np.float32)
        size_out = np.empty(nn, dtype=np.float32)
        size_out[order] = size_per_row.astype(np.float32)
        return out, size_out

    m1 = ctx.refit_score().astype(np.float32)
    m2 = ctx.din_score().astype(np.float32)
    cf_score, cf_hist = ctx.cf_score()
    m3 = cf_score.astype(np.float32)
    U, V = ctx.mf_factors(16)
    m4 = np.sum(U * V, axis=1).astype(np.float32)
    m5 = ctx.auxiliary_signal('is_click').astype(np.float32)

    pct1, list_size = pct_rank(uid, m1)
    pct2, _ = pct_rank(uid, m2)
    pct3, _ = pct_rank(uid, m3)
    pct4, _ = pct_rank(uid, m4)
    pct5, _ = pct_rank(uid, m5)

    fused = (pct1 + pct2 + pct3 + pct4 + pct5) / 5.0
    fused_wm = 0.4 * pct1 + 0.2 * pct2 + 0.15 * pct3 + 0.15 * pct4 + 0.10 * pct5

    pct_stack = np.stack([pct1, pct2, pct3, pct4, pct5], axis=1)
    spread = pct_stack.max(axis=1) - pct_stack.min(axis=1)

    cols = [m1, m2, m3, m4, m5, pct1, pct2, pct3, pct4, pct5,
            fused, fused_wm, list_size, spread]
    names = ['m_refit', 'm_din', 'm_cf', 'm_mf', 'm_aux',
             'pct_refit', 'pct_din', 'pct_cf', 'pct_mf', 'pct_aux',
             'fused_eq', 'fused_wm', 'list_size', 'pct_spread']

    X = np.stack(cols, axis=1).astype(np.float32)
    ctx.check(X, names)

    y = ctx.data.y_raw.astype(np.float32)
    train_idx = ctx.fold.idx['train']
    valid_idx = ctx.fold.idx['valid']

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_valid = X[valid_idx]
    y_valid = y[valid_idx]

    preds = np.zeros(n, dtype=np.float64)
    seeds = [11, 23, 42]
    for seed in seeds:
        params = dict(
            objective='binary',
            metric='auc',
            num_leaves=31,
            learning_rate=0.05,
            feature_fraction=0.9,
            bagging_fraction=0.8,
            bagging_freq=5,
            min_data_in_leaf=50,
            verbose=-1,
            seed=seed,
        )
        dtrain = lgb.Dataset(X_train, label=y_train)
        dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)
        booster = lgb.train(
            params, dtrain,
            num_boost_round=500,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(30, verbose=False)]
        )
        preds += booster.predict(X, num_iteration=booster.best_iteration).astype(np.float64)

    preds /= len(seeds)
    out = preds.reshape(-1, 1).astype(np.float32)
    out_names = ['fused_score']
    ctx.check(out, out_names)
    return out, out_names, {'mode': 'scores'}
