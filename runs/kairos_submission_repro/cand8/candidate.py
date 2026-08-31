def build(ctx):
    import numpy as np
    import lightgbm as lgb

    n = ctx.data.n
    y = ctx.data.y_raw.astype(np.float32)
    uid = ctx.data.user_id

    base = ctx.refit_score()
    din = ctx.din_score()
    cf_score, cf_hist = ctx.cf_score()

    def user_rank(uid, score):
        order = np.lexsort((score, uid))
        sorted_uid = uid[order]
        n = len(uid)
        ranks = np.empty(n, dtype=np.float64)
        start = 0
        i = 1
        idx_in_group = np.arange(n)
        # compute rank within each user group via searchsorted boundaries
        boundaries = np.flatnonzero(np.diff(sorted_uid)) + 1
        boundaries = np.concatenate(([0], boundaries, [n]))
        out = np.empty(n, dtype=np.float64)
        for b in range(len(boundaries) - 1):
            s, e = boundaries[b], boundaries[b+1]
            grp = order[s:e]
            size = e - s
            # rank within group (already sorted by score ascending)
            out[grp] = (np.arange(size) + 1) / (size + 1)
        return out.astype(np.float32)

    # feature matrix for lightgbm member
    U, V = ctx.mf_factors(dim=16)
    cf_dot = np.sum(U * V, axis=1).astype(np.float32)

    feat_list = [base, din, cf_score, cf_hist.astype(np.float32), cf_dot]
    feat_names = ['refit_score', 'din_score', 'cf_score', 'cf_hist', 'cf_dot']
    X = np.stack(feat_list, axis=1).astype(np.float32)

    train_idx = ctx.fold.idx['train']
    valid_idx = ctx.fold.idx['valid']

    seeds = [0, 1, 2]
    preds = np.zeros(n, dtype=np.float64)

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'min_data_in_leaf': 50,
    }

    for seed in seeds:
        p = dict(params)
        p['seed'] = seed
        p['bagging_seed'] = seed
        p['feature_fraction_seed'] = seed
        dtrain = lgb.Dataset(X[train_idx], label=y[train_idx])
        dvalid = lgb.Dataset(X[valid_idx], label=y[valid_idx], reference=dtrain)
        model = lgb.train(
            p, dtrain, num_boost_round=300,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(30, verbose=False)]
        )
        preds += model.predict(X, num_iteration=model.best_iteration).astype(np.float64)

    preds /= len(seeds)
    m1_score = preds.astype(np.float32)

    r_m1 = user_rank(uid, m1_score)
    r_m2 = user_rank(uid, din)
    r_m3 = user_rank(uid, cf_score)

    final = 0.55 * r_m1 + 0.30 * r_m2 + 0.15 * r_m3
    final = final.astype(np.float32).reshape(-1, 1)

    ctx.check(final, ['final_score'])
    return final, ['final_score'], {'mode': 'scores'}
