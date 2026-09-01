def build(ctx):
    import numpy as np
    import lightgbm as lgb

    n = ctx.data.n
    y = ctx.data.y_raw.astype(np.float32)
    horizon = ctx.window_horizons(ctx.data.date, ctx.OFFICIAL_WINDOWS)
    labeled = np.ones(n, dtype=np.int64)

    uid = ctx.data.user_id
    vid = ctx.data.video_id
    aid = ctx.video_attr('author_id').astype(np.int64)

    def smooth(pos, cnt, prior, alpha):
        return (pos + prior * alpha) / (cnt + alpha)

    global_rate = float(np.mean(y))

    n_lab_u, n_pos_u = ctx.frozen_prefix(uid.astype(np.int64), ctx.data.date, y, labeled, horizon)
    n_lab_i, n_pos_i = ctx.frozen_prefix(vid.astype(np.int64), ctx.data.date, y, labeled, horizon)
    n_lab_a, n_pos_a = ctx.frozen_prefix(aid, ctx.data.date, y, labeled, horizon)

    rate_u = smooth(n_pos_u, n_lab_u, global_rate, 20.0)
    rate_i = smooth(n_pos_i, n_lab_i, global_rate, 20.0)
    rate_a = smooth(n_pos_a, n_lab_a, global_rate, 20.0)

    U, V = ctx.mf_factors(dim=16)
    mf_dot = np.sum(U * V, axis=1, dtype=np.float32)

    duration_ms = ctx.col('duration_ms').astype(np.float32)
    hourmin = ctx.col('hourmin').astype(np.float32)
    video_duration = ctx.video_attr('video_duration').astype(np.float32)
    tab = ctx.col('tab').astype(np.float32)

    feats = [
        rate_u.astype(np.float32), np.log1p(n_lab_u).astype(np.float32),
        rate_i.astype(np.float32), np.log1p(n_lab_i).astype(np.float32),
        rate_a.astype(np.float32), np.log1p(n_lab_a).astype(np.float32),
        mf_dot,
        duration_ms, hourmin, video_duration, tab,
    ]
    feats.append(U)
    feats.append(V)

    Xm = np.concatenate([f.reshape(n, -1) if f.ndim > 1 else f.reshape(n, 1) for f in feats], axis=1).astype(np.float32)

    train_idx = ctx.fold.idx['train']
    valid_idx = ctx.fold.idx['valid']

    preds = np.zeros(n, dtype=np.float64)
    seeds = [11, 23, 37]
    for sd in seeds:
        params = dict(
            objective='binary',
            num_leaves=63,
            learning_rate=0.05,
            min_data_in_leaf=200,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=1,
            seed=sd,
            verbose=-1,
        )
        dtrain = lgb.Dataset(Xm[train_idx], label=y[train_idx])
        booster = lgb.train(params, dtrain, num_boost_round=300)
        preds += booster.predict(Xm) / len(seeds)

    tree_score = preds.astype(np.float32)

    def within_user_rank(score, users):
        order = np.lexsort((score, users))
        sorted_users = users[order]
        n_ = len(score)
        ranks = np.empty(n_, dtype=np.float64)
        start = 0
        i = 0
        while i < n_:
            j = i
            while j < n_ and sorted_users[j] == sorted_users[i]:
                j += 1
            grp = order[i:j]
            m = j - i
            grp_scores = score[grp]
            order_within = np.argsort(grp_scores, kind='mergesort')
            r = np.empty(m, dtype=np.float64)
            r[order_within] = np.arange(1, m + 1)
            ranks[grp] = (r - 0.5) / m
            i = j
        return ranks

    baseline = ctx.baseline_score.astype(np.float64)
    cf_score, hist_count = ctx.cf_score()
    cf_score = cf_score.astype(np.float64)
    aux = ctx.auxiliary_signal('is_click').astype(np.float64)

    r_base = within_user_rank(baseline, uid)
    r_cf = within_user_rank(cf_score, uid)
    r_aux = within_user_rank(aux, uid)
    r_tree = within_user_rank(tree_score.astype(np.float64), uid)

    final = 0.50 * r_base + 0.18 * r_cf + 0.14 * r_aux + 0.18 * r_tree
    X = final.reshape(n, 1).astype(np.float32)
    names = ['fused_rank_score']

    train_cfg = {'mode': 'scores'}

    ctx.check(X, names)
    return X, names, train_cfg