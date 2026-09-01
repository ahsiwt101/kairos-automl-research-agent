def build(ctx):
    import numpy as np
    n = ctx.data.n

    def rankpct(vals, groups):
        order = np.lexsort((vals, groups))
        g_sorted = groups[order]
        v_sorted = vals[order]
        n_ = len(vals)
        ranks = np.empty(n_, dtype=np.float64)
        i = 0
        while i < n_:
            j = i
            while j < n_ and g_sorted[j] == g_sorted[i]:
                j += 1
            m = j - i
            if m == 1:
                ranks[i] = 0.5
            else:
                k = i
                while k < j:
                    kk = k
                    while kk < j and v_sorted[kk] == v_sorted[k]:
                        kk += 1
                    avg_rank = (k - i + kk - i - 1) / 2.0 / (m - 1)
                    ranks[k:kk] = avg_rank
                    k = kk
            i = j
        out = np.empty(n_, dtype=np.float64)
        out[order] = ranks
        return out

    try:
        s_fm = ctx.refit_score()
    except Exception:
        s_fm = ctx.baseline_score
    s_fm = np.asarray(s_fm, dtype=np.float64)

    user_id = ctx.data.user_id
    video_id = ctx.data.video_id
    date = ctx.data.date
    time_ms = ctx.data.time_ms
    y = ctx.data.y_raw.astype(np.float64)
    labeled = np.ones(n, dtype=np.float64)

    dur = ctx.video_attr('video_duration').astype(np.float64)
    dur = np.where(dur > 0, dur, 1.0)
    logdur = np.log1p(dur)

    tag = ctx.video_attr('tag')
    try:
        video_type = ctx.video_attr('video_type')
    except Exception:
        video_type = np.zeros(n, dtype=np.int64)

    horizon = ctx.window_horizons(date, ctx.OFFICIAL_WINDOWS)

    # per-user prefix stats (all long_view)
    n_lab_u, n_pos_u = ctx.frozen_prefix(user_id, date, y, labeled, horizon)
    user_rate = n_pos_u / np.maximum(n_lab_u, 1.0)

    # user x category
    cat_key = np.unique(np.stack([user_id, video_type], 1), axis=0, return_inverse=True)[1]
    n_lab_uc, n_pos_uc = ctx.frozen_prefix(cat_key, date, y, labeled, horizon)
    alpha = 20.0
    cat_rate = (n_pos_uc + alpha * user_rate) / (n_lab_uc + alpha)

    # user x tag
    tag_key = np.unique(np.stack([user_id, tag], 1), axis=0, return_inverse=True)[1]
    n_lab_ut, n_pos_ut = ctx.frozen_prefix(tag_key, date, y, labeled, horizon)
    tag_rate = (n_pos_ut + alpha * user_rate) / (n_lab_ut + alpha)

    # video global prefix rate
    n_lab_v, n_pos_v = ctx.frozen_prefix(video_id, date, y, labeled, horizon)
    vid_rate = (n_pos_v + 10.0 * 0.1) / (n_lab_v + 10.0)
    vid_pop = np.log1p(n_lab_v)

    # user x duration-bucket affinity for long-view mean duration approx
    dur_bucket = np.floor(logdur * 2).astype(np.int64)
    ud_key = np.unique(np.stack([user_id, dur_bucket], 1), axis=0, return_inverse=True)[1]
    n_lab_ud, n_pos_ud = ctx.frozen_prefix(ud_key, date, y, labeled, horizon)
    ud_rate = (n_pos_ud + alpha * user_rate) / (n_lab_ud + alpha)

    feat_names = ['logdur', 'cat_rate', 'tag_rate', 'vid_rate', 'vid_pop', 'ud_rate', 'user_rate_interact']
    X_hist = np.stack([
        logdur,
        cat_rate,
        tag_rate,
        vid_rate,
        vid_pop,
        ud_rate,
        cat_rate * vid_rate,
    ], axis=1).astype(np.float32)

    import lightgbm as lgb
    train_idx = ctx.fold.idx['train']
    valid_idx = ctx.fold.idx['valid']

    all_idx = np.arange(n)
    test_mask = np.ones(n, dtype=bool)
    test_mask[train_idx] = False
    test_mask[valid_idx] = False
    test_idx = all_idx[test_mask]

    seeds = [11, 23, 37]
    preds_valid = np.zeros(len(valid_idx), dtype=np.float64)
    preds_test = np.zeros(len(test_idx), dtype=np.float64)
    preds_train = np.zeros(len(train_idx), dtype=np.float64)

    for sd in seeds:
        params = dict(objective='binary', num_leaves=63, learning_rate=0.05,
                      n_estimators=400, min_child_samples=50, subsample=0.8,
                      colsample_bytree=0.8, random_state=sd, verbosity=-1)
        model = lgb.LGBMClassifier(**params)
        model.fit(X_hist[train_idx], ctx.data.y_raw[train_idx])
        if len(valid_idx) > 0:
            preds_valid += model.predict_proba(X_hist[valid_idx])[:, 1] / len(seeds)
        if len(test_idx) > 0:
            preds_test += model.predict_proba(X_hist[test_idx])[:, 1] / len(seeds)
        preds_train += model.predict_proba(X_hist[train_idx])[:, 1] / len(seeds)

    s_hist = np.zeros(n, dtype=np.float64)
    s_hist[train_idx] = preds_train
    if len(valid_idx) > 0:
        s_hist[valid_idx] = preds_valid
    if len(test_idx) > 0:
        s_hist[test_idx] = preds_test

    r_fm = rankpct(s_fm, user_id)
    r_hist = rankpct(s_hist, user_id)

    best_w = 0.6
    if len(valid_idx) > 0:
        y_valid = ctx.data.y_raw[valid_idx].astype(np.float64)
        best_score = -1.0
        for w in [0.5, 0.6, 0.7]:
            score_v = w * r_fm[valid_idx] + (1 - w) * r_hist[valid_idx]
            try:
                from scipy.stats import spearmanr
                corr = spearmanr(score_v, y_valid).correlation
            except Exception:
                corr = np.corrcoef(score_v, y_valid)[0, 1]
            if np.isnan(corr):
                corr = -1.0
            if corr > best_score:
                best_score = corr
                best_w = w

    final = best_w * r_fm + (1 - best_w) * r_hist
    X = final.reshape(-1, 1).astype(np.float32)
    names = ['fused_score']
    ctx.check(X, names)
    train_cfg = {'mode': 'scores'}
    return X, names, train_cfg
