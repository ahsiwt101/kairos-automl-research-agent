def build(ctx):
    import numpy as np

    n = ctx.data.n
    date = ctx.data.date
    y = ctx.data.y_raw.astype(np.float32)
    user_id = ctx.data.user_id
    video_duration = ctx.video_attr('video_duration').astype(np.float64)
    video_duration = np.nan_to_num(video_duration, nan=0.0)
    video_duration = np.clip(video_duration, 1.0, None)

    # duration deciles based on global distribution
    edges = np.quantile(video_duration, np.linspace(0, 1, 11))
    edges[0] -= 1.0
    edges[-1] += 1.0
    decile = np.digitize(video_duration, edges[1:-1], right=True).astype(np.int64)

    # composite key user x decile
    stacked = np.stack([user_id.astype(np.int64), decile], axis=1)
    _, key_ud = np.unique(stacked, axis=0, return_inverse=True)
    key_d = decile
    key_u = user_id.astype(np.int64)

    labeled = np.ones(n, dtype=bool)
    horizon = ctx.window_horizons(date, ctx.OFFICIAL_WINDOWS)

    n_labeled_ud, n_pos_ud = ctx.frozen_prefix(key_ud, date, y, labeled, horizon)
    n_labeled_d, n_pos_d = ctx.frozen_prefix(key_d, date, y, labeled, horizon)
    n_labeled_u, n_pos_u = ctx.frozen_prefix(key_u, date, y, labeled, horizon)

    global_mean = float(np.mean(y)) if n > 0 else 0.5

    global_rate_d = ctx.smoothed_rate(n_pos_d, n_labeled_d, global_mean, 50.0)
    featureA = ctx.smoothed_rate(n_pos_ud, n_labeled_ud, global_rate_d, 20.0)
    user_rate = ctx.smoothed_rate(n_pos_u, n_labeled_u, global_mean, 20.0)
    featureB = featureA - user_rate

    conf_ud = np.log1p(n_labeled_ud.astype(np.float64))

    log_duration = np.log1p(video_duration)

    # list-relative duration percentile within user's rows (current window)
    def group_rank_pct(keys, values):
        order = np.argsort(keys, kind='stable')
        sorted_keys = keys[order]
        sorted_vals = values[order]
        m = len(keys)
        ranks = np.empty(m, dtype=np.float64)
        start = 0
        while start < m:
            end = start
            k = sorted_keys[start]
            while end < m and sorted_keys[end] == k:
                end += 1
            group_vals = sorted_vals[start:end]
            order2 = np.argsort(group_vals, kind='stable')
            r = np.empty(len(group_vals), dtype=np.float64)
            r[order2] = np.arange(len(group_vals))
            denom = max(len(group_vals) - 1, 1)
            pct = r / denom
            ranks[start:end] = pct
            start = end
        out = np.empty(m, dtype=np.float64)
        out[order] = ranks
        return out

    E = group_rank_pct(user_id.astype(np.int64), video_duration)

    baseline = ctx.baseline_score.astype(np.float64)

    X = np.stack([
        baseline,
        featureA.astype(np.float64),
        featureB.astype(np.float64),
        conf_ud,
        decile.astype(np.float64),
        log_duration,
        E,
    ], axis=1).astype(np.float32)

    names = [
        'baseline_score',
        'user_decile_lv_rate_smoothed',
        'user_decile_lv_rate_minus_user_rate',
        'user_decile_conf_log1p',
        'duration_decile',
        'log_duration',
        'duration_pct_within_user',
    ]

    ctx.check(X, names)
    return X, names
