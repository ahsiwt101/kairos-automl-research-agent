def build(ctx):
    import numpy as np

    n = ctx.data.n
    y = ctx.data.y_raw.astype(np.float64)
    uid = ctx.data.user_id
    vdur = ctx.video_attr('video_duration').astype(np.float64)
    horizon = ctx.fold.horizon
    date = ctx.data.date

    # labeled mask: rows dated strictly before horizon are usable as history
    labeled = (date <= horizon).astype(np.int64)

    # duration deciles from overall distribution of valid durations
    valid_dur = vdur[np.isfinite(vdur) & (vdur > 0)]
    if valid_dur.size == 0:
        edges = np.array([0, 1])
    else:
        qs = np.linspace(0, 1, 11)
        edges = np.unique(np.quantile(valid_dur, qs))
        if edges.size < 2:
            edges = np.array([valid_dur.min(), valid_dur.max() + 1])
    dur_dec = np.clip(np.searchsorted(edges, vdur, side='right') - 1, 0, len(edges) - 2)

    # composite key user x dur_dec
    key_ud = np.unique(np.stack([uid, dur_dec], axis=1), axis=0, return_inverse=True)[1]

    n_labeled_ud, n_pos_ud = ctx.frozen_prefix(key_ud, date, ctx.data.y_raw, labeled,
                                                ctx.window_horizons(date, ctx.OFFICIAL_WINDOWS))

    # global rate per dur_dec (using same frozen prefix logic, key = dur_dec alone)
    n_labeled_d, n_pos_d = ctx.frozen_prefix(dur_dec, date, ctx.data.y_raw, labeled,
                                              ctx.window_horizons(date, ctx.OFFICIAL_WINDOWS))
    rate_d = np.where(n_labeled_d > 0, n_pos_d / np.maximum(n_labeled_d, 1), 0.5)

    # global rate per user
    n_labeled_u, n_pos_u = ctx.frozen_prefix(uid, date, ctx.data.y_raw, labeled,
                                              ctx.window_horizons(date, ctx.OFFICIAL_WINDOWS))
    rate_u = np.where(n_labeled_u > 0, n_pos_u / np.maximum(n_labeled_u, 1), 0.5)

    prior_ud = 0.5 * rate_u + 0.5 * rate_d
    a = 20.0
    rate_ud = (n_pos_ud + a * prior_ud) / (n_labeled_ud + a)
    delta_ud = rate_ud - rate_u
    log_n_imp_ud = np.log1p(n_labeled_ud.astype(np.float64))

    # popularity decile: use video popularity proxy via log-count of item labeled history
    vid = ctx.data.video_id
    n_labeled_v, n_pos_v = ctx.frozen_prefix(vid, date, ctx.data.y_raw, labeled,
                                              ctx.window_horizons(date, ctx.OFFICIAL_WINDOWS))
    pop_score = np.log1p(n_labeled_v.astype(np.float64))
    valid_pop = pop_score[np.isfinite(pop_score)]
    if valid_pop.size == 0:
        pedges = np.array([0, 1])
    else:
        qs = np.linspace(0, 1, 11)
        pedges = np.unique(np.quantile(valid_pop, qs))
        if pedges.size < 2:
            pedges = np.array([valid_pop.min(), valid_pop.max() + 1])
    pop_dec = np.clip(np.searchsorted(pedges, pop_score, side='right') - 1, 0, len(pedges) - 2)

    key_up = np.unique(np.stack([uid, pop_dec], axis=1), axis=0, return_inverse=True)[1]
    n_labeled_up, n_pos_up = ctx.frozen_prefix(key_up, date, ctx.data.y_raw, labeled,
                                                ctx.window_horizons(date, ctx.OFFICIAL_WINDOWS))
    n_labeled_p, n_pos_p = ctx.frozen_prefix(pop_dec, date, ctx.data.y_raw, labeled,
                                              ctx.window_horizons(date, ctx.OFFICIAL_WINDOWS))
    rate_p = np.where(n_labeled_p > 0, n_pos_p / np.maximum(n_labeled_p, 1), 0.5)
    prior_up = 0.5 * rate_u + 0.5 * rate_p
    rate_up = (n_pos_up + a * prior_up) / (n_labeled_up + a)
    delta_up = rate_up - rate_u
    log_n_imp_up = np.log1p(n_labeled_up.astype(np.float64))

    # preferred log-duration per user, from positive (long_view) history using frozen prefix trick
    logdur = np.log1p(np.clip(vdur, 0, None))
    # weighted sums via frozen_prefix won't directly give means; approximate with causal_prefix style using cumulative sums per user
    # build using simple grouping with np on labeled history (train-only leak-safe approx using horizon)
    horizon_row = ctx.window_horizons(date, ctx.OFFICIAL_WINDOWS)
    order = np.argsort(uid, kind='stable')
    sorted_uid = uid[order]
    sorted_logdur = logdur[order]
    sorted_y = ctx.data.y_raw[order].astype(np.float64)
    sorted_date = date[order]
    sorted_horizon = horizon_row[order]

    pref_sum = np.zeros(n)
    pref_cnt = np.zeros(n)

    uniq_u, start_idx = np.unique(sorted_uid, return_index=True)
    start_idx = list(start_idx) + [n]
    for i in range(len(uniq_u)):
        s, e = start_idx[i], start_idx[i + 1]
        seg_date = sorted_date[s:e]
        seg_hor = sorted_horizon[s:e]
        seg_y = sorted_y[s:e]
        seg_ld = sorted_logdur[s:e]
        # for each row, use rows dated <= its horizon and y==1 as history
        csum = np.cumsum(seg_ld * seg_y)
        ccnt = np.cumsum(seg_y)
        idx_pos = np.searchsorted(seg_date, seg_hor, side='right') - 1
        idx_pos = np.clip(idx_pos, -1, len(seg_date) - 1)
        valid_mask = idx_pos >= 0
        seg_pref_sum = np.zeros(e - s)
        seg_pref_cnt = np.zeros(e - s)
        seg_pref_sum[valid_mask] = csum[idx_pos[valid_mask]]
        seg_pref_cnt[valid_mask] = ccnt[idx_pos[valid_mask]]
        pref_sum[s:e] = seg_pref_sum
        pref_cnt[s:e] = seg_pref_cnt

    inv_order = np.empty(n, dtype=np.int64)
    inv_order[order] = np.arange(n)
    pref_sum = pref_sum[inv_order]
    pref_cnt = pref_cnt[inv_order]

    global_pref = np.nanmean(logdur[np.isfinite(logdur)]) if np.isfinite(logdur).any() else 0.0
    pref_logdur = np.where(pref_cnt > 0, pref_sum / np.maximum(pref_cnt, 1), global_pref)

    dur_gap = logdur - pref_logdur
    abs_dur_gap = np.abs(dur_gap)

    baseline = ctx.baseline_score.astype(np.float64)
    refit = ctx.refit_score().astype(np.float64)
    din = ctx.din_score().astype(np.float64)
    exp_ctx = ctx.expert_score('context').astype(np.float64)
    exp_item = ctx.expert_score('item').astype(np.float64)
    exp_user = ctx.expert_score('user').astype(np.float64)

    U, V = ctx.mf_factors(dim=16)
    cf_dot = np.sum(U * V, axis=1).astype(np.float64)

    feats = [
        baseline, refit, din, exp_ctx, exp_item, exp_user, cf_dot,
        rate_ud, delta_ud, log_n_imp_ud,
        rate_up, delta_up, log_n_imp_up,
        dur_gap, abs_dur_gap,
        rate_u, rate_d, rate_p,
        logdur, pref_logdur,
    ]
    names = [
        'baseline_score', 'refit_score', 'din_score', 'expert_context', 'expert_item',
        'expert_user', 'cf_dot',
        'rate_user_durdecile', 'delta_user_durdecile', 'log_n_imp_user_durdecile',
        'rate_user_popdecile', 'delta_user_popdecile', 'log_n_imp_user_popdecile',
        'dur_gap', 'abs_dur_gap',
        'rate_user_global', 'rate_durdecile_global', 'rate_popdecile_global',
        'logdur', 'pref_logdur',
    ]

    X = np.stack(feats, axis=1).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    ctx.check(X, names)
    train_cfg = {'objective': 'binary', 'group': 'user_day'}
    return X, names, train_cfg
