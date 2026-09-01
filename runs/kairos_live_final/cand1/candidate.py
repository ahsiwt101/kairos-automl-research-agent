def build(ctx):
    import numpy as np
    try:
        import lightgbm as lgb
    except Exception:
        lgb = None

    n = ctx.data.n
    user_id = ctx.data.user_id
    date = ctx.data.date

    def group_percentile(vals, groups):
        vals = np.asarray(vals, dtype=np.float64)
        groups = np.asarray(groups)
        order = np.lexsort((vals, groups))
        sorted_groups = groups[order]
        m = len(vals)
        change = np.empty(m, dtype=bool)
        change[0] = True
        if m > 1:
            change[1:] = sorted_groups[1:] != sorted_groups[:-1]
        group_start = np.where(change)[0]
        group_id = np.cumsum(change) - 1
        start_of_group = group_start[group_id]
        pos_in_group = np.arange(m) - start_of_group
        group_end = np.empty(len(group_start), dtype=np.int64)
        if len(group_start) > 1:
            group_end[:-1] = group_start[1:]
        group_end[-1] = m
        size_of_group = group_end[group_id] - start_of_group
        pct = pos_in_group / np.maximum(size_of_group - 1, 1)
        result = np.empty(m, dtype=np.float64)
        result[order] = pct
        return result

    # score A: baseline
    A = np.asarray(ctx.baseline_score, dtype=np.float64)

    # score B: cf score
    cf_score, hist_count = ctx.cf_score()
    B = np.asarray(cf_score, dtype=np.float64)

    # score C: dot of mf factors
    U, V = ctx.mf_factors(dim=16)
    C = np.sum(U * V, axis=1).astype(np.float64)

    # score D: auxiliary signal (average of a few signals)
    aux_names = ['is_click', 'is_like', 'is_follow']
    aux_sum = np.zeros(n, dtype=np.float64)
    cnt = 0
    for nm in aux_names:
        try:
            aux_sum += np.asarray(ctx.auxiliary_signal(nm), dtype=np.float64)
            cnt += 1
        except Exception:
            pass
    D = aux_sum / max(cnt, 1)

    # score E: lightgbm trained only on non-calibrated inputs
    horizon = ctx.window_horizons(date, ctx.OFFICIAL_WINDOWS)
    y = ctx.data.y_raw.astype(np.float64)
    labeled = np.ones(n, dtype=bool)

    author_id = ctx.video_attr('author_id')
    tag = ctx.video_attr('tag')
    duration_ms = ctx.col('duration_ms')
    dur_bucket = (np.asarray(duration_ms) // 5000).astype(np.int64)

    # composite key user+duration bucket
    stacked = np.stack([user_id, dur_bucket], axis=1)
    _, user_dur_key = np.unique(stacked, axis=0, return_inverse=True)

    keys_list = {
        'user': user_id,
        'author': author_id,
        'tag': tag,
        'user_dur': user_dur_key,
    }

    global_rate = float(np.mean(y)) if n > 0 else 0.1

    feat_cols = []
    feat_names = []
    for kname, karr in keys_list.items():
        n_lab, n_pos = ctx.frozen_prefix(karr, date, y, labeled, horizon)
        rate = ctx.smoothed_rate(n_pos, n_lab, global_rate, 10.0)
        feat_cols.append(np.asarray(n_lab, dtype=np.float64))
        feat_names.append(f'{kname}_count')
        feat_cols.append(np.asarray(rate, dtype=np.float64))
        feat_names.append(f'{kname}_rate')

    hourmin = np.asarray(ctx.col('hourmin'), dtype=np.float64)
    hour = np.floor(hourmin / 100.0)
    feat_cols.append(hour)
    feat_names.append('hour')

    dow = (date.astype(np.int64) % 7).astype(np.float64)
    feat_cols.append(dow)
    feat_names.append('dow')

    video_duration = np.asarray(ctx.video_attr('video_duration'), dtype=np.float64)
    feat_cols.append(video_duration)
    feat_names.append('video_duration')

    tab = np.asarray(ctx.col('tab'), dtype=np.float64)
    feat_cols.append(tab)
    feat_names.append('tab')

    Xe = np.stack(feat_cols, axis=1).astype(np.float32)

    E = np.zeros(n, dtype=np.float64)
    if lgb is not None:
        train_idx = ctx.fold.idx['train']
        seeds = [11, 23, 37]
        preds_sum = np.zeros(n, dtype=np.float64)
        for sd in seeds:
            dtrain = lgb.Dataset(Xe[train_idx], label=y[train_idx])
            params = {
                'objective': 'binary',
                'learning_rate': 0.05,
                'num_leaves': 31,
                'verbose': -1,
                'seed': sd,
            }
            booster = lgb.train(params, dtrain, num_boost_round=500)
            preds_sum += booster.predict(Xe)
        E = preds_sum / len(seeds)
    else:
        E = A.copy()

    # within-user percentile ranks
    rA = group_percentile(A, user_id)
    rB = group_percentile(B, user_id)
    rC = group_percentile(C, user_id)
    rD = group_percentile(D, user_id)
    rE = group_percentile(E, user_id)

    w_A = 0.55
    w_B = 0.15
    w_C = 0.10
    w_D = 0.10
    w_E = 0.10

    final = (w_A * rA + w_B * rB + w_C * rC + w_D * rD + w_E * rE).astype(np.float32)
    final = final.reshape(-1, 1)

    names = ['fused_score']
    train_cfg = {'mode': 'scores'}

    ctx.check(final, names)
    return final, names, train_cfg
