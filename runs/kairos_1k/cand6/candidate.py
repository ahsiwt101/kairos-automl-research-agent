def build(ctx):
    import numpy as np

    n = ctx.data.n
    y = ctx.data.y_raw.astype(np.float64)
    date = ctx.data.date
    video_id = ctx.data.video_id
    user_id = ctx.data.user_id

    video_duration = ctx.video_attr('video_duration').astype(np.float64)
    log_dur = np.log1p(np.maximum(video_duration, 0))

    # item popularity: count of impressions per video_id in the whole log
    uniq_vid, vid_inv, vid_counts = np.unique(video_id, return_inverse=True, return_counts=True)
    pop = vid_counts[vid_inv].astype(np.float64)
    log_pop = np.log1p(pop)

    def decile(x):
        edges = np.quantile(x, np.linspace(0, 1, 11)[1:-1])
        d = np.searchsorted(edges, x, side='right')
        return np.clip(d, 0, 9).astype(np.int64)

    duration_decile = decile(log_dur)
    pop_decile = decile(log_pop)

    horizon = ctx.fold.horizon
    is_train_prior = date < horizon

    # global rates per decile computed on prior data
    def global_rate_map(dec, mask):
        g = np.zeros(10, dtype=np.float64)
        for d in range(10):
            sel = mask & (dec == d)
            cnt = sel.sum()
            g[d] = y[sel].sum() / cnt if cnt > 0 else y[mask].sum() / max(mask.sum(), 1)
        return g

    global_rate_ud = global_rate_map(duration_decile, is_train_prior)
    global_rate_up = global_rate_map(pop_decile, is_train_prior)

    k = 20.0

    order = np.argsort(date, kind='stable')
    sorted_date = date[order]

    def build_expanding(keys, is_prior_all):
        # keys: array of int category ids (combining user with decile)
        n_local = len(keys)
        sums = np.zeros(n_local, dtype=np.float64)
        cnts = np.zeros(n_local, dtype=np.float64)
        state_sum = {}
        state_cnt = {}
        for day in np.unique(sorted_date):
            day_mask_sorted = sorted_date == day
            idxs = order[day_mask_sorted]
            for i in idxs:
                kk = keys[i]
                sums[i] = state_sum.get(kk, 0.0)
                cnts[i] = state_cnt.get(kk, 0.0)
            for i in idxs:
                if is_prior_all[i] or day < horizon:
                    kk = keys[i]
                    state_sum[kk] = state_sum.get(kk, 0.0) + y[i]
                    state_cnt[kk] = state_cnt.get(kk, 0.0) + 1.0
        return sums, cnts

    # combine user_id with decile into single key
    uid_max = int(user_id.max()) + 1
    key_ud = user_id.astype(np.int64) * 10 + duration_decile
    key_up = user_id.astype(np.int64) * 10 + pop_decile

    is_prior_flag = np.zeros(n, dtype=bool)  # unused placeholder, always false since expanding uses day<horizon logic

    sum_ud, cnt_ud = build_expanding(key_ud, is_prior_flag)
    sum_up, cnt_up = build_expanding(key_up, is_prior_flag)

    gr_ud = global_rate_ud[duration_decile]
    gr_up = global_rate_up[pop_decile]

    ud_rate = (sum_ud + k * gr_ud) / (cnt_ud + k)
    up_rate = (sum_up + k * gr_up) / (cnt_up + k)
    ud_lift = ud_rate - gr_ud
    up_lift = up_rate - gr_up
    ud_n = np.log1p(cnt_ud)
    up_n = np.log1p(cnt_up)

    # user-level long-view duration centroid/std, expanding prior
    sum_dur = np.zeros(n, dtype=np.float64)
    sumsq_dur = np.zeros(n, dtype=np.float64)
    cnt_dur = np.zeros(n, dtype=np.float64)
    state_sum_dur = {}
    state_sumsq_dur = {}
    state_cnt_dur = {}
    for day in np.unique(sorted_date):
        day_mask_sorted = sorted_date == day
        idxs = order[day_mask_sorted]
        for i in idxs:
            uu = user_id[i]
            sum_dur[i] = state_sum_dur.get(uu, 0.0)
            sumsq_dur[i] = state_sumsq_dur.get(uu, 0.0)
            cnt_dur[i] = state_cnt_dur.get(uu, 0.0)
        for i in idxs:
            if y[i] > 0.5:
                uu = user_id[i]
                state_sum_dur[uu] = state_sum_dur.get(uu, 0.0) + log_dur[i]
                state_sumsq_dur[uu] = state_sumsq_dur.get(uu, 0.0) + log_dur[i] ** 2
                state_cnt_dur[uu] = state_cnt_dur.get(uu, 0.0) + 1.0

    global_mean_dur = log_dur[is_train_prior].mean() if is_train_prior.sum() > 0 else log_dur.mean()
    global_std_dur = log_dur[is_train_prior].std() if is_train_prior.sum() > 0 else log_dur.std()
    if not np.isfinite(global_std_dur) or global_std_dur <= 0:
        global_std_dur = 1.0

    has_hist = cnt_dur > 0
    centroid = np.where(has_hist, sum_dur / np.maximum(cnt_dur, 1.0), global_mean_dur)
    mean_sq = np.where(has_hist, sumsq_dur / np.maximum(cnt_dur, 1.0), global_mean_dur ** 2)
    var = mean_sq - centroid ** 2
    var = np.maximum(var, 0.0)
    std = np.sqrt(var)
    std = np.where(has_hist & (std > 1e-6), std, global_std_dur)

    dur_gap = log_dur - centroid
    dur_gap_z = dur_gap / np.maximum(std + 1.0, 1e-3)

    dur_gap = np.nan_to_num(dur_gap, nan=0.0, posinf=0.0, neginf=0.0)
    dur_gap_z = np.nan_to_num(dur_gap_z, nan=0.0, posinf=0.0, neginf=0.0)

    baseline = ctx.baseline_score
    cf_score, cf_hist = ctx.cf_score()
    U, V = ctx.mf_factors(dim=16)

    cols = [
        baseline.astype(np.float32),
        cf_score.astype(np.float32),
        cf_hist.astype(np.float32),
        ud_rate.astype(np.float32),
        ud_lift.astype(np.float32),
        ud_n.astype(np.float32),
        up_rate.astype(np.float32),
        up_lift.astype(np.float32),
        up_n.astype(np.float32),
        dur_gap.astype(np.float32),
        dur_gap_z.astype(np.float32),
        duration_decile.astype(np.float32),
        pop_decile.astype(np.float32),
    ]
    names = [
        'baseline_score', 'cf_score', 'cf_hist_count',
        'ud_rate', 'ud_lift', 'ud_n',
        'up_rate', 'up_lift', 'up_n',
        'dur_gap', 'dur_gap_z',
        'duration_decile', 'pop_decile',
    ]

    for d in range(U.shape[1]):
        cols.append(U[:, d].astype(np.float32))
        names.append(f'mf_u_{d}')
    for d in range(V.shape[1]):
        cols.append(V[:, d].astype(np.float32))
        names.append(f'mf_v_{d}')

    X = np.stack(cols, axis=1).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    ctx.check(X, names)
    return X, names
