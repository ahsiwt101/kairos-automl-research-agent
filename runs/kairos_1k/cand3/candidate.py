def build(ctx):
    import numpy as np
    from collections import defaultdict

    n = ctx.data.n
    y = ctx.data.y_raw.astype(np.float64)
    dur = ctx.video_attr('video_duration').astype(np.float64)
    log_dur = np.log1p(np.clip(dur, 0, None))
    uid = ctx.data.user_id
    vid = ctx.data.video_id
    time_ms = ctx.data.time_ms

    order = np.argsort(time_ms, kind='mergesort')

    dec_edges = np.quantile(log_dur, np.linspace(0, 1, 11))
    dur_dec = np.clip(np.searchsorted(dec_edges, log_dur, side='right') - 1, 0, 9)

    vid_count = defaultdict(int)
    counts_so_far = np.zeros(n, dtype=np.float64)
    for idx in order:
        v = vid[idx]
        counts_so_far[idx] = vid_count[v]
        vid_count[v] += 1
    log_pop = np.log1p(counts_so_far)
    pop_edges = np.quantile(log_pop, np.linspace(0, 1, 11))
    pop_dec = np.clip(np.searchsorted(pop_edges, log_pop, side='right') - 1, 0, 9)

    user_pos = defaultdict(float)
    user_cnt = defaultdict(float)
    user_logdur_pos_sum = defaultdict(float)
    user_pos_cnt = defaultdict(float)

    user_global_rate = np.zeros(n)
    dur_gap = np.zeros(n)
    abs_dur_gap = np.zeros(n)

    ud_pos = defaultdict(float)
    ud_cnt = defaultdict(float)
    rate_ud = np.zeros(n)
    log_n_ud = np.zeros(n)
    delta_ud = np.zeros(n)

    dd_pos = defaultdict(float)
    dd_cnt = defaultdict(float)

    up_pos = defaultdict(float)
    up_cnt = defaultdict(float)
    rate_up = np.zeros(n)
    log_n_up = np.zeros(n)
    delta_up = np.zeros(n)

    pd_pos = defaultdict(float)
    pd_cnt = defaultdict(float)

    global_pos = 0.0
    global_cnt = 0.0

    alpha = 20.0
    default_pref = float(np.mean(log_dur))

    for idx in order:
        u = int(uid[idx])
        dd = int(dur_dec[idx])
        pd = int(pop_dec[idx])

        gcnt = user_cnt[u]
        gpos = user_pos[u]
        if gcnt > 0:
            urate = gpos / gcnt
        elif global_cnt > 0:
            urate = global_pos / global_cnt
        else:
            urate = 0.5
        user_global_rate[idx] = urate

        pcnt = user_pos_cnt[u]
        if pcnt > 0:
            pref = user_logdur_pos_sum[u] / pcnt
        else:
            pref = default_pref
        gap = log_dur[idx] - pref
        dur_gap[idx] = gap
        abs_dur_gap[idx] = abs(gap)

        key_ud = (u, dd)
        n_ud = ud_cnt[key_ud]
        p_ud = ud_pos[key_ud]
        dd_c = dd_cnt[dd]
        dd_p = dd_pos[dd]
        if dd_c > 0:
            dd_rate = dd_p / dd_c
        elif global_cnt > 0:
            dd_rate = global_pos / global_cnt
        else:
            dd_rate = 0.5
        prior_ud = 0.5 * urate + 0.5 * dd_rate
        rate_ud[idx] = (p_ud + alpha * prior_ud) / (n_ud + alpha)
        log_n_ud[idx] = np.log1p(n_ud)
        delta_ud[idx] = rate_ud[idx] - urate

        key_up = (u, pd)
        n_up = up_cnt[key_up]
        p_up = up_pos[key_up]
        pd_c = pd_cnt[pd]
        pd_p = pd_pos[pd]
        if pd_c > 0:
            pd_rate = pd_p / pd_c
        elif global_cnt > 0:
            pd_rate = global_pos / global_cnt
        else:
            pd_rate = 0.5
        prior_up = 0.5 * urate + 0.5 * pd_rate
        rate_up[idx] = (p_up + alpha * prior_up) / (n_up + alpha)
        log_n_up[idx] = np.log1p(n_up)
        delta_up[idx] = rate_up[idx] - urate

        yy = y[idx]
        user_cnt[u] += 1
        user_pos[u] += yy
        if yy > 0:
            user_logdur_pos_sum[u] += log_dur[idx]
            user_pos_cnt[u] += 1
        ud_cnt[key_ud] += 1
        ud_pos[key_ud] += yy
        dd_cnt[dd] += 1
        dd_pos[dd] += yy
        up_cnt[key_up] += 1
        up_pos[key_up] += yy
        pd_cnt[pd] += 1
        pd_pos[pd] += yy
        global_cnt += 1
        global_pos += yy

    baseline = ctx.baseline_score
    refit = ctx.refit_score()
    din = ctx.din_score()
    cf_score, cf_hist = ctx.cf_score()

    feats = [baseline, refit, din, cf_score, cf_hist,
              user_global_rate, dur_gap, abs_dur_gap,
              rate_ud, log_n_ud, delta_ud,
              rate_up, log_n_up, delta_up,
              log_dur, log_pop]
    names = ['baseline_score', 'refit_score', 'din_score', 'cf_score', 'cf_hist',
              'user_global_rate', 'dur_gap', 'abs_dur_gap',
              'rate_user_durdec', 'log_n_user_durdec', 'delta_user_durdec',
              'rate_user_popdec', 'log_n_user_popdec', 'delta_user_popdec',
              'log_duration', 'log_pop']

    X = np.stack(feats, axis=1).astype(np.float32)
    ctx.check(X, names)
    return X, names
