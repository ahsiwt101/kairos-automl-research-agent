def build(ctx):
    import numpy as np

    n = ctx.data.n
    user_id = ctx.data.user_id

    def percentile_rank_by_group(group_ids, values):
        order = np.lexsort((values, group_ids))
        sorted_gid = group_ids[order]
        m = len(values)
        change = np.empty(m, dtype=bool)
        change[0] = True
        if m > 1:
            change[1:] = sorted_gid[1:] != sorted_gid[:-1]
        group_start = np.where(change)[0]
        group_sizes = np.diff(np.append(group_start, m))
        group_id_start = np.repeat(group_start, group_sizes)
        pos_in_group = np.arange(m) - group_id_start
        sizes_repeated = np.repeat(group_sizes, group_sizes)
        pct = pos_in_group / np.maximum(sizes_repeated - 1, 1)
        result = np.empty(m, dtype=np.float64)
        result[order] = pct
        return result.astype(np.float32)

    baseline = np.asarray(ctx.baseline_score, dtype=np.float32)
    refit = np.asarray(ctx.refit_score(), dtype=np.float32)
    din = np.asarray(ctx.din_score(), dtype=np.float32)
    cf_score, cf_hist = ctx.cf_score()
    cf_score = np.asarray(cf_score, dtype=np.float32)
    cf_hist = np.asarray(cf_hist, dtype=np.float32)
    aux = np.asarray(ctx.auxiliary_signal('is_click'), dtype=np.float32)

    U, V = ctx.mf_factors(dim=16)
    mf_dot = np.sum(U * V, axis=1).astype(np.float32)

    r_fm = percentile_rank_by_group(user_id, baseline)
    r_refit = percentile_rank_by_group(user_id, refit)
    r_din = percentile_rank_by_group(user_id, din)
    r_cf = percentile_rank_by_group(user_id, cf_score)
    r_mf = percentile_rank_by_group(user_id, mf_dot)
    r_aux = percentile_rank_by_group(user_id, aux)

    fused = (0.34 * r_fm + 0.20 * r_refit + 0.14 * r_din +
             0.13 * r_cf + 0.11 * r_mf + 0.08 * r_aux).astype(np.float32)
    fused_minus_fm = (fused - r_fm).astype(np.float32)

    duration_ms = np.asarray(ctx.col('duration_ms'), dtype=np.float32)
    play_time_ms = np.asarray(ctx.col('play_time_ms'), dtype=np.float32)
    video_duration = np.asarray(ctx.video_attr('video_duration'), dtype=np.float32)
    hourmin = np.asarray(ctx.col('hourmin'), dtype=np.float32)
    tab = np.asarray(ctx.col('tab'), dtype=np.float32)

    feats = [
        baseline, refit, din, cf_score, cf_hist, mf_dot, aux,
        r_fm, r_refit, r_din, r_cf, r_mf, r_aux,
        fused, fused_minus_fm,
        duration_ms, play_time_ms, video_duration, hourmin, tab,
    ]
    names = [
        'baseline_score', 'refit_score', 'din_score', 'cf_score', 'cf_hist',
        'mf_dot', 'aux_click',
        'r_fm', 'r_refit', 'r_din', 'r_cf', 'r_mf', 'r_aux',
        'fused', 'fused_minus_fm',
        'duration_ms', 'play_time_ms', 'video_duration', 'hourmin', 'tab',
    ]

    X = np.stack(feats, axis=1).astype(np.float32)

    ctx.check(X, names)

    train_cfg = {
        'objective': 'binary',
        'group': 'user_day',
        'hparams': {},
        'mode': 'features',
    }

    return X, names, train_cfg
