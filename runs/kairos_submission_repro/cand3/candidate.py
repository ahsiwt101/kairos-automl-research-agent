def build(ctx):
    import numpy as np
    n = ctx.data.n
    y = ctx.data.y_raw.astype(np.float32)
    labeled = np.ones(n, dtype=np.float32)
    date = ctx.data.date
    horizon = ctx.window_horizons(date, ctx.OFFICIAL_WINDOWS)

    user_id = ctx.data.user_id.astype(np.int64)
    author_id = ctx.video_attr('author_id').astype(np.int64)
    tag = ctx.video_attr('tag')
    tag = np.asarray(tag)
    if tag.ndim > 1:
        tag = tag[:, 0]
    tag = tag.astype(np.int64)
    duration = ctx.video_attr('video_duration').astype(np.float32)

    # duration deciles (global, static bucketing - no label leakage)
    finite = np.isfinite(duration)
    qs = np.nanpercentile(duration[finite], np.linspace(0, 100, 11)) if finite.any() else np.linspace(0, 1, 11)
    qs = np.unique(qs)
    if len(qs) < 2:
        qs = np.array([0.0, 1.0])
    dur_bucket = np.searchsorted(qs[1:-1], duration).astype(np.int64)

    def factorize(*arrs):
        stacked = np.stack(arrs, axis=1)
        return np.unique(stacked, axis=0, return_inverse=True)[1].astype(np.int64)

    key_user = user_id
    key_user_dur = factorize(user_id, dur_bucket)
    key_user_tag = factorize(user_id, tag)
    key_user_author = factorize(user_id, author_id)
    key_dur = dur_bucket
    key_tag = tag

    # global rate for priors
    global_rate = float(np.clip(y.mean(), 1e-3, 1 - 1e-3))

    n_lab_dur, n_pos_dur = ctx.frozen_prefix(key_dur, date, y, labeled, horizon)
    prior_dur = ctx.smoothed_rate(n_pos_dur, n_lab_dur, global_rate, 50.0)

    n_lab_tag, n_pos_tag = ctx.frozen_prefix(key_tag, date, y, labeled, horizon)
    prior_tag = ctx.smoothed_rate(n_pos_tag, n_lab_tag, global_rate, 50.0)

    n_lab_user, n_pos_user = ctx.frozen_prefix(key_user, date, y, labeled, horizon)
    user_base = ctx.smoothed_rate(n_pos_user, n_lab_user, global_rate, 20.0)

    n_lab_ud, n_pos_ud = ctx.frozen_prefix(key_user_dur, date, y, labeled, horizon)
    aff_dur = ctx.smoothed_rate(n_pos_ud, n_lab_ud, prior_dur, 20.0)

    n_lab_ut, n_pos_ut = ctx.frozen_prefix(key_user_tag, date, y, labeled, horizon)
    aff_tag = ctx.smoothed_rate(n_pos_ut, n_lab_ut, prior_tag, 20.0)

    n_lab_ua, n_pos_ua = ctx.frozen_prefix(key_user_author, date, y, labeled, horizon)
    aff_author = ctx.smoothed_rate(n_pos_ua, n_lab_ua, user_base, 20.0)

    dev_dur = aff_dur - user_base
    dev_tag = aff_tag - user_base
    dev_author = aff_author - user_base

    log_n_dur = np.log1p(n_lab_ud.astype(np.float32))
    log_n_tag = np.log1p(n_lab_ut.astype(np.float32))
    log_n_author = np.log1p(n_lab_ua.astype(np.float32))
    log_duration = np.log1p(np.nan_to_num(duration, nan=0.0))

    baseline = ctx.baseline_score.astype(np.float32)
    refit = ctx.refit_score().astype(np.float32)
    din = ctx.din_score().astype(np.float32)

    cf_score, cf_hist = ctx.cf_score()
    cf_score = cf_score.astype(np.float32)
    cf_hist = cf_hist.astype(np.float32)

    U, V = ctx.mf_factors(dim=16)
    mf_dot = np.sum(U * V, axis=1).astype(np.float32)

    aux_names = ['is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward']
    aux_feats = [ctx.auxiliary_signal(name).astype(np.float32) for name in aux_names]

    exp_context = ctx.expert_score('context').astype(np.float32)
    exp_item = ctx.expert_score('item').astype(np.float32)
    exp_user = ctx.expert_score('user').astype(np.float32)

    feat_list = [
        aff_dur.astype(np.float32),
        aff_tag.astype(np.float32),
        aff_author.astype(np.float32),
        dev_dur.astype(np.float32),
        dev_tag.astype(np.float32),
        dev_author.astype(np.float32),
        log_n_dur,
        log_n_tag,
        log_n_author,
        log_duration,
        user_base.astype(np.float32),
        baseline,
        refit,
        din,
        cf_score,
        cf_hist,
        mf_dot,
        exp_context,
        exp_item,
        exp_user,
    ] + aux_feats

    names = [
        'aff_dur', 'aff_tag', 'aff_author',
        'dev_dur', 'dev_tag', 'dev_author',
        'log_n_dur', 'log_n_tag', 'log_n_author',
        'log_duration', 'user_base',
        'baseline_score', 'refit_score', 'din_score',
        'cf_score', 'cf_hist', 'mf_dot',
        'expert_context', 'expert_item', 'expert_user',
    ] + aux_names

    X = np.stack(feat_list, axis=1).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    ctx.check(X, names)
    return X, names