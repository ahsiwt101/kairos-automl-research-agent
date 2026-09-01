def build(ctx):
    import numpy as np
    n = ctx.data.n
    y = ctx.data.y_raw.astype(np.float64)
    uid = ctx.data.user_id
    vid = ctx.data.video_id
    date = ctx.data.date
    time_ms = ctx.data.time_ms

    train_idx = ctx.fold.idx['train']
    valid_idx = ctx.fold.idx['valid']
    labeled = np.zeros(n, dtype=bool)
    labeled[train_idx] = True
    labeled[valid_idx] = True

    horizon = ctx.window_horizons(date, ctx.OFFICIAL_WINDOWS)

    dur = ctx.video_attr('video_duration').astype(np.float64)
    play = ctx.col('play_time_ms').astype(np.float64)
    is_click = ctx.col('is_click').astype(np.float64)

    global_prior = float(np.mean(y[labeled])) if labeled.any() else 0.1

    # video-level frozen prefix stats
    n_lab_v, n_pos_v = ctx.frozen_prefix(vid, date, y, labeled, horizon)
    rate_v = ctx.smoothed_rate(n_pos_v, n_lab_v, global_prior, 50.0)

    n_lab_v_click, n_pos_v_click = ctx.frozen_prefix(vid, date, is_click, labeled, horizon)
    click_rate_v = ctx.smoothed_rate(n_pos_v_click, n_lab_v_click, float(np.mean(is_click[labeled])) if labeled.any() else 0.1, 50.0)

    # tag-level frozen prefix stats
    tag = ctx.video_attr('tag')
    n_lab_t, n_pos_t = ctx.frozen_prefix(tag, date, y, labeled, horizon)
    rate_t = ctx.smoothed_rate(n_pos_t, n_lab_t, global_prior, 50.0)

    # user x tag frozen prefix stats
    ut_key = np.unique(np.stack([uid, tag], axis=1), axis=0, return_inverse=True)[1]
    n_lab_ut, n_pos_ut = ctx.frozen_prefix(ut_key, date, y, labeled, horizon)
    rate_ut = ctx.smoothed_rate(n_pos_ut, n_lab_ut, global_prior, 30.0)

    playthrough = np.clip(play / np.maximum(dur, 1.0), 0, 5.0)

    upload_type = ctx.video_attr('upload_type').astype(np.float64)
    music_type = ctx.video_attr('music_type').astype(np.float64)

    item_feats = {
        'n_impressions_v': n_lab_v.astype(np.float64),
        'rate_v': rate_v,
        'click_rate_v': click_rate_v,
        'rate_t': rate_t,
        'rate_ut': rate_ut,
        'duration': dur,
        'playthrough': playthrough,
        'upload_type': upload_type,
        'music_type': music_type,
    }

    # within-user contrast features: rank and z-score per user group
    order = np.argsort(uid, kind='stable')
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(n)
    uid_sorted = uid[order]
    boundaries = np.searchsorted(uid_sorted, uid_sorted, side='left')
    _, group_start, group_count = np.unique(uid_sorted, return_index=True, return_counts=True)
    group_id_sorted = np.repeat(np.arange(len(group_start)), group_count)

    feat_cols = []
    names = []
    for fname, farr in item_feats.items():
        feat_cols.append(farr.astype(np.float32))
        names.append(f'raw_{fname}')

        sorted_vals = farr[order]
        rank_within = np.zeros(n, dtype=np.float64)
        z_within = np.zeros(n, dtype=np.float64)
        idx = 0
        for g in range(len(group_start)):
            start = group_start[g]
            cnt = group_count[g]
            seg = sorted_vals[start:start+cnt]
            if cnt > 1:
                order_seg = np.argsort(np.argsort(seg))
                rank_seg = order_seg / (cnt - 1)
                mean_seg = seg.mean()
                std_seg = seg.std()
                z_seg = (seg - mean_seg) / std_seg if std_seg > 1e-8 else np.zeros(cnt)
            else:
                rank_seg = np.zeros(cnt)
                z_seg = np.zeros(cnt)
            rank_within[start:start+cnt] = rank_seg
            z_within[start:start+cnt] = z_seg
        rank_orig = rank_within[inv_order]
        z_orig = z_within[inv_order]
        feat_cols.append(rank_orig.astype(np.float32))
        names.append(f'rankwu_{fname}')
        feat_cols.append(z_orig.astype(np.float32))
        names.append(f'zwu_{fname}')

    X_C = np.stack(feat_cols, axis=1).astype(np.float32)
    X_C = np.nan_to_num(X_C, nan=0.0, posinf=0.0, neginf=0.0)

    import lightgbm as lgb
    train_mask = np.zeros(n, dtype=bool)
    train_mask[train_idx] = True
    valid_mask = np.zeros(n, dtype=bool)
    valid_mask[valid_idx] = True

    y_bin = ctx.data.y_raw.astype(np.int32)

    seeds = [1, 2, 3]
    preds = np.zeros(n, dtype=np.float64)
    for seed in seeds:
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1,
            'seed': seed,
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'min_data_in_leaf': 50,
        }
        dtrain = lgb.Dataset(X_C[train_mask], label=y_bin[train_mask])
        dvalid = lgb.Dataset(X_C[valid_mask], label=y_bin[valid_mask], reference=dtrain)
        model = lgb.train(
            params, dtrain, num_boost_round=300,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        preds += model.predict(X_C, num_iteration=model.best_iteration)
    member_C = (preds / len(seeds)).astype(np.float64)

    member_A = ctx.baseline_score.astype(np.float64)
    member_B = ctx.refit_score().astype(np.float64)

    def within_user_rank(score):
        s_sorted = score[order]
        rk = np.zeros(n, dtype=np.float64)
        for g in range(len(group_start)):
            start = group_start[g]
            cnt = group_count[g]
            seg = s_sorted[start:start+cnt]
            if cnt > 1:
                rk[start:start+cnt] = np.argsort(np.argsort(seg)) / (cnt - 1)
            else:
                rk[start:start+cnt] = 0.5
        return rk[inv_order]

    rank_A = within_user_rank(member_A)
    rank_B = within_user_rank(member_B)
    rank_C = within_user_rank(member_C)

    best_score = -1.0
    best_weights = (0.34, 0.33, 0.33)
    y_valid = y_bin[valid_idx]

    from itertools import product
    steps = [i / 10.0 for i in range(11)]
    for wa in steps:
        for wb in steps:
            wc = 1.0 - wa - wb
            if wc < -1e-9 or wc > 1.0 + 1e-9:
                continue
            wc = max(0.0, wc)
            fused = wa * rank_A + wb * rank_B + wc * rank_C
            fv = fused[valid_idx]
            if len(np.unique(y_valid)) < 2:
                continue
            order_idx = np.argsort(-fv)
            ys = y_valid[order_idx]
            n_pos = ys.sum()
            n_neg = len(ys) - n_pos
            if n_pos == 0 or n_neg == 0:
                continue
            ranks = np.argsort(np.argsort(fv))
            pos_ranks_sum = ranks[y_valid == 1].sum()
            auc = (pos_ranks_sum - n_pos * (n_pos - 1) / 2.0) / (n_pos * n_neg)
            if auc > best_score:
                best_score = auc
                best_weights = (wa, wb, wc)

    wa, wb, wc = best_weights
    final_score = (wa * rank_A + wb * rank_B + wc * rank_C).astype(np.float32).reshape(-1, 1)

    names_out = ['fused_score']
    train_cfg = {'mode': 'scores'}
    ctx.check(final_score, names_out)
    return final_score, names_out, train_cfg