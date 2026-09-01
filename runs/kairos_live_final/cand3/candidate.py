def build(ctx):
    import numpy as np

    n = ctx.data.n
    user_id = ctx.data.user_id

    def rank_pct_by_user(vals, users):
        order = np.argsort(users, kind='stable')
        u_sorted = users[order]
        v_sorted = vals[order].astype(np.float64)
        n_ = len(v_sorted)
        out = np.empty(n_, dtype=np.float64)
        start = 0
        for i in range(1, n_ + 1):
            if i == n_ or u_sorted[i] != u_sorted[start]:
                seg = v_sorted[start:i]
                m = i - start
                if m == 1:
                    out[start:i] = 0.5
                else:
                    ranks = np.argsort(np.argsort(seg, kind='stable'), kind='stable').astype(np.float64)
                    out[start:i] = ranks / (m - 1)
                start = i
        result = np.empty(n_, dtype=np.float64)
        result[order] = out
        return result

    s_base = np.asarray(ctx.baseline_score, dtype=np.float64)

    try:
        cf_score, hist_count = ctx.cf_score()
        s_cf = np.asarray(cf_score, dtype=np.float64)
        s_cf = np.where(np.asarray(hist_count) > 0, s_cf, np.nanmedian(s_cf))
    except Exception:
        s_cf = np.full(n, 0.5, dtype=np.float64)

    try:
        s_aux = np.asarray(ctx.auxiliary_signal('is_like'), dtype=np.float64)
    except Exception:
        s_aux = np.full(n, 0.5, dtype=np.float64)

    try:
        U, V = ctx.mf_factors(dim=16)
        s_mf = np.sum(np.asarray(U, dtype=np.float64) * np.asarray(V, dtype=np.float64), axis=1)
    except Exception:
        s_mf = np.full(n, 0.5, dtype=np.float64)

    # impute NaNs with median before ranking
    def impute(x):
        x = x.copy()
        mask = ~np.isfinite(x)
        if mask.any():
            med = np.nanmedian(x[~mask]) if (~mask).any() else 0.5
            x[mask] = med
        return x

    s_base = impute(s_base)
    s_cf = impute(s_cf)
    s_aux = impute(s_aux)
    s_mf = impute(s_mf)

    r_base = rank_pct_by_user(s_base, user_id)
    r_cf = rank_pct_by_user(s_cf, user_id)
    r_aux = rank_pct_by_user(s_aux, user_id)
    r_mf = rank_pct_by_user(s_mf, user_id)

    # coarse grid search on validation set for weights, w0 in [0.45,1.0]
    valid_idx = ctx.fold.idx['valid']
    y_valid = ctx.data.y_raw[valid_idx].astype(np.float64)
    u_valid = user_id[valid_idx]

    def gauc_like(scores, y, users):
        # simple within-user auc-like metric approximation via rank correlation with labels
        order = np.argsort(users, kind='stable')
        u_sorted = users[order]
        s_sorted = scores[order]
        y_sorted = y[order]
        total = 0.0
        cnt = 0
        n_ = len(u_sorted)
        start = 0
        for i in range(1, n_ + 1):
            if i == n_ or u_sorted[i] != u_sorted[start]:
                seg_y = y_sorted[start:i]
                seg_s = s_sorted[start:i]
                if seg_y.max() > seg_y.min():
                    pos = seg_s[seg_y == 1]
                    neg = seg_s[seg_y == 0]
                    if len(pos) > 0 and len(neg) > 0:
                        auc = np.mean(pos[:, None] > neg[None, :])
                        total += auc
                        cnt += 1
                start = i
        return total / cnt if cnt > 0 else 0.5

    r_base_v = r_base[valid_idx]
    r_cf_v = r_cf[valid_idx]
    r_aux_v = r_aux[valid_idx]
    r_mf_v = r_mf[valid_idx]

    best_score = -1.0
    best_w = (1.0, 0.0, 0.0, 0.0)
    step = 0.1
    w0_vals = np.arange(0.45, 1.001, step)
    for w0 in w0_vals:
        rem = 1.0 - w0
        if rem < -1e-9:
            continue
        steps2 = max(int(round(rem / step)), 0)
        for a in range(steps2 + 1):
            for b in range(steps2 - a + 1):
                w1 = a * step
                w2 = b * step
                w3 = rem - w1 - w2
                if w3 < -1e-9:
                    continue
                w3 = max(w3, 0.0)
                final_v = w0 * r_base_v + w1 * r_cf_v + w2 * r_aux_v + w3 * r_mf_v
                score = gauc_like(final_v, y_valid, u_valid)
                if score > best_score:
                    best_score = score
                    best_w = (w0, w1, w2, w3)

    w0, w1, w2, w3 = best_w

    final = w0 * r_base + w1 * r_cf + w2 * r_aux + w3 * r_mf

    X = final.reshape(-1, 1).astype(np.float32)
    names = ['rank_fusion_score']

    ctx.check(X, names)

    train_cfg = {'mode': 'scores'}
    return X, names, train_cfg