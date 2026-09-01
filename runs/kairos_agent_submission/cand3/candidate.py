def build(ctx):
    import numpy as np

    n = ctx.data.n
    user_id = ctx.data.user_id

    baseline = ctx.refit_score()
    din = ctx.din_score()
    cf_score, cf_hist = ctx.cf_score()
    aux_click = ctx.auxiliary_signal('is_click')
    aux_like = ctx.auxiliary_signal('is_like')

    U, V = ctx.mf_factors(dim=16)
    mf_dot = np.sum(U * V, axis=1).astype(np.float32)

    base_cols = {
        'baseline': baseline.astype(np.float32),
        'din': din.astype(np.float32),
        'cf': cf_score.astype(np.float32),
        'aux_click': aux_click.astype(np.float32),
        'aux_like': aux_like.astype(np.float32),
        'mf_dot': mf_dot.astype(np.float32),
    }

    # group by user_id
    order = np.argsort(user_id, kind='stable')
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(n)

    sorted_uid = user_id[order]
    # boundaries of groups
    change = np.empty(n, dtype=bool)
    change[0] = True
    change[1:] = sorted_uid[1:] != sorted_uid[:-1]
    group_start = np.flatnonzero(change)
    group_end = np.append(group_start[1:], n)

    group_id = np.cumsum(change) - 1  # group index per sorted position
    n_groups = group_start.size

    group_size_sorted = (group_end - group_start)[group_id]

    feats = []
    names = []

    for cname, col in base_cols.items():
        sorted_col = col[order]

        # rank within group
        rank_sorted = np.empty(n, dtype=np.float32)
        mean_sorted = np.empty(n, dtype=np.float32)
        std_sorted = np.empty(n, dtype=np.float32)
        max_sorted = np.empty(n, dtype=np.float32)

        for g in range(n_groups):
            s, e = group_start[g], group_end[g]
            seg = sorted_col[s:e]
            order_seg = np.argsort(seg, kind='stable')
            ranks = np.empty(len(seg), dtype=np.float32)
            ranks[order_seg] = np.arange(len(seg), dtype=np.float32)
            denom = max(len(seg) - 1, 1)
            rank_sorted[s:e] = ranks / denom
            m = seg.mean()
            sd = seg.std()
            mx = seg.max()
            mean_sorted[s:e] = m
            std_sorted[s:e] = sd
            max_sorted[s:e] = mx

        z_sorted = (sorted_col - mean_sorted) / (std_sorted + 1e-6)
        centered_sorted = sorted_col - mean_sorted
        maxdiff_sorted = sorted_col - max_sorted

        rank_col = np.empty(n, dtype=np.float32)
        z_col = np.empty(n, dtype=np.float32)
        centered_col = np.empty(n, dtype=np.float32)
        maxdiff_col = np.empty(n, dtype=np.float32)

        rank_col[order] = rank_sorted
        z_col[order] = z_sorted
        centered_col[order] = centered_sorted
        maxdiff_col[order] = maxdiff_sorted

        feats.append(col)
        names.append(cname)
        feats.append(rank_col)
        names.append(cname + '_rank')
        feats.append(z_col)
        names.append(cname + '_z')
        feats.append(centered_col)
        names.append(cname + '_centered')
        feats.append(maxdiff_col)
        names.append(cname + '_maxdiff')

    group_size_col = np.empty(n, dtype=np.float32)
    group_size_col[order] = np.log1p(group_size_sorted).astype(np.float32)
    feats.append(group_size_col)
    names.append('group_size_log')

    X = np.stack(feats, axis=1).astype(np.float32)

    ctx.check(X, names)

    train_cfg = {
        'objective': 'binary',
        'group': 'user',
        'hparams': {}
    }

    return X, names, train_cfg
