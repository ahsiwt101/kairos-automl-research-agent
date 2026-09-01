def build(ctx):
    import numpy as np
    n = ctx.data.n
    uid = ctx.data.user_id

    baseline = ctx.refit_score()
    din = ctx.din_score()
    cf, cf_hist = ctx.cf_score()
    U, V = ctx.mf_factors(dim=16)
    mf_dot = np.sum(U * V, axis=1).astype(np.float32)

    aux_names = ['is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward']
    aux = [ctx.auxiliary_signal(a) for a in aux_names]

    signals = [baseline, din, cf, mf_dot] + aux
    sig_names = ['baseline', 'din', 'cf', 'mf_dot'] + aux_names

    # group by user
    order = np.argsort(uid, kind='stable')
    inv = np.empty(n, dtype=np.int64)
    inv[order] = np.arange(n)
    sorted_uid = uid[order]
    # find group boundaries
    change = np.empty(n, dtype=bool)
    change[0] = True
    change[1:] = sorted_uid[1:] != sorted_uid[:-1]
    group_id_sorted = np.cumsum(change) - 1
    group_id = np.empty(n, dtype=np.int64)
    group_id[order] = group_id_sorted
    n_groups = group_id_sorted[-1] + 1

    # group counts
    group_size = np.bincount(group_id, minlength=n_groups).astype(np.float32)
    n_group_row = group_size[group_id]

    feats = []
    names = []

    feats.append(baseline.astype(np.float32)); names.append('baseline_score')
    feats.append(din.astype(np.float32)); names.append('din_score')
    feats.append(cf.astype(np.float32)); names.append('cf_score')
    feats.append(cf_hist.astype(np.float32)); names.append('cf_hist_count')
    feats.append(mf_dot); names.append('mf_dot')
    for a, an in zip(aux, aux_names):
        feats.append(a.astype(np.float32)); names.append('aux_' + an)

    feats.append(n_group_row.astype(np.float32)); names.append('n_group')

    pct_list = []
    for s, sn in zip(signals, sig_names):
        s = s.astype(np.float64)
        # sort within group using group_id then value
        sort_key = np.lexsort((s, group_id))
        ranks = np.empty(n, dtype=np.float64)
        # rank within group: position within sorted group block
        # compute using group boundaries on sort_key order
        gs = group_id[sort_key]
        pos_in_group = np.empty(n, dtype=np.float64)
        # start index of each group in sort_key order
        change2 = np.empty(n, dtype=bool)
        change2[0] = True
        change2[1:] = gs[1:] != gs[:-1]
        start_idx = np.where(change2)[0]
        group_start_for_pos = np.zeros(n, dtype=np.int64)
        # cumulative count within group
        counter = np.arange(n) - np.repeat(start_idx, np.diff(np.append(start_idx, n)))
        pos_in_group[sort_key] = counter
        denom = np.maximum(n_group_row - 1, 1.0)
        pct = pos_in_group / denom
        pct[n_group_row <= 1] = 0.5

        # mean/std within group
        sum_s = np.bincount(group_id, weights=s, minlength=n_groups)
        mean_s = sum_s / np.maximum(group_size, 1)
        sum_s2 = np.bincount(group_id, weights=s * s, minlength=n_groups)
        var_s = sum_s2 / np.maximum(group_size, 1) - mean_s ** 2
        var_s = np.maximum(var_s, 0)
        std_s = np.sqrt(var_s)
        mean_row = mean_s[group_id]
        std_row = std_s[group_id]
        z = (s - mean_row) / (std_row + 1e-6)

        # max within group
        max_s = np.zeros(n_groups, dtype=np.float64)
        np.maximum.at(max_s, group_id, s)
        max_row = max_s[group_id]
        gap = s - max_row

        feats.append(pct.astype(np.float32)); names.append('pct_' + sn)
        feats.append(z.astype(np.float32)); names.append('z_' + sn)
        feats.append(gap.astype(np.float32)); names.append('gap_' + sn)
        pct_list.append(pct)

    pct_arr = np.stack(pct_list, axis=1)
    mean_pct = pct_arr.mean(axis=1)
    disagree = pct_arr.std(axis=1)
    feats.append(mean_pct.astype(np.float32)); names.append('mean_pct')
    feats.append(disagree.astype(np.float32)); names.append('disagree_pct')
    feats.append((pct_list[0] - mean_pct).astype(np.float32)); names.append('baseline_minus_consensus')

    X = np.stack(feats, axis=1).astype(np.float32)
    ctx.check(X, names)
    train_cfg = {'objective': 'binary', 'group': 'user_day', 'hparams': {}}
    return X, names, train_cfg
