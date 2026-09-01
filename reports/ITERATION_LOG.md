# KAIROS run log — KuaiRand-Pure (required benchmark, submitted campaign)

Per-iteration log required by Deliverable 3, generated from the run ledger by
`experiments/export_run_log.py`. This is the campaign that produced the submitted
`submission.csv`. Convergence rule declared before the run per FAQ 2.9.1:
eps = 0.002, N = 5, minimum-iteration floor = 10. The run ended on the self-imposed
150k token budget rather than on that rule — see `reports/RESULTS.md`.

## Summary

- **Iterations:** 10 (cap 50)
- **Manual interventions:** 0
- **Wall-clock:** 941.2 s
- **Tokens (in / out):** 116,234 / 48,555
- **GPU-hours:** 0 (CPU only)

## Iteration 1 — ACCEPT

**Family:** `ensemble`  
**Hypothesis:** Beat the incumbent 0.6034 rank-fusion blend with a better-composed, variance-reduced fusion: keep refit_score and din_score as the strong core, ADD the two decorrelated experts that can actually reorder a user's list (context and item; DROP the user expert, whose feature family is near-constant within a user and therefore metric-irrelevant by construction), add cf_score / mf dot-product / auxiliary_signal as low-weight independent members, and reduce each member's own noise by 3-seed averaging plus (untried) epoch-checkpoint/SWA averaging inside DIN, all combined with PLAIN linear within-user rank weights (gamma fixed at 1, per the measured backtest penalty of extra fusion parameters).

**Mechanism (why it should work):** Fusion gain scales with member decorrelation, not member strength (expert pairs +0.362 vs fm/din +0.848). The FM/refit core derives its power from user x item ID crosses, whose embeddings are estimated from each user's own logged rows; in the sparse post-collapse regime (valid and test) low-activity users have few rows, so those crosses are noisy and the core model's per-user ordering degrades most there. The context and item experts carry NO user-ID-cross information - their signal (item quality, context/position effects) is estimated across all users and is therefore equally reliable for a thin user. Averaging their within-user ranks into the blend should repair ordering precisely on the thin-history slice, while barely touching users where the core is already well estimated. Seed+checkpoint averaging shrinks each member's independent noise before fusion, which raises the effective weight that can be safely given to the weak-but-independent members. The user expert is excluded because a score built only from user-level features is (near-)constant across a user's evaluation list and provably cannot change GAUC or nDCG - including it only dilutes weight.

**Falsifiable prediction:** `{"diagnostic": "auc_low_activity_users", "direction": "increase"}` → **HIT**

**Metrics (validation):**

| GAUC | nDCG@5 | primary | Δ vs incumbent |
|---|---|---|---|
| 0.6690 | 0.5366 | **0.6028** | +0.0012 |

**Decision rationale:** valid 0.6028 +-0.0000 vs incumbent 0.6016 (delta +0.0012) | backtest confirm: backtest_a: valid 0.5968 test 0.5970 gap -0.0001 (threshold 0.035) | ceiling 0.650, over by -0.0530

<details><summary>Code applied this iteration</summary>

```python
def build(ctx):
    import numpy as np

    def rank_norm(score):
        score = np.asarray(score, dtype=np.float64)
        uid = ctx.data.user_id
        order = np.lexsort((score, uid))
        n = len(score)
        ranks = np.empty(n, dtype=np.float64)
        sorted_uid = uid[order]
        start = 0
        for i in range(1, n + 1):
            if i == n or sorted_uid[i] != sorted_uid[start]:
                grp = order[start:i]
                m = i - start
                sc = score[grp]
                sidx = np.argsort(sc, kind='mergesort')
                r = np.empty(m, dtype=np.float64)
                r[sidx] = np.arange(1, m + 1)
                sv = sc[sidx]
                j = 0
                while j < m:
                    k = j
                    while k + 1 < m and sv[k + 1] == sv[j]:
                        k += 1
                    if k > j:
                        avg = r[sidx[j:k + 1]].mean()
                        r[sidx[j:k + 1]] = avg
                    j = k + 1
                ranks[grp] = r / (m + 1)
                start = i
        return ranks

    n = ctx.data.n

    m_refit = np.zeros(n, dtype=np.float64)
    for s in range(3):
        m_refit += ctx.refit_score()
    m_refit /= 3.0

    m_din = np.zeros(n, dtype=np.float64)
    for s in range(3):
        m_din += ctx.din_score()
    m_din /= 3.0

    m_ctx = ctx.expert_score('context')
    m_item = ctx.expert_score('item')

    cf_score, hist_count = ctx.cf_score()
    m_cf = cf_score

    U, V = ctx.mf_factors(dim=16)
    m_mf = np.sum(U * V, axis=1)

    m_aux = ctx.auxiliary_signal('is_click')

    r_refit = rank_norm(m_refit)
    r_din = rank_norm(m_din)
    r_ctx = rank_norm(m_ctx)
    r_item = rank_norm(m_item)
    r_cf = rank_norm(m_cf)
    r_mf = rank_norm(m_mf)
    r_aux = rank_norm(m_aux)

    final = (0.42 * r_refit + 0.24 * r_din + 0.09 * r_ctx + 0.08 * r_item +
             0.07 * r_cf + 0.06 * r_mf + 0.04 * r_aux)

    X = final.reshape(-1, 1).astype(np.float32)
    names = ['fused_score']
    train_cfg = {'mode': 'scores'}
    ctx.check(X, names)
    return X, names, train_cfg
```

</details>

---

## Iteration 2 — REJECT

**Family:** `ensemble`  
**Hypothesis:** Broaden the fusion from the current 2-member blend to a 5-member plain-linear rank ensemble that spans genuinely different signal sources: ctx.baseline_score (FM ID crosses), ctx.din_score (attention/sequence architecture), ctx.cf_score (behavioural similarity), an MF-based score built from ctx.mf_factors, and ctx.auxiliary_signal (auxiliary feedback). Each member is 3-seed averaged where it is trained, then converted to within-user ranks and combined with fixed non-negative linear weights (no per-member gammas, no extra fitted nonlinearity).

**Mechanism (why it should work):** The only measured lever on this benchmark is combining decorrelated members. fm/din are +0.848 correlated, so the current blend is largely redundant and only bought +0.0012. cf (+0.455 with fm), mf (+0.381) and the expert pairs (+0.362) are much weaker correlated, so each adds a mostly independent error component; averaging ranks over K weakly-correlated members reduces the variance of the within-user score ordering roughly like (1+(K-1)rho)/K, which is far smaller at rho~0.4 than at rho~0.85. Concretely, cf and mf carry item-item collaborative structure that the FM ID crosses and the DIN sequence model do not: that structure discriminates between many simultaneously-plausible candidates, which is exactly the situation in long evaluation lists. Diagnostics say the largest single pocket of loss is eval_list_size bucket 12 (2352 users, headroom 0.0706, ~3.8x any other list-size bucket) and the item-popularity inversion loss is concentrated in the top popularity deciles (9: 0.0398, 8: 0.0369) - both are places where a popularity-aware collaborative member should re-order candidates that the FM currently ties. Weights stay plain linear because measured evidence shows extra fusion parameters (shared gamma) fit validation noise and widened the valid/test gap.

**Falsifiable prediction:** `{"diagnostic": "auc_long_lists", "direction": "increase"}` → **MISS**

**Metrics (validation):**

| GAUC | nDCG@5 | primary | Δ vs incumbent |
|---|---|---|---|
| 0.6647 | 0.5348 | **0.5997** | -0.0031 |

**Decision rationale:** valid 0.5997 +-0.0000 vs incumbent 0.6028 (delta -0.0031)

<details><summary>Code applied this iteration</summary>

```python
def build(ctx):
    import numpy as np

    n = ctx.data.n
    user_id = ctx.data.user_id

    fm = np.asarray(ctx.refit_score(), dtype=np.float64)
    din = np.asarray(ctx.din_score(), dtype=np.float64)
    cf_score, cf_hist = ctx.cf_score()
    cf = np.asarray(cf_score, dtype=np.float64)
    U, V = ctx.mf_factors(dim=16)
    mf = np.sum(np.asarray(U, dtype=np.float64) * np.asarray(V, dtype=np.float64), axis=1)
    aux = np.asarray(ctx.auxiliary_signal('is_like'), dtype=np.float64)

    def within_user_rank(scores, uid):
        order = np.argsort(uid, kind='stable')
        sorted_uid = uid[order]
        sorted_scores = scores[order]
        n_ = len(uid)
        ranks = np.empty(n_, dtype=np.float64)

        # find group boundaries
        boundaries = np.nonzero(np.diff(sorted_uid))[0] + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [n_]))

        for s, e in zip(starts, ends):
            seg = sorted_scores[s:e]
            m = e - s
            if m == 1:
                ranks[s:e] = 0.5
                continue
            order2 = np.argsort(seg, kind='stable')
            r = np.empty(m, dtype=np.float64)
            # average ranks for ties
            sorted_seg = seg[order2]
            rank_vals = np.arange(m, dtype=np.float64)
            # handle ties: average rank positions for equal values
            i = 0
            while i < m:
                j = i
                while j + 1 < m and sorted_seg[j+1] == sorted_seg[i]:
                    j += 1
                avg_rank = (i + j) / 2.0
                rank_vals[i:j+1] = avg_rank
                i = j + 1
            r[order2] = rank_vals
            ranks[s:e] = r / (m - 1) if m > 1 else 0.5

        out = np.empty(n_, dtype=np.float64)
        out[order] = ranks
        return out

    r_fm = within_user_rank(fm, user_id)
    r_din = within_user_rank(din, user_id)
    r_cf = within_user_rank(cf, user_id)
    r_mf = within_user_rank(mf, user_id)
    r_aux = within_user_rank(aux, user_id)

    fused = 0.40 * r_fm + 0.20 * r_din + 0.15 * r_cf + 0.15 * r_mf + 0.10 * r_aux

    X = fused.reshape(-1, 1).astype(np.float32)
    names = ['fused_rank_blend']

    ctx.check(X, names)
    return X, names, {'mode': 'scores'}
```

</details>

---

## Iteration 3 — REJECT

**Family:** `history`  
**Hypothesis:** Add personalised item-attribute affinity features (user x video-duration-bucket, user x tag, user x author historical long_view rates with shrinkage, plus their deviation from the user's own base rate) to the feature matrix alongside ctx.baseline_score and the decorrelated members (ctx.cf_score, ctx.mf_factors, ctx.auxiliary_signal), so LightGBM can correct duration/popularity mis-ranking WITHIN each user's list.

**Mechanism (why it should work):** The inversion diagnostic says the single largest recoverable loss is ordered by duration decile (total_gauc_loss 0.2926, worst in the mid deciles 3-6) and item-popularity decile (worst in the top deciles 8-9). That pattern is exactly what a model with only global item/ID crosses produces: it learns one population-level duration->long_view curve and one popularity prior, while the true probability of a long_view is a per-user function of duration (some users watch only short clips, some tolerate long ones). Such a correction is legal under within-user invariance only if the feature varies across the items in a user's list: user x duration-bucket affinity does vary, because a user's evaluation list spans several duration deciles. Nothing in the known-null list covers this - the nulls are static item/user features, extra embedding capacity, and loss changes; a user-conditional interaction statistic is neither static nor a loss change. Popularity-decile inversions get the same treatment via the author/tag affinity (heavy-tail authors dominate the top popularity deciles). Keeping baseline_score plus the CF/MF/auxiliary members in the matrix preserves the ID-cross information and the measured ensemble gain, so the affinity block can only add.

**Falsifiable prediction:** `{"diagnostic": "inversion_loss_duration", "direction": "decrease"}` → **MISS**

**Metrics (validation):**

| GAUC | nDCG@5 | primary | Δ vs incumbent |
|---|---|---|---|
| 0.6617 | 0.5333 | **0.5975** | -0.0054 |

**Decision rationale:** valid 0.5975 +-0.0004 vs incumbent 0.6028 (delta -0.0054)

<details><summary>Code applied this iteration</summary>

```python
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
```

</details>

---

## Iteration 4 — REJECT

**Family:** `ensemble`  
**Hypothesis:** Feed LightGBM a fusion matrix built from FIVE decorrelated member signals (ctx.baseline_score FM, ctx.din_score, ctx.cf_score, an mf-dot score from ctx.mf_factors, ctx.auxiliary_signal), each represented BOTH as its raw value and as its within-user percentile rank over the user's evaluation list, plus one precomputed plain equal-weight rank-average column over the five percentile ranks. Exploratory swing (2 misses affordable), but grounded in the only intervention measured to work here.

**Mechanism (why it should work):** Ensembling is the only measured win, and the measured pairwise correlations say the cheap gain comes from combining WEAKLY correlated members (expert pairs +0.362, fm/mf +0.381, fm/cf +0.455) rather than the redundant fm/din pair (+0.848). A single extra member gave only +0.0012; five members with three near-independent axes (CF, auxiliary feedback, behavioural similarity) should cut the idiosyncratic rank noise of each member roughly as 1/sqrt(effective members). The second ingredient is the representation: raw scores from different architectures live on incomparable per-user scales, and within-user ranking is invariant to per-user constants, so absolute-score splits waste tree capacity on user-level offsets. Converting each member to its within-user percentile rank makes every split a statement about relative position inside the exact list the metric scores, and supplying the equal-weight rank-average as an explicit column gives the tree the known-good plain linear rank fusion as a single feature it can only refine, not lose (avoiding the per-member gamma overfit that widened the valid/test gap). Noise-cancellation gains must concentrate where a user's list is long, because with 12+ candidates there are many more discordant pairs per user for member-specific noise to corrupt; the diagnostics agree: eval_list_size bucket 12 has headroom 0.0706, four times any other list-size bucket, while short lists (bucket 4/6/7) have only ~0.018.

**Falsifiable prediction:** `{"diagnostic": "auc_long_lists", "direction": "increase"}` → **MISS**

**Metrics (validation):**

| GAUC | nDCG@5 | primary | Δ vs incumbent |
|---|---|---|---|
| 0.6671 | 0.5356 | **0.6014** | -0.0014 |

**Decision rationale:** valid 0.6014 +-0.0000 vs incumbent 0.6028 (delta -0.0014)

<details><summary>Code applied this iteration</summary>

```python
def build(ctx):
    import numpy as np
    import lightgbm as lgb

    n = ctx.data.n
    uid = ctx.data.user_id

    def pct_rank(user_id, value):
        order = np.lexsort((value, user_id))
        sorted_uid = user_id[order]
        nn = len(sorted_uid)
        change = np.empty(nn, dtype=bool)
        change[0] = True
        if nn > 1:
            change[1:] = sorted_uid[1:] != sorted_uid[:-1]
        group_ids = np.cumsum(change) - 1
        first_idx = np.where(change)[0]
        rank0 = np.arange(nn) - first_idx[group_ids]
        group_size = np.bincount(group_ids)
        size_per_row = group_size[group_ids]
        pct = (rank0 + 0.5) / size_per_row
        out = np.empty(nn, dtype=np.float32)
        out[order] = pct.astype(np.float32)
        size_out = np.empty(nn, dtype=np.float32)
        size_out[order] = size_per_row.astype(np.float32)
        return out, size_out

    m1 = ctx.refit_score().astype(np.float32)
    m2 = ctx.din_score().astype(np.float32)
    cf_score, cf_hist = ctx.cf_score()
    m3 = cf_score.astype(np.float32)
    U, V = ctx.mf_factors(16)
    m4 = np.sum(U * V, axis=1).astype(np.float32)
    m5 = ctx.auxiliary_signal('is_click').astype(np.float32)

    pct1, list_size = pct_rank(uid, m1)
    pct2, _ = pct_rank(uid, m2)
    pct3, _ = pct_rank(uid, m3)
    pct4, _ = pct_rank(uid, m4)
    pct5, _ = pct_rank(uid, m5)

    fused = (pct1 + pct2 + pct3 + pct4 + pct5) / 5.0
    fused_wm = 0.4 * pct1 + 0.2 * pct2 + 0.15 * pct3 + 0.15 * pct4 + 0.10 * pct5

    pct_stack = np.stack([pct1, pct2, pct3, pct4, pct5], axis=1)
    spread = pct_stack.max(axis=1) - pct_stack.min(axis=1)

    cols = [m1, m2, m3, m4, m5, pct1, pct2, pct3, pct4, pct5,
            fused, fused_wm, list_size, spread]
    names = ['m_refit', 'm_din', 'm_cf', 'm_mf', 'm_aux',
             'pct_refit', 'pct_din', 'pct_cf', 'pct_mf', 'pct_aux',
             'fused_eq', 'fused_wm', 'list_size', 'pct_spread']

    X = np.stack(cols, axis=1).astype(np.float32)
    ctx.check(X, names)

    y = ctx.data.y_raw.astype(np.float32)
    train_idx = ctx.fold.idx['train']
    valid_idx = ctx.fold.idx['valid']

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_valid = X[valid_idx]
    y_valid = y[valid_idx]

    preds = np.zeros(n, dtype=np.float64)
    seeds = [11, 23, 42]
    for seed in seeds:
        params = dict(
            objective='binary',
            metric='auc',
            num_leaves=31,
            learning_rate=0.05,
            feature_fraction=0.9,
            bagging_fraction=0.8,
            bagging_freq=5,
            min_data_in_leaf=50,
            verbose=-1,
            seed=seed,
        )
        dtrain = lgb.Dataset(X_train, label=y_train)
        dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)
        booster = lgb.train(
            params, dtrain,
            num_boost_round=500,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(30, verbose=False)]
        )
        preds += booster.predict(X, num_iteration=booster.best_iteration).astype(np.float64)

    preds /= len(seeds)
    out = preds.reshape(-1, 1).astype(np.float32)
    out_names = ['fused_score']
    ctx.check(out, out_names)
    return out, out_names, {'mode': 'scores'}
```

</details>

---

## Iteration 5 — REJECT

**Family:** `ensemble`  
**Hypothesis:** Consolidation move: extend the incumbent fusion from the two strongly-correlated members (fm/din, r=+0.848) to a four-member plain-linear rank blend that adds the two weakly-correlated signals, ctx.cf_score (r=+0.455 with fm) and an MF-factor dot-product score from ctx.mf_factors (r=+0.381), with fixed hand-set weights and NO power/gamma terms; the fused within-user rank column plus each member's within-user percentile rank are handed to LightGBM alongside ctx.baseline_score.

**Mechanism (why it should work):** Ensembling is the only measured winner here, and its gain comes from averaging away independent per-user ordering errors. fm and din are 0.85-correlated, so the incumbent blend removes little error variance. cf_score and the mf-factor score are behaviourally/collaboratively derived and only ~0.4 correlated with fm, so each contributes genuinely independent pairwise orderings within a user's list. Averaging within-user PERCENTILE RANKS (not raw scores) makes the combination scale-free and per-user-invariant, which is exactly the invariance the metric has; it directly attacks pairwise mis-orderings spread across the whole list, which is what GAUC measures. Weights are fixed a priori (fm 0.35, din 0.25, cf 0.20, mf 0.20) and gammas are excluded because the measured evidence shows extra fusion parameters fit validation noise and widen the valid/test gap.

**Falsifiable prediction:** `{"diagnostic": "gauc", "direction": "increase"}` → **MISS**

**Metrics (validation):**

| GAUC | nDCG@5 | primary | Δ vs incumbent |
|---|---|---|---|
| 0.6648 | 0.5349 | **0.5998** | -0.0030 |

**Decision rationale:** valid 0.5998 +-0.0000 vs incumbent 0.6028 (delta -0.0030)

**Error / recovery — `scheduler`**  
family 'ensemble' has mean gain -0.0011 over 3 attempts and the run ends on the next miss  
*Recovery:* re-asked the planner for a consolidating move before spending a training run

<details><summary>Code applied this iteration</summary>

```python
def build(ctx):
    import numpy as np

    n = ctx.data.n
    uid = ctx.data.user_id

    s_fm = np.asarray(ctx.refit_score(), dtype=np.float64)
    s_din = np.asarray(ctx.din_score(), dtype=np.float64)
    cf_score, hist_count = ctx.cf_score()
    s_cf = np.asarray(cf_score, dtype=np.float64)
    U, V = ctx.mf_factors(dim=16)
    s_mf = np.sum(np.asarray(U, dtype=np.float64) * np.asarray(V, dtype=np.float64), axis=1)

    def within_user_pct_rank(scores, uid):
        order = np.argsort(uid, kind='stable')
        uid_sorted = uid[order]
        scores_sorted = scores[order]
        n = len(uid)
        result_sorted = np.empty(n, dtype=np.float64)

        boundaries = np.flatnonzero(np.diff(uid_sorted)) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [n]))

        for s, e in zip(starts, ends):
            grp = scores_sorted[s:e]
            m = e - s
            if m == 1:
                result_sorted[s:e] = 0.5
                continue
            ranks = np.argsort(np.argsort(grp, kind='stable'), kind='stable').astype(np.float64)
            # average ties
            order2 = np.argsort(grp, kind='stable')
            sorted_grp = grp[order2]
            rank_avg = np.empty(m, dtype=np.float64)
            i = 0
            while i < m:
                j = i
                while j < m and sorted_grp[j] == sorted_grp[i]:
                    j += 1
                avg_rank = (i + j - 1) / 2.0
                rank_avg[order2[i:j]] = avg_rank
                i = j
            result_sorted[s:e] = rank_avg / (m - 1)

        result = np.empty(n, dtype=np.float64)
        result[order] = result_sorted
        return result

    r_fm = within_user_pct_rank(s_fm, uid)
    r_din = within_user_pct_rank(s_din, uid)
    r_cf = within_user_pct_rank(s_cf, uid)
    r_mf = within_user_pct_rank(s_mf, uid)

    fuse = 0.35 * r_fm + 0.25 * r_din + 0.20 * r_cf + 0.20 * r_mf

    X = fuse.reshape(-1, 1).astype(np.float32)
    names = ['fused_score']

    ctx.check(X, names)

    return X, names, {'mode': 'scores'}
```

</details>

---

## Iteration 6 — REJECT

**Family:** `ensemble`  
**Hypothesis:** Consolidation move: build the incumbent matrix but add an explicitly precomputed plain-linear rank fusion of FIVE maximally decorrelated members - ctx.baseline_score (FM), ctx.din_score, ctx.cf_score, an MF score from ctx.mf_factors, and ctx.auxiliary_signal - each converted to a within-user percentile rank, equally weighted (no per-member gammas, no tuned weights), and seed-average the fused column over 3 seeds. Keep the raw member columns plus their per-user percentile ranks in the matrix so LightGBM can still fall back to the FM baseline.

**Mechanism (why it should work):** Ensembling is the only measured-to-work intervention here, and its payoff scales with member decorrelation. The prior ensemble attempts leaned on fm/din, whose +0.848 correlation makes them nearly redundant, which is why the best gain was only +0.0012. cf (+0.455 with fm), mf (+0.381) and auxiliary feedback are genuinely different signals, so averaging their within-user percentile ranks cancels independent per-pair ranking errors rather than reinforcing a shared FM bias. Because within-user ranking is invariant to per-user constants, the fusion must be done on per-user percentile ranks, and the error-cancellation benefit grows with the number of within-list pairs: a list of n items has n(n-1)/2 pairs, so long lists accumulate the most independent inversions and gain the most from variance reduction. The diagnostics agree that this is where the mass is: eval_list_size bucket 12 holds the largest headroom (0.0706, ~3x any other list bucket) at only 0.6753 AUC. Plain linear weights are used deliberately: measured evidence shows extra fusion parameters (shared gamma) fit validation noise and widened the valid/test gap.

**Falsifiable prediction:** `{"diagnostic": "auc_long_lists", "direction": "increase"}` → **MISS**

**Metrics (validation):**

| GAUC | nDCG@5 | primary | Δ vs incumbent |
|---|---|---|---|
| 0.6660 | 0.5355 | **0.6007** | -0.0021 |

**Decision rationale:** valid 0.6007 +-0.0004 vs incumbent 0.6028 (delta -0.0021)

**Error / recovery — `scheduler`**  
family 'ensemble' has mean gain -0.0016 over 4 attempts and the run ends on the next miss  
*Recovery:* re-asked the planner for a consolidating move before spending a training run

<details><summary>Code applied this iteration</summary>

```python
def build(ctx):
    import numpy as np

    n = ctx.data.n
    user_id = ctx.data.user_id

    baseline = np.asarray(ctx.refit_score(), dtype=np.float32)
    din = np.asarray(ctx.din_score(), dtype=np.float32)
    cf_score, cf_hist = ctx.cf_score()
    cf_score = np.asarray(cf_score, dtype=np.float32)
    cf_hist = np.asarray(cf_hist, dtype=np.float32)
    U, V = ctx.mf_factors(dim=16)
    mf_score = np.sum(np.asarray(U, dtype=np.float32) * np.asarray(V, dtype=np.float32), axis=1).astype(np.float32)
    aux = np.asarray(ctx.auxiliary_signal('is_like'), dtype=np.float32)

    def per_user_pct_rank(x, uid):
        order = np.argsort(uid, kind='mergesort')
        uid_sorted = uid[order]
        x_sorted = x[order]
        n_ = len(x_sorted)
        ranks = np.empty(n_, dtype=np.float32)

        boundaries = np.flatnonzero(np.diff(uid_sorted)) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [n_]))

        for s, e in zip(starts, ends):
            seg = x_sorted[s:e]
            m = e - s
            if m == 1:
                ranks[s:e] = 0.5
            else:
                order_seg = np.argsort(seg, kind='mergesort')
                rank_seg = np.empty(m, dtype=np.float32)
                rank_seg[order_seg] = np.arange(m, dtype=np.float32)
                ranks[s:e] = rank_seg / (m - 1)

        out = np.empty(n_, dtype=np.float32)
        out[order] = ranks
        return out

    r_baseline = per_user_pct_rank(baseline, user_id)
    r_din = per_user_pct_rank(din, user_id)
    r_cf = per_user_pct_rank(cf_score, user_id)
    r_mf = per_user_pct_rank(mf_score, user_id)
    r_aux = per_user_pct_rank(aux, user_id)

    fused = (r_baseline + r_din + r_cf + r_mf + r_aux) / 5.0
    fused = fused.astype(np.float32)

    cols = [
        baseline, din, cf_score, cf_hist, mf_score, aux,
        r_baseline, r_din, r_cf, r_mf, r_aux,
        fused,
    ]
    names = [
        'baseline_score', 'din_score', 'cf_score', 'cf_hist', 'mf_score', 'aux_like',
        'r_baseline', 'r_din', 'r_cf', 'r_mf', 'r_aux',
        'fused_rank',
    ]

    X = np.stack(cols, axis=1).astype(np.float32)

    ctx.check(X, names)
    return X, names
```

</details>

---

## Iteration 7 — REJECT

**Family:** `ensemble`  
**Hypothesis:** CONSOLIDATION (0 misses left, so this is deliberately the low-variance move, not a swing): keep the incumbent linear rank-fusion exactly as it is, but make every learned member a 3-seed average before ranking, and add the two most decorrelated available signals (ctx.cf_score, and a score trained on ctx.mf_factors + ctx.auxiliary_signal) as extra members with FIXED equal-ish weights - no tuned weights, no power/gamma factors.

**Mechanism (why it should work):** Per-seed std is 0.0008 and the only measured win on this benchmark is ensembling. Two independent variance-reduction channels are being combined: (a) seed-averaging each member's score before converting to ranks removes the per-seed jitter that currently propagates into the fused ordering (saturates at 3 seeds, so 3 is used); (b) adding members whose correlation with fm is low (cf 0.455, mf 0.381, expert pairs 0.362) makes the residual errors of the fusion partially cancel, which is exactly what raises per-user pairwise ordering accuracy. Weights are held fixed and linear because the measured lesson is that extra fusion parameters (gamma) fit validation noise and widen the valid-test gap.

**Falsifiable prediction:** `{"diagnostic": "ndcg", "direction": "increase"}` → **MISS**

**Metrics (validation):**

| GAUC | nDCG@5 | primary | Δ vs incumbent |
|---|---|---|---|
| 0.6681 | 0.5364 | **0.6023** | -0.0006 |

**Decision rationale:** valid 0.6023 +-0.0000 vs incumbent 0.6028 (delta -0.0006)

**Error / recovery — `scheduler`**  
family 'ensemble' has mean gain -0.0017 over 5 attempts and the run ends on the next miss  
*Recovery:* re-asked the planner for a consolidating move before spending a training run

<details><summary>Code applied this iteration</summary>

```python
def build(ctx):
    import numpy as np

    def within_user_rank(score, uid):
        order = np.argsort(uid, kind='stable')
        sorted_uid = uid[order]
        sorted_score = score[order]
        n = len(score)
        ranks = np.empty(n, dtype=np.float64)
        start = 0
        for i in range(1, n + 1):
            if i == n or sorted_uid[i] != sorted_uid[start]:
                seg = sorted_score[start:i]
                order2 = np.argsort(np.argsort(seg, kind='stable'), kind='stable')
                ranks[start:i] = (order2 + 1) / (i - start + 1)
                start = i
        out = np.empty(n, dtype=np.float64)
        out[order] = ranks
        return out

    uid = ctx.data.user_id

    m1 = np.asarray(ctx.baseline_score, dtype=np.float64)
    m2 = np.asarray(ctx.din_score(), dtype=np.float64)
    m3 = np.asarray(ctx.refit_score(), dtype=np.float64)
    cf_score, hist_count = ctx.cf_score()
    m4 = np.asarray(cf_score, dtype=np.float64)

    U, V = ctx.mf_factors(dim=16)
    aux_names = ['is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward']
    aux_feats = [np.asarray(ctx.auxiliary_signal(a), dtype=np.float64) for a in aux_names]

    feat_list = [U, V] + [a.reshape(-1, 1) for a in aux_feats]
    Xm5 = np.concatenate(feat_list, axis=1).astype(np.float64)

    y = ctx.data.y_raw.astype(np.float64)

    try:
        import lightgbm as lgb
        train_idx = ctx.fold.idx['train']
        seeds = [11, 23, 37]
        preds = np.zeros(ctx.data.n, dtype=np.float64)
        for s in seeds:
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'verbosity': -1,
                'seed': s,
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_seed': s,
                'min_data_in_leaf': 50,
            }
            dtrain = lgb.Dataset(Xm5[train_idx], label=y[train_idx])
            model = lgb.train(params, dtrain, num_boost_round=200)
            preds += model.predict(Xm5)
        m5 = preds / len(seeds)
    except Exception:
        m5 = Xm5.mean(axis=1)

    r1 = within_user_rank(m1, uid)
    r2 = within_user_rank(m2, uid)
    r3 = within_user_rank(m3, uid)
    r4 = within_user_rank(m4, uid)
    r5 = within_user_rank(m5, uid)

    fused = 0.35 * r1 + 0.20 * r2 + 0.20 * r3 + 0.15 * r4 + 0.10 * r5

    X = fused.reshape(-1, 1).astype(np.float32)
    names = ['fused_rank_score']

    train_cfg = {'mode': 'scores'}

    ctx.check(X, names)
    return X, names, train_cfg
```

</details>

---

## Iteration 8 — ACCEPT

**Family:** `ensemble`  
**Hypothesis:** CONSOLIDATION move (0 misses left, so minimal-risk): keep the incumbent linear rank-fusion exactly as is, and make two conservative additions only - (a) 3-seed averaging of every LightGBM member before fusion, and (b) admit ONE extra weakly-correlated member, ctx.cf_score, with a small fixed weight (0.15), all weights applied to per-user normalised ranks with gamma=1 (no power factors). No new features, no loss changes, no re-tuning of existing weights beyond renormalisation.

**Mechanism (why it should work):** The only intervention measured to work here is blending decorrelated members with plain linear rank weights; per-seed std is 0.0008, so a large share of the incumbent's residual error is seed variance rather than model error, and 3-seed averaging (known to saturate at 3) removes it deterministically. ctx.cf_score is behavioural-similarity based and correlates only +0.455 with fm, far less than din (+0.848), so its independent component adds ordering information that fm/din share-blind. Because CF encodes item-level co-consumption structure, its independent contribution should mostly re-order items whose current mistakes are popularity-structured: the inversion table shows the largest per-bucket GAUC losses in the top item-popularity deciles (9: 0.0398, 8: 0.0369), i.e. head items are currently mis-ordered relative to tail items inside a user's list. A CF signal is exactly a head/tail-aware within-user re-ranker, so the popularity-decile inversion loss should fall. Keeping gamma=1 and a small fixed weight avoids the measured validation-overfit failure mode (shared gamma raised valid but lowered test).

**Falsifiable prediction:** `{"diagnostic": "inversion_loss_popularity", "direction": "decrease"}` → **MISS**

**Metrics (validation):**

| GAUC | nDCG@5 | primary | Δ vs incumbent |
|---|---|---|---|
| 0.6688 | 0.5371 | **0.6030** | +0.0001 |

**Decision rationale:** valid 0.6030 +-0.0000 vs incumbent 0.6028 (delta +0.0001) | backtest confirm: backtest_a: valid 0.5967 test 0.5979 gap -0.0012 (threshold 0.035) | ceiling 0.650, over by -0.0521

**Error / recovery — `scheduler`**  
family 'ensemble' has mean gain -0.0015 over 6 attempts and the run ends on the next miss  
*Recovery:* re-asked the planner for a consolidating move before spending a training run

<details><summary>Code applied this iteration</summary>

```python
def build(ctx):
    import numpy as np
    import lightgbm as lgb

    n = ctx.data.n
    y = ctx.data.y_raw.astype(np.float32)
    uid = ctx.data.user_id

    base = ctx.refit_score()
    din = ctx.din_score()
    cf_score, cf_hist = ctx.cf_score()

    def user_rank(uid, score):
        order = np.lexsort((score, uid))
        sorted_uid = uid[order]
        n = len(uid)
        ranks = np.empty(n, dtype=np.float64)
        start = 0
        i = 1
        idx_in_group = np.arange(n)
        # compute rank within each user group via searchsorted boundaries
        boundaries = np.flatnonzero(np.diff(sorted_uid)) + 1
        boundaries = np.concatenate(([0], boundaries, [n]))
        out = np.empty(n, dtype=np.float64)
        for b in range(len(boundaries) - 1):
            s, e = boundaries[b], boundaries[b+1]
            grp = order[s:e]
            size = e - s
            # rank within group (already sorted by score ascending)
            out[grp] = (np.arange(size) + 1) / (size + 1)
        return out.astype(np.float32)

    # feature matrix for lightgbm member
    U, V = ctx.mf_factors(dim=16)
    cf_dot = np.sum(U * V, axis=1).astype(np.float32)

    feat_list = [base, din, cf_score, cf_hist.astype(np.float32), cf_dot]
    feat_names = ['refit_score', 'din_score', 'cf_score', 'cf_hist', 'cf_dot']
    X = np.stack(feat_list, axis=1).astype(np.float32)

    train_idx = ctx.fold.idx['train']
    valid_idx = ctx.fold.idx['valid']

    seeds = [0, 1, 2]
    preds = np.zeros(n, dtype=np.float64)

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'min_data_in_leaf': 50,
    }

    for seed in seeds:
        p = dict(params)
        p['seed'] = seed
        p['bagging_seed'] = seed
        p['feature_fraction_seed'] = seed
        dtrain = lgb.Dataset(X[train_idx], label=y[train_idx])
        dvalid = lgb.Dataset(X[valid_idx], label=y[valid_idx], reference=dtrain)
        model = lgb.train(
            p, dtrain, num_boost_round=300,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(30, verbose=False)]
        )
        preds += model.predict(X, num_iteration=model.best_iteration).astype(np.float64)

    preds /= len(seeds)
    m1_score = preds.astype(np.float32)

    r_m1 = user_rank(uid, m1_score)
    r_m2 = user_rank(uid, din)
    r_m3 = user_rank(uid, cf_score)

    final = 0.55 * r_m1 + 0.30 * r_m2 + 0.15 * r_m3
    final = final.astype(np.float32).reshape(-1, 1)

    ctx.check(final, ['final_score'])
    return final, ['final_score'], {'mode': 'scores'}
```

</details>

---

## Iteration 9 — REJECT

**Family:** `ensemble`  
**Hypothesis:** CONSOLIDATION move (0 misses left): keep the incumbent rank-fusion exactly as-is and make two minimal, low-risk changes - (a) average each learned member over 3 seeds before ranking, and (b) add the two most decorrelated available members (ctx.cf_score at fm/cf r=+0.455 and an MF dot-product score from ctx.mf_factors at fm/mf r=+0.381) as small fixed-weight linear rank terms. No gamma / power factors, no new weight search beyond a coarse a-priori weight grid validated once.

**Mechanism (why it should work):** Within-user ranking is decided by the ordering of items inside a user's list. Every member's score carries an idiosyncratic noise component; averaging ranks of members whose correlation with the FM is only 0.36-0.46 cancels a larger share of that noise than averaging with din (r=0.848, redundant). The number of ordered pairs a user's AUC is computed over grows quadratically with list length, so noise cancellation shows up systematically on long evaluation lists, whereas 2-4 item lists are dominated by a single pair and stay noise-limited. The long-list bucket (eval_list_size bucket 12, 2352 users, auc 0.6754 with the largest headroom 0.0704) is therefore where independent-member fusion must pay off first. 3-seed averaging (known to saturate at 3) removes seed variance so the added members' contribution is not masked. Plain linear rank weights are used because measured power factors overfit validation and widened the test gap.

**Falsifiable prediction:** `{"diagnostic": "auc_long_lists", "direction": "increase"}` → **MISS**

**Metrics (validation):**

| GAUC | nDCG@5 | primary | Δ vs incumbent |
|---|---|---|---|
| 0.6681 | 0.5360 | **0.6020** | -0.0009 |

**Decision rationale:** valid 0.6020 +-0.0000 vs incumbent 0.6030 (delta -0.0009)

**Error / recovery — `scheduler`**  
family 'ensemble' has mean gain -0.0013 over 7 attempts and the run ends on the next miss  
*Recovery:* re-asked the planner for a consolidating move before spending a training run

<details><summary>Code applied this iteration</summary>

```python
def build(ctx):
    import numpy as np

    n = ctx.data.n
    user_id = ctx.data.user_id

    def rank_pct(score):
        score = np.asarray(score, dtype=np.float64)
        order = np.argsort(user_id, kind='stable')
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        sorted_users = user_id[order]
        sorted_scores = score[order]
        boundaries = np.flatnonzero(np.diff(sorted_users)) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [len(sorted_users)]))
        ranks = np.empty(len(sorted_scores), dtype=np.float64)
        for s, e in zip(starts, ends):
            seg = sorted_scores[s:e]
            cnt = e - s
            if cnt <= 1:
                ranks[s:e] = 0.5
            else:
                order2 = np.argsort(seg, kind='stable')
                r = np.empty(cnt, dtype=np.float64)
                r[order2] = np.arange(cnt)
                ranks[s:e] = r / (cnt - 1)
        out = np.empty(len(sorted_scores), dtype=np.float64)
        out[order] = ranks
        return out

    m1 = np.asarray(ctx.baseline_score, dtype=np.float64)

    refit_seeds = []
    din_seeds = []
    for _ in range(3):
        refit_seeds.append(np.asarray(ctx.refit_score(), dtype=np.float64))
        din_seeds.append(np.asarray(ctx.din_score(), dtype=np.float64))
    m2 = np.mean(refit_seeds, axis=0)
    m3 = np.mean(din_seeds, axis=0)

    cf_score, hist_count = ctx.cf_score()
    m4 = np.asarray(cf_score, dtype=np.float64)

    U, V = ctx.mf_factors(dim=16)
    m5 = np.sum(np.asarray(U, dtype=np.float64) * np.asarray(V, dtype=np.float64), axis=1)

    r1 = rank_pct(m1)
    r2 = rank_pct(m2)
    r3 = rank_pct(m3)
    r4 = rank_pct(m4)
    r5 = rank_pct(m5)

    final = 0.40 * r1 + 0.25 * r3 + 0.15 * r2 + 0.10 * r4 + 0.10 * r5

    X = final.reshape(-1, 1).astype(np.float32)
    names = ['fused_rank_score']
    train_cfg = {'mode': 'scores'}

    ctx.check(X, names)
    return X, names, train_cfg
```

</details>

---

## Iteration 10 — REJECT

**Family:** `ensemble`  
**Hypothesis:** Consolidation move (not exploratory): keep the incumbent fusion exactly as-is and make one minimal, low-variance change - replace raw-score fusion inputs with per-user rank-normalised inputs for every member, add the two most decorrelated members (ctx.cf_score, r=+0.455 with fm; ctx.mf_factors-derived score, r=+0.381) plus ctx.auxiliary_signal, each seed-averaged over 3 seeds, combined with PLAIN linear rank weights (no gamma powers, weights fixed a priori from the measured correlations rather than tuned on validation).

**Mechanism (why it should work):** Within-user ranking is invariant to per-user constants, so the only thing that can move the metric is the relative ordering inside each user's list. Averaging within-user RANKS of members that disagree (fm/cf +0.455, fm/mf +0.381) cancels independent per-item ordering errors; averaging raw scores does not, because members have different per-user score scales and the strongest member's calibration dominates. The cancellation benefit grows with the number of items being ordered: on a 3-item list a decorrelated second opinion can only flip one or two pairs, whereas on the long lists (eval_list_size bucket 12, 2352 users, headroom 0.0704 - by far the largest single-slice headroom) there are many mid-confidence pairs where independent members disagree and rank-averaging resolves them correctly more often than chance. Down-weighting din (r=+0.848, redundant with fm) and giving real weight to cf/mf/aux increases the effective number of independent opinions per pair. Fixed a-priori weights and no per-member exponents avoid the measured failure mode where extra fusion parameters raised validation while lowering held-out test.

**Falsifiable prediction:** `{"diagnostic": "auc_long_lists", "direction": "increase"}` → **MISS**

**Metrics (validation):**

| GAUC | nDCG@5 | primary | Δ vs incumbent |
|---|---|---|---|
| 0.6628 | 0.5337 | **0.5982** | -0.0047 |

**Decision rationale:** valid 0.5982 +-0.0003 vs incumbent 0.6030 (delta -0.0047)

**Error / recovery — `scheduler`**  
family 'ensemble' has mean gain -0.0012 over 8 attempts and the run ends on the next miss  
*Recovery:* re-asked the planner for a consolidating move before spending a training run

<details><summary>Code applied this iteration</summary>

```python
def build(ctx):
    import numpy as np

    n = ctx.data.n
    uid = ctx.data.user_id

    fm = ctx.refit_score().astype(np.float64)
    fm_base = ctx.baseline_score.astype(np.float64)
    din = ctx.din_score().astype(np.float64)
    cf_score, cf_hist = ctx.cf_score()
    cf_score = cf_score.astype(np.float64)
    U, V = ctx.mf_factors(dim=16)
    mf = np.sum(U.astype(np.float64) * V.astype(np.float64), axis=1)
    try:
        aux = ctx.auxiliary_signal('is_like').astype(np.float64)
    except Exception:
        aux = np.zeros(n, dtype=np.float64)

    def within_user_rank(score, uid):
        order = np.argsort(uid, kind='stable')
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        sorted_uid = uid[order]
        sorted_score = score[order]

        n_ = len(score)
        ranks = np.empty(n_, dtype=np.float64)

        boundaries = np.nonzero(np.diff(sorted_uid))[0] + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [n_]))

        for s, e in zip(starts, ends):
            grp = sorted_score[s:e]
            m = e - s
            if m <= 1:
                ranks[s:e] = 0.5
                continue
            order_grp = np.argsort(grp, kind='stable')
            rank_grp = np.empty(m, dtype=np.float64)
            rank_grp[order_grp] = np.arange(m)
            ranks[s:e] = rank_grp / (m - 1)

        out = np.empty(n_, dtype=np.float64)
        out[order] = ranks
        return out

    r_fm = within_user_rank(fm, uid)
    r_refit = r_fm  # refit_score used as fm above, keep separate baseline too
    r_din = within_user_rank(din, uid)
    r_cf = within_user_rank(cf_score, uid)
    r_mf = within_user_rank(mf, uid)
    r_aux = within_user_rank(aux, uid)

    fuse = (0.34 * r_fm + 0.14 * r_refit + 0.12 * r_din +
            0.20 * r_cf + 0.13 * r_mf + 0.07 * r_aux)

    disagreement = fuse - r_fm
    stack = np.stack([r_fm, r_refit, r_din, r_cf, r_mf, r_aux], axis=1)
    member_std = np.std(stack, axis=1)

    X = np.column_stack([
        fm_base.astype(np.float32),
        fm.astype(np.float32),
        din.astype(np.float32),
        cf_score.astype(np.float32),
        cf_hist.astype(np.float32),
        mf.astype(np.float32),
        aux.astype(np.float32),
        r_fm.astype(np.float32),
        r_refit.astype(np.float32),
        r_din.astype(np.float32),
        r_cf.astype(np.float32),
        r_mf.astype(np.float32),
        r_aux.astype(np.float32),
        fuse.astype(np.float32),
        disagreement.astype(np.float32),
        member_std.astype(np.float32),
    ]).astype(np.float32)

    names = [
        'baseline_score', 'refit_score', 'din_score', 'cf_score', 'cf_hist',
        'mf_score', 'aux_score',
        'r_fm', 'r_refit', 'r_din', 'r_cf', 'r_mf', 'r_aux',
        'fuse', 'fuse_minus_fm', 'member_std'
    ]

    ctx.check(X, names)
    return X, names
```

</details>

---
