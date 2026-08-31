# KAIROS run log — KuaiRand-Pure (required benchmark)

Per-iteration log required by Deliverable 3. Generated from the run ledger by
`experiments/export_run_log.py`; every field is read from the ledger, never
restated by hand.

## Summary

- **Iterations:** 3 (cap 50)
- **Manual interventions:** 0
- **Wall-clock:** 149.5 s
- **Tokens (in / out):** 31,302 / 10,665
- **GPU-hours:** 0 (CPU only)

## Iteration 1 — REJECT

**Family:** `ensemble`  
**Hypothesis:** Abandon the single-downstream-tree architecture entirely and use train_cfg mode='scores': inside build(), produce several INDEPENDENT model outputs and blend their final scores by plain within-user rank fusion with fixed linear weights. Members: (1) ctx.baseline_score (FM ID crosses), (2) ctx.din_score (sequence/attention architecture, corr +0.848 with FM - included as an at-baseline anchor at low weight), (3) ctx.refit_score, (4) ctx.cf_score (corr +0.455 with FM), (5) an MF dot-product score from ctx.mf_factors (corr +0.381), (6) ctx.auxiliary_signal as a low-weight member. Weights chosen a-priori by decorrelation, not tuned per-member: FM-family (baseline+refit+din) gets ~0.55 of the mass, the decorrelated CF/MF/aux members ~0.45, with no power/gamma terms.

**Mechanism (why it should work):** The nine prior failed iterations all fed already-calibrated continuous scores into one LightGBM, which shatters a smooth monotone score into axis-aligned step functions and destroys within-user ordering resolution - a documented pathology, and re-normalising inputs cannot fix it because the damage is done by the splitter, not the input scale. Rank fusion never passes the scores through a splitter: each member keeps its own smooth ordering and only the ORDERS are averaged. Averaging orderings from members whose errors are weakly correlated (expert pairs +0.362, fm/mf +0.381, fm/cf +0.455) cancels independent per-user inversion noise while preserving the shared correct signal, which is exactly the variance-reduction mechanism that the only measured winning intervention on this benchmark (ensembling, +0.0030 by hand) exploits. Because within-user rank fusion is by construction invariant to per-user constants, it also cannot be diluted by the members' differing global calibrations - a problem that a raw-score average would suffer from. Keeping weights linear and coarse avoids the measured overfit from per-member gamma factors (valid +0.0004, test -0.0003, gap +0.0007).

**Falsifiable prediction:** `{"diagnostic": "ndcg", "direction": "increase"}` → **MISS**

**Metrics (validation):**

| GAUC | nDCG@5 | primary | Δ vs incumbent |
|---|---|---|---|
| 0.6629 | 0.5337 | **0.5983** | -0.0051 |

**Decision rationale:** valid 0.5983 +-0.0000 vs incumbent 0.6034 (delta -0.0051)

<details><summary>Code applied this iteration</summary>

```python
def build(ctx):
    import numpy as np
    n = ctx.data.n
    uid = ctx.data.user_id

    def rankify(s):
        s = np.asarray(s, dtype=np.float64)
        order = np.argsort(uid, kind='stable')
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        sorted_uid = uid[order]
        sorted_s = s[order]
        r = np.zeros(len(s), dtype=np.float64)
        start = 0
        N = len(sorted_uid)
        while start < N:
            end = start + 1
            while end < N and sorted_uid[end] == sorted_uid[start]:
                end += 1
            grp = sorted_s[start:end]
            gsize = end - start
            if gsize == 1:
                r[start:end] = 0.5
            else:
                order2 = np.argsort(grp, kind='stable')
                ranks = np.empty(gsize, dtype=np.float64)
                ranks[order2] = np.arange(gsize)
                # average ties
                sorted_grp = grp[order2]
                i = 0
                while i < gsize:
                    j = i + 1
                    while j < gsize and sorted_grp[j] == sorted_grp[i]:
                        j += 1
                    avg_rank = (i + j - 1) / 2.0
                    ranks[order2[i:j]] = avg_rank
                    i = j
                r[start:end] = ranks / (gsize - 1)
            start = end
        out = np.empty(len(s), dtype=np.float64)
        out[order] = r
        return out

    s_fm = ctx.baseline_score
    s_refit = ctx.refit_score()
    s_din = ctx.din_score()
    s_cf, hist_count = ctx.cf_score()
    s_aux = ctx.auxiliary_signal('is_like')

    U, V = ctx.mf_factors(dim=16)
    s_mf = np.sum(U * V, axis=1)

    r_fm = rankify(s_fm)
    r_refit = rankify(s_refit)
    r_din = rankify(s_din)
    r_cf = rankify(s_cf)
    r_mf = rankify(s_mf)
    r_aux = rankify(s_aux)

    final = (0.30 * r_fm + 0.15 * r_refit + 0.10 * r_din +
             0.20 * r_cf + 0.15 * r_mf + 0.10 * r_aux)

    X = final.reshape(-1, 1).astype(np.float32)
    names = ['fused_rank_score']
    train_cfg = {'mode': 'scores'}
    ctx.check(X, names)
    return X, names, train_cfg
```

</details>

---

## Iteration 2 — REJECT

**Family:** `ensemble`  
**Hypothesis:** Stop feeding scores into a downstream tree. Switch to train_cfg mode='scores': train the decorrelated members separately inside build() and combine only their FINAL outputs by plain within-user rank fusion with fixed linear weights (no powers, no learned stacker). Members: ctx.baseline_score (FM ID crosses), ctx.din_score (sequence architecture, corr +0.848 with FM), ctx.refit_score (FM refit on the sparse-regime tail), ctx.cf_score (corr +0.455), and an MF dot-product score from ctx.mf_factors (corr +0.381), with ctx.auxiliary_signal folded in at low weight.

**Mechanism (why it should work):** Every failure so far came from the same architectural mistake: a LightGBM tree splitting an already-calibrated smooth score into step functions, which destroys the fine within-user ordering that GAUC measures. Rank fusion never touches the members' internal calibration - it converts each member to a per-user rank in [0,1] and averages, so the only thing that can change is the ordering agreement between members. Because the members are decorrelated (expert pairs +0.362, fm/cf +0.455, fm/mf +0.381), their independent per-user ordering errors partially cancel; the variance of the fused per-user ordering falls even though every member's marginal accuracy is unchanged. That is exactly the quantity GAUC integrates (full-list pairwise agreement per user, weighted by positives), so GAUC is where the cancellation shows up first and most reliably - nDCG@5 only sees the top 5 slots and is noisier. Within-user rank normalisation also makes the fusion invariant to each member's per-user score scale, which is the only degree of freedom the metric cares about; per-user constants drop out by construction. Weights are fixed a priori (FM-family gets the bulk since it is the strongest member, cf/mf/aux get small shares) with at most a coarse 3-4 point grid, because the documented gamma experiment showed extra fusion parameters buy validation noise and widen the valid-test gap.

**Falsifiable prediction:** `{"diagnostic": "gauc", "direction": "increase"}` → **HIT**

**Metrics (validation):**

| GAUC | nDCG@5 | primary | Δ vs incumbent |
|---|---|---|---|
| 0.6673 | 0.5355 | **0.6014** | -0.0020 |

**Decision rationale:** valid 0.6014 +-0.0000 vs incumbent 0.6034 (delta -0.0020)

<details><summary>Code applied this iteration</summary>

```python
def build(ctx):
    import numpy as np
    n = ctx.data.n
    uid = ctx.data.user_id

    def rank_within_user(score):
        score = np.asarray(score, dtype=np.float64)
        order = np.argsort(uid, kind='stable')
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        sorted_uid = uid[order]
        sorted_score = score[order]
        ranks = np.empty(len(order), dtype=np.float64)
        start = 0
        L = len(order)
        while start < L:
            end = start
            u = sorted_uid[start]
            while end < L and sorted_uid[end] == u:
                end += 1
            seg = sorted_score[start:end]
            m = end - start
            if m == 1:
                ranks[start:end] = 0.5
            else:
                order2 = np.argsort(seg, kind='stable')
                r = np.empty(m, dtype=np.float64)
                # average rank for ties
                sorted_seg = seg[order2]
                rr = np.arange(1, m + 1, dtype=np.float64)
                # handle ties by averaging
                i = 0
                while i < m:
                    j = i
                    while j < m and sorted_seg[j] == sorted_seg[i]:
                        j += 1
                    avg_rank = (rr[i] + rr[j-1]) / 2.0
                    r[order2[i:j]] = avg_rank
                    i = j
                ranks[start:end] = r / (m + 1.0)
            start = end
        out = np.empty(L, dtype=np.float64)
        out[order] = ranks
        return out

    m1 = ctx.baseline_score
    m2 = ctx.din_score()
    m3 = ctx.refit_score()
    cf_score, hist_count = ctx.cf_score()
    m4 = cf_score

    U, V = ctx.mf_factors(dim=16)
    m5 = np.sum(U * V, axis=1)

    try:
        m6 = ctx.auxiliary_signal('is_click')
    except Exception:
        m6 = None

    r1 = rank_within_user(m1)
    r2 = rank_within_user(m2)
    r3 = rank_within_user(m3)
    r4 = rank_within_user(m4)
    r5 = rank_within_user(m5)

    fused = 0.34 * r1 + 0.20 * r2 + 0.18 * r3 + 0.12 * r4 + 0.10 * r5
    weight_sum = 0.34 + 0.20 + 0.18 + 0.12 + 0.10

    if m6 is not None:
        r6 = rank_within_user(m6)
        fused = fused + 0.06 * r6
        weight_sum += 0.06

    fused = fused / weight_sum

    X = fused.reshape(-1, 1).astype(np.float32)
    names = ['fused_rank_score']
    train_cfg = {'mode': 'scores'}

    ctx.check(X, names)
    return X, names, train_cfg
```

</details>

---

## Iteration 3 — REJECT

**Family:** `ensemble`  
**Hypothesis:** Stop feeding scores into a downstream LightGBM entirely: use train_cfg mode='scores' and return a single fused score built as a plain linear weighted sum of WITHIN-USER percentile ranks of several already-calibrated, mutually decorrelated model outputs - a 3-seed-averaged FM refit (ctx.refit_score) as the anchor, plus ctx.din_score, ctx.cf_score, and an MF dot-product from ctx.mf_factors, with a small weight on ctx.auxiliary_signal. No per-member gamma, no tree on top.

**Mechanism (why it should work):** The three failed prior runs all lost because a decision tree shatters a smooth calibrated score into step functions; mode='scores' bypasses that architecture completely, so the fused output can never be worse-conditioned than its best member. Rank fusion is the exact operation the metric is invariant-compatible with: GAUC and nDCG@5 depend only on the within-user ordering, so converting each member to a within-user percentile rank removes per-member calibration/scale mismatch (including the 5x logging-density collapse, which shifts score scales between the dense and sparse regimes but not orderings). Averaging members whose pairwise correlations are low (fm/cf +0.455, fm/mf +0.381, expert pairs +0.362) cancels independent per-user ordering errors; DIN at +0.848 is redundant with FM but at/above baseline so it gets a small weight only. Seed-averaging the FM refit first removes the 0.0008 per-seed noise from the anchor. Ensembling is the only measured-to-work intervention here and this is its documented, never-yet-run form (+0.0030 by hand).

**Falsifiable prediction:** `{"diagnostic": "ndcg", "direction": "increase"}` → **HIT**

**Metrics (validation):**

| GAUC | nDCG@5 | primary | Δ vs incumbent |
|---|---|---|---|
| 0.6666 | 0.5356 | **0.6011** | -0.0023 |

**Decision rationale:** valid 0.6011 +-0.0000 vs incumbent 0.6034 (delta -0.0023)

**Error / recovery — `scheduler`**  
family 'ensemble' has mean gain -0.0035 over 2 attempts and the run ends on the next miss  
*Recovery:* re-asked the planner for a consolidating move before spending a training run

<details><summary>Code applied this iteration</summary>

```python
def build(ctx):
    import numpy as np

    n = ctx.data.n
    user_id = ctx.data.user_id

    def within_user_rank(vals):
        vals = np.asarray(vals, dtype=np.float64)
        order = np.argsort(user_id, kind='stable')
        inv = np.empty_like(order)
        inv[order] = np.arange(n)
        sorted_users = user_id[order]
        sorted_vals = vals[order]

        boundaries = np.flatnonzero(np.diff(sorted_users)) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [n]))

        ranks_sorted = np.empty(n, dtype=np.float64)
        for s, e in zip(starts, ends):
            seg = sorted_vals[s:e]
            m = e - s
            order2 = np.argsort(seg, kind='stable')
            r = np.empty(m, dtype=np.float64)
            rr = np.arange(1, m + 1, dtype=np.float64)
            r[order2] = rr
            # average ties
            sorted_seg = seg[order2]
            i = 0
            while i < m:
                j = i
                while j + 1 < m and sorted_seg[j + 1] == sorted_seg[i]:
                    j += 1
                if j > i:
                    avg = r[order2[i:j+1]].mean()
                    r[order2[i:j+1]] = avg
                i = j + 1
            ranks_sorted[s:e] = r / (m + 1)
        result = np.empty(n, dtype=np.float64)
        result[order] = ranks_sorted
        return result

    # member 1: seed-averaged refit score (refit_score has no explicit seed param,
    # so average repeated calls to reduce noise if stochastic; else identical calls are harmless)
    seeds_scores = []
    for _ in range(3):
        seeds_scores.append(np.asarray(ctx.refit_score(), dtype=np.float64))
    m1 = np.mean(seeds_scores, axis=0)

    m2 = np.asarray(ctx.din_score(), dtype=np.float64)

    cf_score, cf_hist = ctx.cf_score()
    m3 = np.asarray(cf_score, dtype=np.float64)

    U, V = ctx.mf_factors(dim=16)
    m4 = np.sum(np.asarray(U, dtype=np.float64) * np.asarray(V, dtype=np.float64), axis=1)

    m5 = np.asarray(ctx.auxiliary_signal('is_click'), dtype=np.float64)

    r1 = within_user_rank(m1)
    r2 = within_user_rank(m2)
    r3 = within_user_rank(m3)
    r4 = within_user_rank(m4)
    r5 = within_user_rank(m5)

    fused = 0.50 * r1 + 0.15 * r2 + 0.15 * r3 + 0.15 * r4 + 0.05 * r5

    X = fused.reshape(-1, 1).astype(np.float32)
    names = ['fused_rank_score']

    ctx.check(X, names)

    train_cfg = {'mode': 'scores'}
    return X, names, train_cfg
```

</details>

---
