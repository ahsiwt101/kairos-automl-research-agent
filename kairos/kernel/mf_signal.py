"""Implicit-feedback matrix factorization (ALS), leakage-safe like baseline_signal.py.

The organizers' own ablation shows FM's pointwise `user_id x video_id` cross has already
absorbed most of what static features can add - but train is only 0.58% dense
(26,210 users x 7,538 items), which is enough signal for collaborative factorization and
too sparse for a per-pair embedding to generalize well from co-occurrence alone. MF learns
a LOW-RANK structure across the whole interaction matrix at once, which is a different
inductive bias than FM's per-ID embeddings, not a bigger version of the same one.

Algorithm: Hu, Koren & Volinsky (2008) implicit-feedback ALS. Each observed long_view is
a preference p_ui=1 with confidence c_ui=1+alpha; every unobserved pair is p_ui=0 with
confidence 1. Alternating least squares admits a closed form per row using the standard
trick  Y^T C_u Y = Y^T Y + alpha * Y_u^T Y_u  (only over the SMALL set of items a user
actually interacted with), so a full pass costs O(rows) small (dim x dim) solves rather
than a dense n_users x n_items computation.

Leakage discipline matches build_fm_signal: one factorization PER FROZEN WINDOW, fit only
on positives dated at or before that window's horizon, applied to that window's own rows.
The factor vectors are returned raw (not as a pre-combined score) - dot product, cosine,
or per-dimension crosses are all legitimate, and it is the agent's call which to use.
"""
import os
import numpy as np
from kairos.kernel.dataset import variant_path
from scipy.sparse import csr_matrix

CACHE_DIR = variant_path('runs/mf_cache')


def _als_pass(fixed, fixed_gram, interactions, dim, reg, alpha):
    """Solve for the free side of ALS. `interactions` is a CSR matrix (n_free, n_fixed);
    `fixed` is that side's factor matrix; `fixed_gram` is fixed.T @ fixed."""
    n_free = interactions.shape[0]
    out = np.zeros((n_free, dim), dtype=np.float64)
    reg_I = reg * np.eye(dim)
    indptr, indices = interactions.indptr, interactions.indices
    for i in range(n_free):
        idx = indices[indptr[i]:indptr[i + 1]]
        if len(idx) == 0:
            continue
        Fi = fixed[idx]                                    # (k, dim), k = #interactions
        A = fixed_gram + alpha * (Fi.T @ Fi) + reg_I
        b = (1.0 + alpha) * Fi.sum(0)
        try:
            out[i] = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            out[i] = np.linalg.lstsq(A, b, rcond=None)[0]
    return out


def build_mf_factors(data, windows, dim=16, reg=0.05, alpha=40.0, iters=6, seed=0,
                     cache_dir=CACHE_DIR, force=False, fit_end=None):
    """Returns (U_row, V_row): per-row user & item factor vectors, float32 (n, dim) each.

    dot(U_row[i], V_row[i]) is a personalised collaborative-filtering score for row i;
    the raw vectors can also be crossed with other features or fed to a tree model
    dimension-by-dimension.
    """
    os.makedirs(cache_dir, exist_ok=True)
    up = os.path.join(cache_dir, f'U_d{dim}.npy')
    vp = os.path.join(cache_dir, f'V_d{dim}.npy')
    if os.path.exists(up) and os.path.exists(vp) and not force:
        return np.load(up), np.load(vp)

    n_users = int(data.user_id.max()) + 1
    n_items = int(data.video_id.max()) + 1
    date = data.date.astype(np.int64)
    U_row = np.zeros((data.n, dim), dtype=np.float32)
    V_row = np.zeros((data.n, dim), dtype=np.float32)
    rng = np.random.default_rng(seed)

    for lo, hi, hz in windows:
        rows = np.flatnonzero((date >= lo) & (date <= hi))
        # FAQ 2.9.2: the co-occurrence / factorisation model is FIT on train-split rows
        # only. The window horizon may run to 20220428; the fit set may not.
        cut = hz if fit_end is None else min(hz, fit_end)
        fit = np.flatnonzero((date <= cut) & (data.y_raw == 1))
        if len(rows) == 0 or len(fit) < 500:
            continue
        u_idx = data.user_id[fit].astype(np.int64)
        v_idx = data.video_id[fit].astype(np.int64)
        R = csr_matrix((np.ones(len(fit), dtype=np.float64), (u_idx, v_idx)),
                       shape=(n_users, n_items))
        R.data[:] = 1.0
        Rt = R.T.tocsr()

        X = rng.normal(0, 0.01, (n_users, dim))
        Y = rng.normal(0, 0.01, (n_items, dim))
        for _ in range(iters):
            X = _als_pass(Y, Y.T @ Y, R, dim, reg, alpha)
            Y = _als_pass(X, X.T @ X, Rt, dim, reg, alpha)

        ru = data.user_id[rows].astype(np.int64)
        rv = data.video_id[rows].astype(np.int64)
        U_row[rows] = X[ru].astype(np.float32)
        V_row[rows] = Y[rv].astype(np.float32)
        print(f"  MF window {lo}-{hi} (horizon {hz}): {len(fit):,} positives, "
              f"{n_users:,}x{n_items:,} matrix -> scored {len(rows):,} rows", flush=True)

    np.save(up, U_row); np.save(vp, V_row)
    return U_row, V_row
