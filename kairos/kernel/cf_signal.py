"""Item-item collaborative filtering with IDF weighting, leakage-safe like mf_signal.py.

Raw co-occurrence cosine similarity (tried in exp04 as item_item_cf) showed weak feature
gain: a popular item co-occurs with almost everything a user watches, so its similarity
scores are uninformative and drown out niche, genuinely diagnostic co-occurrences. IDF
weighting (Breese, Heckerman & Kadie 1998) down-weights each item's contribution by how
common it is - the way TF-IDF down-weights common words in text retrieval. Two users who
both watched an obscure video is much stronger evidence of taste overlap than two users
who both watched a blockbuster.

Leakage discipline matches mf_signal.py exactly: one similarity matrix PER FROZEN WINDOW,
built only from positives dated at or before that window's horizon; a row's score is the
mean similarity between its candidate item and the items in THIS USER's frozen history
from that same fit set - never the row's own list-mates, since fit always predates the
window's own rows by construction of the window schedule.
"""
import os
import numpy as np
from kairos.kernel.dataset import variant_path
from scipy.sparse import csr_matrix

CACHE_DIR = variant_path('runs/cf_cache')


def build_cf_score(data, windows, cache_dir=CACHE_DIR, force=False, fit_end=None):
    """Returns (score, hist_count): float32 (n,) each.
       score      = mean IDF-weighted similarity between the row's item and the items in
                    this user's frozen history (0 if the user has no prior history)
       hist_count = size of that history (0 = cold start; usable as a confidence weight)
    """
    os.makedirs(cache_dir, exist_ok=True)
    sp = os.path.join(cache_dir, 'score.npy'); cp = os.path.join(cache_dir, 'count.npy')
    if os.path.exists(sp) and os.path.exists(cp) and not force:
        return np.load(sp), np.load(cp)

    n_items = int(data.video_id.max()) + 1
    n_users = int(data.user_id.max()) + 1
    date = data.date.astype(np.int64)
    score = np.zeros(data.n, dtype=np.float32)
    count = np.zeros(data.n, dtype=np.float32)

    for lo, hi, hz in windows:
        rows = np.flatnonzero((date >= lo) & (date <= hi))
        # FAQ 2.9.2: the co-occurrence / factorisation model is FIT on train-split rows
        # only. The window horizon may run to 20220428; the fit set may not.
        cut = hz if fit_end is None else min(hz, fit_end)
        fit = np.flatnonzero((date <= cut) & (data.y_raw == 1))
        if len(rows) == 0 or len(fit) < 200:
            continue
        u_idx = data.user_id[fit].astype(np.int64)
        v_idx = data.video_id[fit].astype(np.int64)
        M = csr_matrix((np.ones(len(fit), dtype=np.float64), (u_idx, v_idx)),
                       shape=(n_users, n_items))
        M.data[:] = 1.0

        item_users = np.asarray(M.sum(0)).ravel()
        idf = np.log1p(n_users / np.maximum(item_users, 1.0))
        Mw = M.multiply(idf[np.newaxis, :]).tocsr()
        C = (M.T @ Mw).toarray()
        norm = np.sqrt(np.maximum(np.diag(C), 1e-9))
        C = (C / norm[:, None] / norm[None, :]).astype(np.float32)
        np.fill_diagonal(C, 0.0)

        # group this window's rows by user: within a window every row of the same user
        # shares the identical frozen history, so the gather is done once per user.
        indptr, indices = M.indptr, M.indices
        ru = data.user_id[rows].astype(np.int64)
        rv = data.video_id[rows].astype(np.int64)
        order = np.argsort(ru, kind='stable')
        ru_s, rv_s, rows_s = ru[order], rv[order], rows[order]
        starts = np.flatnonzero(np.r_[True, ru_s[1:] != ru_s[:-1]])
        sizes = np.diff(np.r_[starts, len(ru_s)])
        for s, sz in zip(starts, sizes):
            u = ru_s[s]
            hist = indices[indptr[u]:indptr[u + 1]]
            if len(hist) == 0:
                continue
            items = rv_s[s:s + sz]
            score[rows_s[s:s + sz]] = C[items][:, hist].mean(1)
            count[rows_s[s:s + sz]] = len(hist)
        print(f"  CF window {lo}-{hi} (horizon {hz}): {len(fit):,} positives -> "
              f"scored {len(rows):,} rows", flush=True)

    np.save(sp, score); np.save(cp, count)
    return score, count
