"""Turn a prediction vector into an evidence digest.

An agent that reflects on a single scalar ("primary went from .601 to .603, try
something else") is guessing.  The same run contains far more information: which users
lose points, which item types get wrongly ranked above positives, and how much of the
metric is even recoverable in each slice.  This module extracts that, and the digest it
emits is what the LLM sees - a few hundred tokens of evidence instead of a raw log.

Headroom attribution
--------------------
primary = mean(GAUC, nDCG@5), both aggregated over users, so per-user contributions are
exact and additive:
    a user's nDCG share is  ndcg_g / n_users, ceiling 1[n_pos_g > 0] / n_users
    a user's GAUC share is  n_pos_g * auc_g / sum_g n_pos_g, ceiling n_pos_g / sum n_pos
The gap between share and ceiling is that slice's recoverable points.  Slices are ranked
by it, so the agent optimises where the points actually are rather than where the
accuracy looks worst.

Inversion attribution
---------------------
GAUC loss IS the weighted rate of misordered (positive, negative) pairs.  Materialising
those pairs and grouping by the attributes of the negative that beat a positive answers
"what kind of item are we systematically over-ranking?", which points at a mechanism
rather than a symptom.
"""
import numpy as np
from kairos.kernel.fastmetrics import per_group_metrics, factorize


def _decile(x, n=10):
    """Rank-based bucketing that tolerates heavy ties (item popularity is very skewed)."""
    r = np.argsort(np.argsort(x, kind='stable'), kind='stable')
    return np.minimum((r * n) // max(len(x), 1), n - 1)


class Diagnostics:
    def __init__(self, fold, part, scores, extra_slices=None):
        d = fold.data
        self.fold, self.part = fold, part
        idx = fold.idx[part]
        self.idx = idx
        self.scores = np.asarray(scores, dtype=np.float64)
        self.uid = d.user_id[idx]
        self.gid, self.n_groups = factorize(self.uid)
        self.y = d.y_raw[idx]
        self.pg = per_group_metrics(self.gid, self.y, self.scores)
        self.n_users = self.n_groups
        self.tot_pos = self.pg['npos'].sum()

        # ---- per-user slice variables ------------------------------------
        tr = fold.idx['train']
        u_train_cnt = np.bincount(d.user_id[tr], minlength=int(d.user_id.max()) + 1)
        i_train_cnt = np.bincount(d.video_id[tr], minlength=int(d.video_id.max()) + 1)
        i_train_pos = np.bincount(d.video_id[tr], weights=d.y_raw[tr].astype(float),
                                  minlength=int(d.video_id.max()) + 1)
        self.i_train_cnt, self.i_train_pos = i_train_cnt, i_train_pos

        # group_key from per_group_metrics is the factorized gid; map back to user ids
        order = np.argsort(self.gid, kind='stable')
        first = np.flatnonzero(np.r_[True, self.gid[order][1:] != self.gid[order][:-1]])
        self.user_of_group = self.uid[order][first]

        self.user_slices = {
            'user_train_impressions': _decile(u_train_cnt[self.user_of_group]),
            'eval_list_size': np.clip(self.pg['size'], 1, 12),
            'eval_n_positives': np.clip(self.pg['npos'].astype(int), 0, 8),
        }
        if extra_slices:
            self.user_slices.update(extra_slices)

        # ---- row-level attributes for inversion analysis -------------------
        dur = d.col('duration_ms')[idx].astype(np.float64)
        self.row_attr = {
            'duration_decile': _decile(dur),
            'item_pop_decile': _decile(i_train_cnt[d.video_id[idx]]),
            'item_cold': (i_train_cnt[d.video_id[idx]] == 0).astype(int),
            'tab': np.clip(d.col('tab')[idx], 0, 12),
        }

    # ------------------------------------------------------------------ headroom
    def headroom_table(self, slice_name):
        """Per-bucket: share of primary held, ceiling, and recoverable points."""
        s = self.user_slices[slice_name]
        nd, npos, auc = self.pg['ndcg'], self.pg['npos'], self.pg['auc']
        valid = self.pg['valid_for_gauc']
        rows = []
        for b in np.unique(s):
            m = s == b
            nd_share = nd[m].sum() / self.n_users
            nd_ceil = (npos[m] > 0).sum() / self.n_users
            mv = m & valid
            g_share = (npos[mv] * auc[mv]).sum() / self.tot_pos
            g_ceil = npos[mv].sum() / self.tot_pos
            rows.append({
                'bucket': int(b), 'users': int(m.sum()),
                'ndcg_mean': float(nd[m].mean()) if m.sum() else 0.0,
                'auc_mean': float(np.nanmean(auc[mv])) if mv.sum() else float('nan'),
                'primary_held': float((nd_share + g_share) / 2),
                'primary_ceiling': float((nd_ceil + g_ceil) / 2),
                'headroom': float((nd_ceil - nd_share + g_ceil - g_share) / 2),
            })
        return sorted(rows, key=lambda r: -r['headroom'])

    # ------------------------------------------------------------------ inversions
    def inversions(self, attr, max_users=None):
        """Group misordered (pos, neg) pairs by an attribute of the NEGATIVE that won.

        Returns per-bucket: inversion rate, and the GAUC points lost to that bucket.
        """
        a = self.row_attr[attr]
        order = np.lexsort((np.arange(len(self.gid)), self.gid))
        g = self.gid[order]
        starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
        sizes = np.diff(np.r_[starts, len(g)])
        loss = {}
        tot = 0.0
        for st, sz in zip(starts, sizes):
            sl = order[st:st + sz]
            yy, ss, aa = self.y[sl], self.scores[sl], a[sl]
            p = np.flatnonzero(yy == 1); n = np.flatnonzero(yy == 0)
            if len(p) == 0 or len(n) == 0:
                continue
            # weight of one pair in GAUC == 1 / (n_neg * total_pos)   [n_pos cancels]
            w = 1.0 / (len(n) * self.tot_pos)
            dif = ss[p][:, None] - ss[n][None, :]
            bad = (dif < 0) + 0.5 * (dif == 0)
            tot += bad.sum() * w
            per_neg = bad.sum(0) * w
            for b, v in zip(aa[n], per_neg):
                loss[int(b)] = loss.get(int(b), 0.0) + float(v)
        return {'total_gauc_loss': float(tot),
                'by_bucket': dict(sorted(loss.items(), key=lambda kv: -kv[1]))}

    # ------------------------------------------------------------------ digest
    def digest(self, top=4):
        """Compact evidence summary - this is what gets sent to the LLM."""
        from kairos.kernel.fastmetrics import fast_evaluate
        m = fast_evaluate(self.gid, self.y, self.scores)
        out = {'primary': round(m['primary'], 5), 'GAUC': round(m['GAUC'], 5),
               'nDCG@5': round(m['nDCG@5'], 5), 'users': self.n_users}
        # ceiling
        ceil = fast_evaluate(self.gid, self.y, self.y.astype(float))
        out['oracle_primary'] = round(ceil['primary'], 5)
        out['headroom_total'] = round(ceil['primary'] - m['primary'], 5)
        out['slices'] = {}
        for name in self.user_slices:
            t = self.headroom_table(name)
            out['slices'][name] = [
                {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
                 if k in ('bucket', 'users', 'auc_mean', 'headroom')} for r in t[:top]]
        out['inversions'] = {}
        for a in ('duration_decile', 'item_pop_decile'):
            inv = self.inversions(a)
            top_b = list(inv['by_bucket'].items())[:top]
            out['inversions'][a] = {'total_gauc_loss': round(inv['total_gauc_loss'], 4),
                                    'worst_buckets': [[b, round(v, 4)] for b, v in top_b]}
        return out
