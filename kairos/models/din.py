"""DIN-style target attention over a user's frozen behaviour sequence.

The organizers flag sequence modelling as completely unexplored on this benchmark, and our
own behavioural features are aggregates - a long_view RATE summarises history but discards
which items it was. Target attention asks a sharper question: how similar is this candidate
video to the specific things this user actually watched?

Leakage discipline is the same as everywhere else: the sequence for a row contains only
items long-viewed at or before that row's FROZEN WINDOW HORIZON, never its list-mates.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_sequences(data, hz, max_len=32):
    """-> seq (N, max_len) int32 item ids (0 = pad), seq_len (N,) int16.

    One pass per distinct horizon: take every long_view up to that horizon, order by time,
    keep each user's most recent `max_len`.
    """
    n = data.n
    seq = np.zeros((n, max_len), dtype=np.int32)
    slen = np.zeros(n, dtype=np.int16)
    date = data.date.astype(np.int64)
    for h in np.unique(hz):
        rows = np.flatnonzero(hz == h)
        if h < 0 or len(rows) == 0:
            continue
        m = np.flatnonzero((date <= h) & (data.y_raw == 1))
        if len(m) == 0:
            continue
        order = m[np.lexsort((np.arange(len(m)), data.time_ms[m], data.user_id[m]))]
        u = data.user_id[order]
        it = data.video_id[order].astype(np.int32) + 1        # reserve 0 for padding
        starts = np.flatnonzero(np.r_[True, u[1:] != u[:-1]])
        sizes = np.diff(np.r_[starts, len(u)])
        # per-user tail of length max_len
        table = {}
        for s, sz in zip(starts, sizes):
            table[int(u[s])] = it[s + max(0, sz - max_len): s + sz]
        for r in rows:
            h_ = table.get(int(data.user_id[r]))
            if h_ is None:
                continue
            k = len(h_)
            seq[r, :k] = h_
            slen[r] = k
    return seq, slen


class DIN(nn.Module):
    """Target attention over the history, concatenated with the usual ID crosses.

    forward(X, item_id, seq, slen):
        X       (B,F) offset-encoded categorical field ids (user, video, author, tab, dur)
        item_id (B,)  raw video id of the candidate, +1 so 0 stays reserved for padding
        seq     (B,L) the user's frozen history, padded with 0
    """

    def __init__(self, n_items, field_dims, k=32, hidden=64, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.item_emb = nn.Embedding(n_items + 1, k, padding_idx=0)
        nn.init.normal_(self.item_emb.weight, 0, 0.01)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()
        self.field_emb = nn.Embedding(field_dims, k)
        nn.init.normal_(self.field_emb.weight, 0, 0.01)
        self.att = nn.Sequential(nn.Linear(4 * k, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.mlp = nn.Sequential(nn.Linear(3 * k, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden // 2), nn.ReLU(),
                                 nn.Linear(hidden // 2, 1))

    def forward(self, X, item_id, seq, slen=None):
        fe = self.field_emb(X).sum(1)                  # (B,k) pooled ID crosses
        tgt = self.item_emb(item_id)                   # (B,k) candidate item
        h = self.item_emb(seq)                         # (B,L,k) history
        mask = seq > 0
        t = tgt.unsqueeze(1).expand_as(h)
        a = self.att(torch.cat([t, h, t - h, t * h], -1)).squeeze(-1)
        a = a.masked_fill(~mask, -1e9)
        w = torch.softmax(a, dim=1) * mask.float()
        # users with no frozen history contribute a zero interest vector rather than NaN
        interest = (w.unsqueeze(-1) * h).sum(1)
        interest = torch.where(mask.any(1, keepdim=True), interest,
                               torch.zeros_like(interest))
        return self.mlp(torch.cat([fe, tgt, interest], -1)).squeeze(-1)
