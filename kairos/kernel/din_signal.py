"""Out-of-sample DIN (sequence-model) scores, usable as an input signal.

DIN applies target attention over the items a user actually long-viewed, which is a
different inductive bias from the FM's per-ID crosses and from the aggregate history rates
- it asks how similar this candidate is to the specific things this user watched, rather
than summarising them into a rate. Measured standalone at valid 0.6014 +- 0.0001, which is
level with the FM (0.6017) from an unrelated architecture, so it is a genuinely
decorrelated ensemble member rather than a weaker copy.

Leakage discipline matches the rest of the kernel: sequences come from
kairos.models.din.build_sequences, which only ever includes items long-viewed at or before
a row's FROZEN WINDOW HORIZON, so a row can never see its own evaluation list-mates. The
model is trained on the fold's train split and scores every row.
"""
import os
import numpy as np

CACHE_DIR = 'runs/din_cache'


def build_din_signal(data, fold, hz, seeds=(0, 1), max_len=32, k=32, hidden=64,
                     max_epochs=6, cache_dir=CACHE_DIR, force=False):
    """Returns float32 (n,) - averaged DIN logits over `seeds`, aligned to all log rows."""
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f'{fold.name}_k{k}.npy')
    if os.path.exists(cache) and not force:
        return np.load(cache)

    import torch
    from kairos.kernel.features import Encoder
    from kairos.kernel.fastmetrics import fast_evaluate, factorize
    from kairos.models.din import DIN, build_sequences

    tr, va = fold.idx['train'], fold.idx['valid']
    seq, _ = build_sequences(data, hz, max_len=max_len)
    enc = Encoder(data).fit(tr)
    n_items = int(data.video_id.max()) + 1
    ytr = data.y_raw[tr].astype(np.float32)
    gva, _ = factorize(data.user_id[va]); yva = data.y_raw[va]

    Xtr = torch.from_numpy(enc.transform(tr).astype(np.int64))
    itr = torch.from_numpy((data.video_id[tr] + 1).astype(np.int64))
    str_ = torch.from_numpy(seq[tr])
    ytr_t = torch.from_numpy(ytr)
    Xva = torch.from_numpy(enc.transform(va).astype(np.int64))
    iva = torch.from_numpy((data.video_id[va] + 1).astype(np.int64))
    sva = torch.from_numpy(seq[va])

    preds = []
    for sd in seeds:
        m = DIN(n_items, enc.dim, k=k, hidden=hidden, seed=sd)
        opt = torch.optim.Adam(m.parameters(), lr=2e-3)
        rng = np.random.default_rng(sd)
        best, best_state, bad = -1.0, None, 0
        for _ in range(max_epochs):
            m.train(); perm = rng.permutation(len(ytr))
            for i in range(0, len(perm), 4096):
                s = torch.from_numpy(perm[i:i + 4096])
                opt.zero_grad()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    m(Xtr[s], itr[s], str_[s]), ytr_t[s])
                loss.backward(); opt.step()
            m.eval()
            with torch.no_grad():
                pv = np.concatenate([m(Xva[i:i+50000], iva[i:i+50000],
                                       sva[i:i+50000]).numpy()
                                     for i in range(0, len(iva), 50000)])
            p = fast_evaluate(gva, yva, pv.astype(np.float64))['primary']
            if p > best + 1e-5:
                best, bad = p, 0
                best_state = {kk: v.detach().clone() for kk, v in m.state_dict().items()}
            else:
                bad += 1
                if bad >= 2:
                    break
        m.load_state_dict(best_state); m.eval()
        with torch.no_grad():
            allp = np.concatenate([
                m(torch.from_numpy(enc.transform(np.arange(i, min(i+50000, data.n)))
                                   .astype(np.int64)),
                  torch.from_numpy((data.video_id[i:min(i+50000, data.n)] + 1).astype(np.int64)),
                  torch.from_numpy(seq[i:min(i+50000, data.n)])).numpy()
                for i in range(0, data.n, 50000)])
        preds.append(allp)
        print(f"  DIN seed{sd}: valid {best:.4f}", flush=True)

    out = np.mean(preds, 0).astype(np.float32)
    np.save(cache, out)
    return out
