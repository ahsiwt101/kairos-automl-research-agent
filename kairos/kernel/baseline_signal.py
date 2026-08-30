"""Out-of-sample baseline (FM) scores, usable as an input feature.

An agent whose only action is "build a feature matrix for a GBDT" cannot reproduce the
official FM, so measuring it against the FM's score guarantees it rejects everything and
stalls out. Giving it the baseline's predictions turns an unwinnable comparison into the
thing a research agent should actually do: start from the best known model and improve it.

Leakage discipline matches the rest of the kernel. For each frozen window, an FM is
trained ONLY on rows dated at or before that window's horizon and then used to predict the
window. Every score is therefore out-of-sample and temporally causal - never a prediction
from a model that has seen the row it is scoring, nor anything after that row's horizon.
"""
import os
import numpy as np

CACHE = 'runs/fm_signal.npy'


def build_fm_signal(data, windows, k=16, lr=1e-3, epochs=8, seed=0, cache=CACHE,
                    force=False):
    if os.path.exists(cache) and not force:
        return np.load(cache)
    import torch
    from kairos.kernel.features import Encoder
    from kairos.models.ranker import FM

    date = data.date.astype(np.int64)
    out = np.zeros(data.n, dtype=np.float32)
    for lo, hi, hz in windows:
        rows = np.flatnonzero((date >= lo) & (date <= hi))
        fit = np.flatnonzero(date <= hz)
        if len(rows) == 0 or len(fit) < 5000:
            continue                       # no usable history yet; leave the score at 0
        enc = Encoder(data).fit(fit)
        X = enc.transform(fit)
        y = data.y_raw[fit].astype(np.float32)
        torch.manual_seed(seed)
        m = FM(enc.dim, k=k, seed=seed)
        opt = torch.optim.Adam(m.parameters(), lr=lr)
        Xt = torch.from_numpy(X.astype(np.int64)); yt = torch.from_numpy(y)
        rng = np.random.default_rng(seed)
        for _ in range(epochs):
            perm = rng.permutation(len(y))
            for i in range(0, len(perm), 8192):
                s = torch.from_numpy(perm[i:i + 8192])
                opt.zero_grad()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    m(Xt[s]), yt[s])
                loss.backward(); opt.step()
        with torch.no_grad():
            Xr = torch.from_numpy(enc.transform(rows).astype(np.int64))
            out[rows] = m(Xr).numpy().astype(np.float32)
        print(f"  window {lo}-{hi} (horizon {hz}): fit {len(fit):,} rows, "
              f"scored {len(rows):,}", flush=True)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.save(cache, out)
    return out
