"""Out-of-sample per-row signals, usable as input features.

An agent whose only action is "build a feature matrix for a GBDT" cannot reproduce the
official FM's ID crosses, so measuring it against the FM's score guarantees it rejects
everything and stalls out. Giving it the baseline's predictions turns an unwinnable
comparison into the thing a research agent should actually do: start from the best known
model and improve it. The same trainer, pointed at a different label column, produces
leakage-safe auxiliary signals for the other feedback types (is_click, is_like, ...).

Leakage discipline is uniform across every signal this module produces. For each frozen
window, an FM is trained ONLY on rows dated at or before that window's horizon and then
used to predict the window. Every score is therefore out-of-sample and temporally causal -
never a prediction from a model that has seen the row it is scoring, nor anything after
that row's horizon.
"""
import os
import numpy as np
from kairos.kernel.dataset import variant_path

CACHE = variant_path('runs/fm_signal.npy')
AUX_CACHE_DIR = variant_path('runs/aux_cache')


def _train_windowed_fm(data, windows, y_all, k=16, lr=1e-3, epochs=8, seed=0):
    """Shared trainer: an independent FM per frozen window, each fit on rows dated at or
    before that window's horizon, predicting y_all restricted to the window's own rows."""
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
        y = y_all[fit].astype(np.float32)
        if y.max() == y.min():             # degenerate window (all-0 or all-1 label)
            out[rows] = float(y.mean())
            continue
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
    return out


def build_fm_signal(data, windows, k=16, lr=1e-3, epochs=8, seed=0, cache=CACHE,
                    force=False):
    """The official baseline's own out-of-sample long_view score."""
    if os.path.exists(cache) and not force:
        return np.load(cache)
    out = _train_windowed_fm(data, windows, data.y_raw.astype(np.float32),
                             k=k, lr=lr, epochs=epochs, seed=seed)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.save(cache, out)
    return out


AUX_COLUMNS = ('is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward')


def build_auxiliary_signal(data, windows, name, k=16, lr=1e-3, epochs=6, seed=0,
                           force=False, cache_dir=AUX_CACHE_DIR):
    """Out-of-sample propensity for an auxiliary feedback signal (is_click, is_like, ...).

    Uses the SAME windowed-FM construction as the baseline score, applied to a different
    binary column. This is the leakage-safe route to feedback signals that are otherwise
    blocked at ctx.col() - the raw column is this row's own outcome and cannot be used
    directly; this is a model's out-of-sample BELIEF about the row, trained on strictly
    earlier data, which is legitimate.
    """
    if name not in AUX_COLUMNS:
        raise ValueError(f"auxiliary_signal: '{name}' not in {AUX_COLUMNS}")
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f'{name}.npy')
    if os.path.exists(cache) and not force:
        return np.load(cache)
    y = (data.col(name) != 0).astype(np.float32)
    out = _train_windowed_fm(data, windows, y, k=k, lr=lr, epochs=epochs, seed=seed)
    np.save(cache, out)
    return out
