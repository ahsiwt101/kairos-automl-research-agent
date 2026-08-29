"""Training loop shared by every objective, so ablations differ only in the loss."""
import time
import numpy as np
import torch
from kairos.kernel.fastmetrics import fast_evaluate, factorize
from kairos.models.ranker import FM, build_groups, pad_groups, compute_loss


def train_fm(fold, enc, loss='bce', k=16, lr=1e-3, epochs=40, patience=4, seed=0,
             group_key='user_week', max_len=32, batch_groups=256, batch_rows=8192,
             l2=1e-6, device='cpu', verbose=False, eval_on='valid', train_parts=('train',),
             recency_tau=None):
    """Returns dict with best valid metrics, the trained model, and the epoch curve."""
    torch.manual_seed(seed)
    d = fold.data
    tr_idx = np.concatenate([fold.idx[p] for p in train_parts])
    va_idx = fold.idx[eval_on]

    Xtr = enc.transform(tr_idx)
    ytr = d.y_raw[tr_idx].astype(np.float32)
    Xva = enc.transform(va_idx)
    gva, _ = factorize(d.user_id[va_idx])
    yva = d.y_raw[va_idx]

    model = FM(enc.dim, k=k, seed=seed).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    Xva_t = torch.from_numpy(Xva.astype(np.int64)).to(device)

    pointwise = (loss == 'bce')
    if pointwise:
        Xtr_t = torch.from_numpy(Xtr.astype(np.int64)).to(device)
        ytr_t = torch.from_numpy(ytr).to(device)
        # Training is dominated by a dense logging regime (59% of rows in 3 days) that the
        # sparse evaluation window does not resemble; exponential recency weighting shifts
        # the effective training distribution toward the regime actually being served.
        if recency_tau:
            from kairos.kernel.frozenfeat import _dayindex
            age = (_dayindex(fold.spec['train'][1]) - _dayindex(d.date[tr_idx])).astype(np.float32)
            w = np.exp(-age / float(recency_tau))
            wtr_t = torch.from_numpy((w / w.mean()).astype(np.float32)).to(device)
        else:
            wtr_t = None
        n = len(ytr)
    else:
        rows, sizes = build_groups(d.user_id[tr_idx], d.date[tr_idx], key=group_key,
                                   max_len=max_len, seed=seed)
        Xp, yp, mk = pad_groups(rows, Xtr, ytr, max_len)
        # groups the metric ignores (all-positive / all-negative) carry no ranking signal
        live = ((yp * mk).sum(1) > 0) & (((1 - yp) * mk).sum(1) > 0)
        Xp, yp, mk = Xp[live], yp[live], mk[live]
        Xp_t = torch.from_numpy(Xp).to(device)
        yp_t = torch.from_numpy(yp).to(device)
        mk_t = torch.from_numpy(mk).to(device)
        n = len(Xp)

    rng = np.random.default_rng(seed)
    best, best_state, bad, curve = -1.0, None, 0, []
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        perm = rng.permutation(n)
        bs = batch_rows if pointwise else batch_groups
        tot, nb = 0.0, 0
        for i in range(0, n, bs):
            sel = torch.from_numpy(perm[i:i + bs]).to(device)
            opt.zero_grad()
            if pointwise:
                s = model(Xtr_t[sel])
                if wtr_t is None:
                    lo = torch.nn.functional.binary_cross_entropy_with_logits(s, ytr_t[sel])
                else:
                    lo = torch.nn.functional.binary_cross_entropy_with_logits(
                        s, ytr_t[sel], weight=wtr_t[sel])
            else:
                s = model(Xp_t[sel])
                lo = compute_loss(loss, s, yp_t[sel], mk_t[sel])
            lo.backward()
            opt.step()
            tot += float(lo.detach()); nb += 1
        model.eval()
        with torch.no_grad():
            sv = model(Xva_t).cpu().numpy().astype(np.float64)
        m = fast_evaluate(gva, yva, sv)
        curve.append({'epoch': ep, 'loss': tot / max(nb, 1), **{kk: m[kk] for kk in
                     ('GAUC', 'nDCG@5', 'primary')}, 'sec': round(time.time() - t0, 1)})
        if verbose:
            print(f"  ep{ep:3d} loss {tot/max(nb,1):.4f} | valid GAUC {m['GAUC']:.4f} "
                  f"nDCG@5 {m['nDCG@5']:.4f} primary {m['primary']:.4f} | {time.time()-t0:.1f}s")
        if m['primary'] > best + 1e-5:
            best, bad = m['primary'], 0
            best_state = {kk: v.detach().clone() for kk, v in model.state_dict().items()}
            best_metrics = m
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return {'model': model, 'valid': best_metrics, 'curve': curve, 'epochs_run': len(curve),
            'best_epoch': int(np.argmax([c['primary'] for c in curve])) + 1}


def predict(model, enc, idx, device='cpu', bs=200_000):
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            X = torch.from_numpy(enc.transform(idx[i:i + bs]).astype(np.int64)).to(device)
            out.append(model(X).cpu().numpy().astype(np.float64))
    return np.concatenate(out)
