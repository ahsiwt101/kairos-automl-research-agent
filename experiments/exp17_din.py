"""Sequence modelling: does target attention over user history beat the aggregates?

The organizers list this as completely unexplored, and our behavioural features so far are
aggregates - a long_view RATE summarises a user's history but forgets which items it was.
DIN asks the sharper question: how similar is this candidate to what this user actually
watched?

Leakage discipline is unchanged: a row's sequence contains only items long-viewed at or
before its FROZEN WINDOW HORIZON, so it can never see its own list-mates.
"""
import sys, time, json; sys.path.insert(0,'.')
import numpy as np
import torch
from kairos.kernel.dataset import Data
from kairos.kernel.features import Encoder
from kairos.kernel.causal import window_horizons
from kairos.kernel.frozenfeat import OFFICIAL_WINDOWS
from kairos.kernel.fastmetrics import fast_evaluate, factorize
from kairos.models.din import DIN, build_sequences

MAXLEN = 32
d = Data(); fold = d.fold('official')
tr, va, te = fold.idx['train'], fold.idx['valid'], fold.idx['test']
hz = window_horizons(d.date.astype(np.int64), OFFICIAL_WINDOWS)

t0 = time.time()
seq, slen = build_sequences(d, hz, max_len=MAXLEN)
print(f"sequences built in {time.time()-t0:.0f}s | mean history length "
      f"{slen.mean():.1f} | rows with no history {100*(slen==0).mean():.1f}%")
for nm, idx in (('train',tr),('valid',va),('test',te)):
    print(f"  {nm:6s} mean len {slen[idx].mean():5.1f}  empty {100*(slen[idx]==0).mean():5.1f}%")

enc = Encoder(d).fit(tr)
Xtr = enc.transform(tr); Xva = enc.transform(va); Xte = enc.transform(te)
ytr = d.y_raw[tr].astype(np.float32); yva = d.y_raw[va]
gva,_ = factorize(d.user_id[va])
n_items = int(d.video_id.max())+1

def T(x, dt=torch.long): return torch.as_tensor(x, dtype=dt)
Xtr_t, Xva_t, Xte_t = T(Xtr), T(Xva), T(Xte)
itr, iva, ite = T(d.video_id[tr]+1), T(d.video_id[va]+1), T(d.video_id[te]+1)
str_, sva, ste = T(seq[tr]), T(seq[va]), T(seq[te])
ytr_t = torch.as_tensor(ytr)

res = {}
for seed in (0,1,2):
    m = DIN(n_items, enc.dim, k=32, hidden=64, seed=seed)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0
    for ep in range(1, 16):
        m.train(); perm = rng.permutation(len(ytr)); t1=time.time(); tot=nb=0
        for i in range(0, len(perm), 4096):
            s_ = torch.as_tensor(perm[i:i+4096])
            opt.zero_grad()
            out = m(Xtr_t[s_], itr[s_], str_[s_])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(out, ytr_t[s_])
            loss.backward(); opt.step(); tot += float(loss.detach()); nb += 1
        m.eval()
        with torch.no_grad():
            pv = np.concatenate([m(Xva_t[i:i+50000], iva[i:i+50000], sva[i:i+50000]).numpy()
                                 for i in range(0, len(iva), 50000)])
        mt = fast_evaluate(gva, yva, pv.astype(np.float64))
        print(f"  seed{seed} ep{ep:2d} loss {tot/nb:.4f} valid {mt['primary']:.4f} "
              f"({time.time()-t1:.0f}s)", flush=True)
        if mt['primary'] > best + 1e-5:
            best, bad = mt['primary'], 0
            best_state = {k: v.detach().clone() for k,v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= 3: break
    m.load_state_dict(best_state); m.eval()
    with torch.no_grad():
        pv = np.concatenate([m(Xva_t[i:i+50000], iva[i:i+50000], sva[i:i+50000]).numpy()
                             for i in range(0, len(iva), 50000)])
        pt = np.concatenate([m(Xte_t[i:i+50000], ite[i:i+50000], ste[i:i+50000]).numpy()
                             for i in range(0, len(ite), 50000)])
    np.save(f'runs/din_s{seed}_va.npy', pv); np.save(f'runs/din_s{seed}_te.npy', pt)
    res[f's{seed}'] = best
    print(f"  seed{seed} best valid {best:.4f}")

vals = list(res.values())
print(f"\nDIN valid {np.mean(vals):.4f} +- {np.std(vals):.4f}  (FM 0.6017, GBDT 0.5992)")
json.dump(res, open('runs/exp17_din.json','w'), indent=2)
