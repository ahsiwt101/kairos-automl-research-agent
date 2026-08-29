"""Stage A (torch only): FM with recency weighting. Saves scores for later fusion.
Run in its own process - torch and lightgbm each link their own libomp and collide."""
import sys, json; sys.path.insert(0,'.')
import numpy as np
from kairos.kernel.dataset import Data
from kairos.kernel.features import Encoder
from kairos.kernel.fastmetrics import fast_evaluate, factorize
from kairos.models.train import train_fm, predict

d = Data(); fold = d.fold('official')
tr, va, te = fold.idx['train'], fold.idx['valid'], fold.idx['test']
gva,_ = factorize(d.user_id[va]); yva = d.y_raw[va]
enc = Encoder(d).fit(tr)
out = {}
for tau in (None, 14, 7, 5):
    for seed in (0,1,2):
        r = train_fm(fold, enc, loss='bce', seed=seed, recency_tau=tau)
        sv = predict(r['model'], enc, va); st = predict(r['model'], enc, te)
        key = f"tau{tau}_s{seed}"
        np.save(f'runs/fm_{key}_va.npy', sv); np.save(f'runs/fm_{key}_te.npy', st)
        out[key] = {'valid': r['valid']['primary'], 'epoch': r['best_epoch']}
        print(f"  {key:12s} valid {r['valid']['primary']:.4f} ep {r['best_epoch']}")
for tau in (None,14,7,5):
    ks=[k for k in out if k.startswith(f"tau{tau}_")]
    m=np.mean([out[k]['valid'] for k in ks]); s=np.std([out[k]['valid'] for k in ks])
    print(f"tau={str(tau):>5}  valid mean {m:.4f} +- {s:.4f}  (n={len(ks)} seeds)")
json.dump(out, open('runs/exp07a_fm.json','w'), indent=2)
