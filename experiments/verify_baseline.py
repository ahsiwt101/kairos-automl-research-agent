"""Scrutinise the reproduction claim properly.

Earlier I compared SINGLE seeds against a published 5-seed mean and called the difference
noise.  That is an assertion, not a measurement.  The published protocol is mean over
seeds 0-4 with std 0.0008, so reproduce that protocol and compare distributions.

Also checks the thing single-number comparisons cannot catch: whether my cached kernel
produces the same ROWS IN THE SAME ORDER as the official data.load().  Submission
alignment is positional and (user_id, video_id) is not unique, so an ordering bug would be
invisible in aggregate metrics and fatal at submission time.
"""
import sys, time, statistics; sys.path.insert(0,'.')
import numpy as np
sys.path.insert(0, '.')

# ---------- 1. row-order equivalence vs the official loader -------------------
print("=== row-order equivalence: official data.load() vs kairos Data ===")
from data import load as official_load
from kairos.kernel.dataset import Data
t0=time.time()
osp = official_load('./KuaiRand-Pure/data')
d = Data(); fold = d.fold('official')
for part in ('train','valid','test'):
    rows = osp[part]
    idx = fold.idx[part]
    assert len(rows) == len(idx), f"{part}: {len(rows)} vs {len(idx)}"
    ou = np.array([int(x[1]) for x in rows]); ov = np.array([int(x[2]) for x in rows])
    oy = np.array([x[6] for x in rows])
    mu = d.user_id[idx].astype(int); mv = d.video_id[idx].astype(int)
    my = d.y_raw[idx].astype(int)
    su = int((ou!=mu).sum()); sv_=int((ov!=mv).sum()); sy=int((oy!=my).sum())
    print(f"  {part:5s} n={len(rows):>9,}  user_id mismatches={su}  "
          f"video_id mismatches={sv_}  label mismatches={sy}")
    assert su==0 and sv_==0 and sy==0, f"{part} ROW ORDER OR CONTENT DIFFERS"
print(f"  -> identical rows in identical order ({time.time()-t0:.0f}s)")

# ---------- 2. official numpy FM, seeds 0-4 (the published protocol) ----------
print("\n=== official numpy FM, seeds 0-4 (published: valid 0.6016 / test 0.5946, std 0.0008) ===")
import baseline as B
res=[]
for seed in range(5):
    r = B.run_fm(osp, k=16, lr=0.001, epochs=40, seed=seed, verbose=False)
    res.append((r['valid']['primary'], r['test']['primary'],
                r['test']['GAUC'], r['test']['nDCG@5']))
    print(f"  seed {seed}: valid {r['valid']['primary']:.4f}  test {r['test']['primary']:.4f}")
vm = statistics.mean(x[0] for x in res); tm = statistics.mean(x[1] for x in res)
ts = statistics.pstdev([x[1] for x in res])
tg = statistics.mean(x[2] for x in res); tn = statistics.mean(x[3] for x in res)
print(f"  MEAN valid {vm:.4f} (published 0.6016, diff {vm-0.6016:+.4f})")
print(f"  MEAN test  {tm:.4f} (published 0.5946, diff {tm-0.5946:+.4f})  std {ts:.4f} "
      f"(published 0.0008)")
print(f"  MEAN test GAUC {tg:.4f} (pub 0.6610)  nDCG@5 {tn:.4f} (pub 0.5282)")

# ---------- 3. random baseline, seeds 0-4 -------------------------------------
print("\n=== random, seeds 0-4 (published test 0.4753) ===")
rs=[B.run_random(osp, seed=s)['test']['primary'] for s in range(5)]
print(f"  MEAN test {statistics.mean(rs):.4f} (published 0.4753, "
      f"diff {statistics.mean(rs)-0.4753:+.4f})")

# ---------- 4. my torch reimplementation, same seeds --------------------------
print("\n=== torch FM reimplementation, seeds 0-4 ===")
from kairos.kernel.features import Encoder
from kairos.models.train import train_fm, predict
enc = Encoder(d).fit(fold.idx['train'])
tv, tt = [], []
for seed in range(5):
    r = train_fm(fold, enc, loss='bce', seed=seed)
    st = predict(r['model'], enc, fold.idx['test'])
    m = fold.scorers['test'].score(st, reason=f'verify_baseline torch seed{seed}')
    tv.append(r['valid']['primary']); tt.append(m['primary'])
    print(f"  seed {seed}: valid {r['valid']['primary']:.4f}  test {m['primary']:.4f}")
print(f"  MEAN valid {statistics.mean(tv):.4f}  test {statistics.mean(tt):.4f} "
      f"std {statistics.pstdev(tt):.4f}")
print(f"\n  numpy vs torch test-mean difference: {statistics.mean(tt)-tm:+.4f} "
      f"(combined seed std ~{max(ts,statistics.pstdev(tt)):.4f})")
