"""Self-audit: assumptions that would corrupt results SILENTLY if wrong.

Each of these is load-bearing for a claim I have already made, and none of them would
show up as an error - they would just quietly produce wrong numbers.
"""
import sys, json; sys.path.insert(0,'.')
import numpy as np
from kairos.kernel.dataset import Data
d = Data()
fails = []
def check(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok: fails.append(name)

print("=== A. side tables are indexed by DENSE id (I index vb[video_id] directly) ===")
vb_vid = d.col('video_id','vb')
check("video_id dense 0..N-1 in video_features_basic",
      bool(np.array_equal(vb_vid, np.arange(len(vb_vid)))),
      f"n={len(vb_vid)}, min={vb_vid.min():.0f}, max={vb_vid.max():.0f}")
check("all log video_ids addressable in vb",
      bool(d.video_id.max() < len(vb_vid)),
      f"log max video_id={d.video_id.max()} vs vb rows={len(vb_vid)}")
uf_uid = d.col('user_id','uf')
check("user_id dense 0..N-1 in user_features",
      bool(np.array_equal(uf_uid, np.arange(len(uf_uid)))),
      f"n={len(uf_uid)}, max={uf_uid.max():.0f}")
check("all log user_ids addressable in uf",
      bool(d.user_id.max() < len(uf_uid)),
      f"log max user_id={d.user_id.max()} vs uf rows={len(uf_uid)}")

print("\n=== B. composite key packing cannot collide ===")
vb_author = d.col('author_id','vb')
amax = float(np.nanmax(vb_author))
check("author_id < 1e7 (packed as uid*1e7 + author)", amax < 1e7,
      f"max author_id={amax:.0f}  {'SAFE' if amax<1e7 else 'COLLISION RISK'}")
check("video_id < 1e5 (packed as uid*1e5 + video)", d.video_id.max() < 1e5,
      f"max video_id={d.video_id.max()}")
check("tab < 100 (packed as uid*100 + tab)", d.col('tab').max() < 100,
      f"max tab={d.col('tab').max()}")
umax = int(d.user_id.max())
check("uid*1e7+author fits in int64", umax*1e7 + amax < 9.2e18,
      f"max packed ~{umax*1e7+amax:.3e}")

print("\n=== C. frozen_prefix date packing (kid*1e8 + date) ===")
n_keys_max = int(d.user_id.max())*10_000_000 + int(amax)
check("date < 1e8", int(d.date.max()) < 1e8, f"max date={d.date.max()}")
print(f"       note: frozen_prefix factorizes keys first, so kid < n_unique_keys, not {n_keys_max:.2e}")

print("\n=== D. fast_evaluate vs official evaluate.py on REAL model scores ===")
from evaluate import evaluate as ref_eval
from kairos.kernel.fastmetrics import fast_evaluate, factorize
fold = d.fold('official')
worst = 0.0
for name in ('runs/sv_fm.npy','runs/gb_s0_va.npy','runs/fm_tau7_s0_va.npy'):
    try: s = np.load(name)
    except Exception: continue
    idx = fold.idx['valid']
    u = d.user_id[idx]; y = d.y_raw[idx]
    r = ref_eval(list(u), list(y), list(s))
    g,_ = factorize(u); f = fast_evaluate(g, y, s)
    dd = max(abs(r[k]-f[k]) for k in ('GAUC','nDCG@5','primary'))
    worst = max(worst, dd)
    print(f"  {name:28s} ref {r['primary']:.9f}  fast {f['primary']:.9f}  diff {dd:.2e}")
if worst: check("fast==reference on real scores", worst < 1e-12, f"worst diff {worst:.2e}")

print("\n=== E. how many times have we consulted the sealed test set? ===")
try:
    lines = [json.loads(l) for l in open('runs/scorer_audit.log')]
    te = [l for l in lines if l['split'].endswith('/test')]
    print(f"  official-fold test scorer calls: {len(te)}")
    print("  (research probes; the agent's own selection must use valid/backtest only)")
except FileNotFoundError:
    print("  no audit log yet")

print("\n" + ("ALL ASSUMPTIONS HOLD" if not fails else f"FAILURES: {fails}"))
