"""Where are the points actually lost?  Diagnose the baseline FM instead of guessing."""
import sys, json; sys.path.insert(0,'.')
import numpy as np
from kairos.kernel.dataset import Data
from kairos.kernel.features import Encoder
from kairos.kernel.diagnostics import Diagnostics
from kairos.models.train import train_fm, predict
from kairos.kernel.fastmetrics import fast_evaluate, factorize

d = Data(); fold = d.fold('official')
enc = Encoder(d).fit(fold.idx['train'])
r = train_fm(fold, enc, loss='bce', seed=0)
sv = predict(r['model'], enc, fold.idx['valid'])
dg = Diagnostics(fold, 'valid', sv)

print("=== overall ===")
dig = dg.digest()
for k in ('primary','GAUC','nDCG@5','oracle_primary','headroom_total','users'):
    print(f"  {k:18s} {dig[k]}")

print("\n=== headroom by slice (top buckets by recoverable primary) ===")
for name in dg.user_slices:
    print(f"\n-- {name} --")
    t = dg.headroom_table(name)
    print(f"  {'bucket':>7} {'users':>7} {'ndcg':>7} {'auc':>7} {'held':>8} {'ceil':>8} {'headroom':>9}")
    for row in t:
        print(f"  {row['bucket']:>7} {row['users']:>7} {row['ndcg_mean']:>7.3f} "
              f"{row['auc_mean']:>7.3f} {row['primary_held']:>8.4f} "
              f"{row['primary_ceiling']:>8.4f} {row['headroom']:>9.4f}")

print("\n=== inversion attribution: which items wrongly beat a positive? ===")
for a in ('duration_decile','item_pop_decile','item_cold','tab'):
    inv = dg.inversions(a)
    print(f"\n-- {a} -- total GAUC lost to inversions: {inv['total_gauc_loss']:.4f}")
    for b,v in list(inv['by_bucket'].items())[:10]:
        print(f"   bucket {b:>3}: {v:.4f}")

# how much of the metric does a trivial item-popularity scorer already capture?
tr = fold.idx['train']; va = fold.idx['valid']
cnt = np.bincount(d.video_id[tr], minlength=int(d.video_id.max())+1)
pos = np.bincount(d.video_id[tr], weights=d.y_raw[tr].astype(float),
                  minlength=int(d.video_id.max())+1)
gm = d.y_raw[tr].mean()
poprate = (pos + 20*gm)/(cnt + 20)
gva,_ = factorize(d.user_id[va]); yva = d.y_raw[va]
print("\n=== reference points on valid ===")
print("  item popularity  ", {k: round(v,4) for k,v in
      fast_evaluate(gva, yva, poprate[d.video_id[va]]).items() if k in ('GAUC','nDCG@5','primary')})
print("  FM               ", {k: round(v,4) for k,v in
      fast_evaluate(gva, yva, sv).items() if k in ('GAUC','nDCG@5','primary')})
from scipy.stats import spearmanr
print(f"  spearman(FM, item_pop) on valid rows = {spearmanr(sv, poprate[d.video_id[va]]).statistic:.4f}")
