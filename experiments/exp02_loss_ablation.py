"""Controlled loss ablation: identical features (the baseline's 5 fields), identical
capacity (k=16), identical optimiser. Only the objective changes.

Discipline note: SELECTION is made on validation only. The test column is recorded to
study the validation->test gap (that gap is the research question), never to choose.
"""
import sys, json, time; sys.path.insert(0,'.')
import numpy as np
from kairos.kernel.dataset import Data
from kairos.kernel.features import Encoder
from kairos.models.train import train_fm, predict

d = Data(); fold = d.fold('official')
enc = Encoder(d).fit(fold.idx['train'])
rows = []
LOSSES = ['bce', 'bpr', 'bpr_gauc', 'listnet', 'lambda_ndcg', 'primary']
print(f"{'loss':<13} {'grp':<12} {'ep':>3} {'valid GAUC':>11} {'valid nDCG':>11} "
      f"{'valid prim':>11} {'test prim':>10} {'gap':>8} {'sec':>6}")
for loss in LOSSES:
    for gk in (['user_week'] if loss == 'bce' else ['user_week']):
        t0 = time.time()
        r = train_fm(fold, enc, loss=loss, group_key=gk, seed=0)
        st = predict(r['model'], enc, fold.idx['test'])
        m = fold.scorers['test'].score(st, reason=f'exp02 ablation {loss}/{gk}')
        v = r['valid']
        rows.append({'loss': loss, 'group': gk, 'best_epoch': r['best_epoch'],
                     'valid': v, 'test': m, 'sec': round(time.time()-t0, 1)})
        print(f"{loss:<13} {gk:<12} {r['best_epoch']:>3} {v['GAUC']:>11.4f} "
              f"{v['nDCG@5']:>11.4f} {v['primary']:>11.4f} {m['primary']:>10.4f} "
              f"{v['primary']-m['primary']:>8.4f} {time.time()-t0:>6.0f}")
json.dump(rows, open('runs/exp02_loss_ablation.json','w'), indent=2, default=float)
print("\nbaseline reference: valid 0.6016 / test 0.5946")
