"""Grade the selection rules on the pool from exp11.

Every rule sees only what it is entitled to see:
  greedy    official-fold validation primary only
  transfer  mean test primary over the BACKTEST folds only (public-label region)
  robust    transfer minus instability penalty minus winner's-curse shrinkage
  oracle    official-fold test primary (unattainable - it exists to measure regret)
"""
import sys, json, math; sys.path.insert(0,'.')
import numpy as np
from kairos.agent.selection import selection_bias

PATH = sys.argv[1] if len(sys.argv) > 1 else 'runs/exp11_selection.json'
res = json.load(open(PATH))
print(f"pool: {PATH}\n")
BACKTESTS = [f for f in next(iter(res.values())) if f.startswith('backtest')]
names = list(res)

rows = []
for n in names:
    r = res[n]
    if 'official' not in r: continue
    bt = [r[f]['test'] for f in BACKTESTS if f in r]
    rows.append({
        'name': n,
        'official_valid': r['official']['valid'],
        'official_test':  r['official']['test'],
        'gap':            r['official']['valid'] - r['official']['test'],
        'transfer':       float(np.mean(bt)) if bt else float('nan'),
        'stability':      float(np.std(bt)) if len(bt) > 1 else 0.0,
        'seed_std':       r['official']['valid_std'],
    })

n_c = len(rows)
for x in rows:
    sigma = max(x['seed_std'], 0.0008)
    x['robust'] = x['transfer'] - 1.0*x['stability'] - selection_bias(n_c, sigma)

print(f"{'candidate':<18} {'off.valid':>10} {'off.test':>9} {'gap':>8} "
      f"{'transfer':>9} {'stab':>7} {'robust':>8}")
print("-"*72)
for x in sorted(rows, key=lambda z: -z['official_valid']):
    print(f"{x['name']:<18} {x['official_valid']:>10.4f} {x['official_test']:>9.4f} "
          f"{x['gap']:>+8.4f} {x['transfer']:>9.4f} {x['stability']:>7.4f} "
          f"{x['robust']:>8.4f}")

def grade(key, label, reverse=True):
    pick = max(rows, key=lambda z: z[key])
    best = max(rows, key=lambda z: z['official_test'])
    return {'rule': label, 'picked': pick['name'],
            'test_obtained': pick['official_test'],
            'regret': best['official_test'] - pick['official_test']}

print(f"\n{'rule':<26} {'picks':<15} {'hidden-test obtained':>21} {'regret':>9}")
print("-"*74)
out = []
for key, label in (('official_valid','greedy (argmax validation)'),
                   ('transfer','transfer (backtest mean)'),
                   ('robust','robust (transfer-corrected)'),
                   ('official_test','oracle (unattainable)')):
    g = grade(key, label); out.append(g)
    print(f"{g['rule']:<26} {g['picked']:<18} {g['test_obtained']:>21.4f} "
          f"{g['regret']:>9.4f}")

print(f"\nbaseline FM hidden test: 0.5946")
gr = [o for o in out if o['rule'].startswith('greedy')][0]
rb = [o for o in out if o['rule'].startswith('robust')][0]
print(f"greedy vs robust on hidden test: {gr['test_obtained']:.4f} -> {rb['test_obtained']:.4f} "
      f"({rb['test_obtained']-gr['test_obtained']:+.4f})")

# is validation even usable as a ranking signal over this pool?
v = np.array([x['official_valid'] for x in rows]); t = np.array([x['official_test'] for x in rows])
tr_ = np.array([x['transfer'] for x in rows])
def sp(a,b):
    ra=np.argsort(np.argsort(a)); rb=np.argsort(np.argsort(b))
    return float(np.corrcoef(ra,rb)[0,1])
print(f"\nrank correlation with hidden test over the pool:")
print(f"  official validation : {sp(v,t):+.3f}")
print(f"  backtest transfer   : {sp(tr_,t):+.3f}")
json.dump({'rows':rows,'rules':out}, open(PATH.replace('.json','_graded.json'),'w'),
          indent=2, default=float)
