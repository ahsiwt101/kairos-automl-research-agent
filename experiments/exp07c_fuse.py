"""Stage C (numpy only): rank fusion. Weight chosen on VALIDATION only."""
import sys, json, glob; sys.path.insert(0,'.')
import numpy as np
from kairos.kernel.dataset import Data
from kairos.kernel.fastmetrics import fast_evaluate, factorize

d = Data(); fold = d.fold('official')
va, te = fold.idx['valid'], fold.idx['test']
gva,_ = factorize(d.user_id[va]); yva = d.y_raw[va]
uva, ute = d.user_id[va], d.user_id[te]

def wrank(scores, users):
    order = np.lexsort((np.arange(len(scores)), -np.asarray(scores,dtype=np.float64), users))
    u = users[order]
    starts = np.flatnonzero(np.r_[True, u[1:]!=u[:-1]])
    sizes = np.diff(np.r_[starts, len(u)]); seg = np.repeat(np.arange(len(starts)), sizes)
    pct = 1.0 - (np.arange(len(u)) - starts[seg]) / np.maximum(sizes[seg]-1, 1)
    out = np.empty(len(scores)); out[order] = pct; return out

fmk = json.load(open('runs/exp07a_fm.json'))
best_tau = max(set(k.split('_')[0] for k in fmk),
               key=lambda t: np.mean([fmk[k]['valid'] for k in fmk if k.startswith(t+'_')]))
print(f"FM variant selected on validation: {best_tau}")
fm_va = np.mean([wrank(np.load(f), uva) for f in sorted(glob.glob(f'runs/fm_{best_tau}_s*_va.npy'))],0)
fm_te = np.mean([wrank(np.load(f), ute) for f in sorted(glob.glob(f'runs/fm_{best_tau}_s*_te.npy'))],0)
gb_va = np.mean([wrank(np.load(f), uva) for f in sorted(glob.glob('runs/gb_s*_va.npy'))],0)
gb_te = np.mean([wrank(np.load(f), ute) for f in sorted(glob.glob('runs/gb_s*_te.npy'))],0)

ev = lambda s: fast_evaluate(gva, yva, s)
print(f"\nFM  (3-seed rank avg) valid {ev(fm_va)['primary']:.4f}")
print(f"GBDT(3-seed rank avg) valid {ev(gb_va)['primary']:.4f}")
best_w, best_p = 0, -1
for w in np.arange(0,1.001,0.02):
    p = ev(w*fm_va + (1-w)*gb_va)['primary']
    if p>best_p: best_p, best_w = p, w
print(f"best fusion weight w_fm={best_w:.2f} valid {best_p:.4f}")

res={}
for tag, sv_, st_ in (('FM 3-seed', fm_va, fm_te), ('GBDT 3-seed', gb_va, gb_te),
                      ('fusion', best_w*fm_va+(1-best_w)*gb_va, best_w*fm_te+(1-best_w)*gb_te)):
    mv=ev(sv_); mt=fold.scorers['test'].score(st_, reason=f'exp07c {tag}')
    res[tag]={'valid':mv,'test':mt}
    print(f"{tag:14s} valid {mv['primary']:.4f} | test {mt['primary']:.4f} | "
          f"gap {mv['primary']-mt['primary']:+.4f} | vs baseline test {mt['primary']-0.5946:+.4f}")
json.dump(res, open('runs/exp07c_fuse.json','w'), indent=2, default=float)
