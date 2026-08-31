"""What does the strict reading of FAQ 2.9.2 cost?

Measured on BACKTEST folds, whose test windows lie entirely inside the public-label region
(<= 20220428). This quantifies the trade-off without consulting the official hidden test -
which we must not do, and which would itself be model selection on test labels.

strict     : every model fits on the fold's train split only.
permissive : a model scoring the fold's TEST window may also fit on its validation window
             (which closes before that window opens). Never touches a test label.
"""
import os, subprocess, sys, json

PROBE = """
import sys; sys.path.insert(0,'.')
import numpy as np, json
from kairos.kernel.dataset import load
from kairos.kernel.refit_signal import build_refit_signal
from kairos.kernel.fastmetrics import fast_evaluate, factorize
d = load(); out = {}
for fold_name in ('backtest_a', 'backtest_b'):
    f = d.fold(fold_name)
    s = build_refit_signal(d, f)
    r = {}
    for split in ('valid', 'test'):
        i = f.idx[split]; g, _ = factorize(d.user_id[i])
        r[split] = fast_evaluate(g, d.y_raw[i], s[i])['primary']
    out[fold_name] = r
print('RESULT ' + json.dumps(out))
"""

res = {}
for mode, val in (('strict', '1'), ('permissive', '0')):
    env = dict(os.environ, KAIROS_STRICT_TRAIN_SPLIT=val, PYTHONWARNINGS='ignore')
    r = subprocess.run([sys.executable, '-c', PROBE], capture_output=True, text=True, env=env)
    line = [l for l in r.stdout.splitlines() if l.startswith('RESULT ')]
    if not line:
        print(f'{mode} FAILED:\n{r.stderr[-1500:]}')
        sys.exit(1)
    res[mode] = json.loads(line[0][7:])
    print(f'  {mode} done', flush=True)

print()
print('  fold          split   strict  permissive   delta')
deltas = []
for fold in res['strict']:
    for split in ('valid', 'test'):
        a, b = res['strict'][fold][split], res['permissive'][fold][split]
        mark = '  <- the quantity in question' if split == 'test' else ''
        print(f'  {fold:12s} {split:6s} {a:.4f}   {b:.4f}   {b-a:+.4f}{mark}')
        if split == 'test':
            deltas.append(b - a)
print()
print(f'  mean test delta from the permissive reading: {sum(deltas)/len(deltas):+.4f}')
json.dump(res, open('runs/exp28_strict_vs_permissive.json', 'w'), indent=1)
