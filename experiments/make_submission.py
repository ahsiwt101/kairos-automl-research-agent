"""Write and validate the final submission.

Alignment is positional and (user_id, video_id) is NOT unique in the evaluation split
(3.06% of test rows are repeated pairs, up to 12 times), so row order is the only key.
We write through the official submit.py helpers and then re-validate with the official
checker, rather than trusting our own writer.
"""
import sys, argparse; sys.path.insert(0,'.')
import numpy as np
from data import load as official_load
from submit import write_submission, read_submission, HEADER
from evaluate import evaluate

ap = argparse.ArgumentParser()
ap.add_argument('--scores', required=True, help='.npy of scores aligned to the split')
ap.add_argument('--split', default='test', choices=['valid','test'])
ap.add_argument('--out', default='submission.csv')
ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
a = ap.parse_args()

splits = official_load(a.data_dir)
rows = splits[a.split]
s = np.load(a.scores).astype(np.float64)
assert len(s) == len(rows), f"{len(s)} scores for {len(rows)} {a.split} rows"
assert np.isfinite(s).all(), "scores contain NaN/Inf - the official checker rejects these"

# nDCG has no tie correction (the reference sort is stable), so tied scores are broken by
# row order.  Only ties WITHIN a user matter - the metric never compares across users - so
# count those; a global duplicate count is alarming and meaningless here.
uu = np.array([x[1] for x in rows])
o = np.lexsort((np.arange(len(s)), s, uu))
n_tied = int(((uu[o][1:] == uu[o][:-1]) & (s[o][1:] == s[o][:-1])).sum())
if n_tied:
    print(f"note: {n_tied:,} of {len(s):,} rows tie with a list-mate; nDCG breaks these by "
          f"row order (measured cost on valid: ~5e-05 primary)")

write_submission(a.out, rows, s)
back = read_submission(a.out, rows)          # official validator
assert np.allclose(np.array(back), s, rtol=0, atol=1e-6), "round-trip mismatch"
print(f"wrote {a.out}: {len(rows):,} rows, split={a.split}, validated by official reader")
if a.split == 'valid':
    r = evaluate([x[1] for x in rows], [x[6] for x in rows], back)
    print(f"  local score  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | "
          f"primary {r['primary']:.4f}")
