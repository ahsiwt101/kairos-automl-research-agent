"""Deliverable 4: re-score the shipped submission.csv with the OFFICIAL evaluate.py.

Everything else in this repo reports numbers that came out of the agent's own run. This
script is the independent check on the one number that decides the ranking: it reads
submission.csv off disk, re-derives the test split with the organizers' data.load(), and
scores it with the organizers' evaluate.py - our fast kernel is not used anywhere here.

A judge should be able to run exactly this and get exactly the table in reports/RESULTS.md.

    ./.venv/bin/python experiments/verify_submission.py

Exits non-zero if alignment fails or if any metric drifts from the reported value, so it
doubles as a regression test on the submission file itself.
"""
import csv, os, sys

sys.path.insert(0, '.')
import data                                   # noqa: E402  (official, unmodified)
import evaluate                                # noqa: E402  (official, unmodified)

DATA_DIR = os.environ.get('KUAIRAND_PURE', 'KuaiRand-Pure/data')
SUBMISSION = sys.argv[1] if len(sys.argv) > 1 else 'submission.csv'

# Published official-baseline hidden-test scores (mean over 5 seeds), from the problem
# statement. reports/RESULTS.md quotes these; experiments/verify_baseline.py reproduces
# them from the shipped baseline.py rather than trusting the printed figures.
BASELINE = {'GAUC': 0.6610, 'nDCG@5': 0.5282, 'primary': 0.5946}

# What this repo claims. Kept here so a drift in the submission file fails loudly.
REPORTED = {'GAUC': 0.6653, 'nDCG@5': 0.5313, 'primary': 0.5983}
TOL = 5e-5


def main():
    test = data.load(DATA_DIR)['test']
    # data.load returns tuples: (date, user_id, video_id, author_id, tab, duration_ms, label)
    uids = [r[1] for r in test]
    vids = [r[2] for r in test]
    labels = [r[6] for r in test]

    rows = list(csv.DictReader(open(SUBMISSION)))
    print(f'{SUBMISSION}: {len(rows):,} rows | test split: {len(test):,} rows')

    # Alignment is positional and (user_id, video_id) is NOT unique - 3.06% of test rows
    # are repeated pairs - so row_id is the only key. Check all three anyway.
    ok = True
    if len(rows) != len(test):
        print('  FAIL row count mismatch'); ok = False
    if [int(r['row_id']) for r in rows] != list(range(len(rows))):
        print('  FAIL row_id is not a 0-based strictly increasing index'); ok = False
    if not all(r['user_id'] == u for r, u in zip(rows, uids)):
        print('  FAIL user_id misaligned against the test split'); ok = False
    if not all(r['video_id'] == v for r, v in zip(rows, vids)):
        print('  FAIL video_id misaligned against the test split'); ok = False
    if not ok:
        return 1
    print('  alignment OK (row_id, user_id, video_id all match data.load order)')

    scores = [float(r['score']) for r in rows]
    m = evaluate.evaluate(uids, labels, scores)

    # The drift check pins the SUBMITTED file against the values this repo reports. Scoring
    # any other file (e.g. reference_handbuilt_ensemble.csv) is a legitimate use of this
    # script, so don't fail it for not being the submission.
    is_submission = os.path.basename(SUBMISSION) == 'submission.csv'

    print(f'\n  scored with the official evaluate.py over {m["users"]:,} users\n')
    print(f'  {"metric":9s} {"baseline":>9s} {"scored":>9s} {"delta":>9s}')
    drift = False
    for k in ('GAUC', 'nDCG@5', 'primary'):
        print(f'  {k:9s} {BASELINE[k]:9.4f} {m[k]:9.4f} {m[k]-BASELINE[k]:+9.4f}')
        if is_submission and abs(m[k] - REPORTED[k]) > TOL:
            drift = True
    # score_dataset per the judging formula: mean of the per-metric absolute deltas.
    # primary is excluded - it is itself the mean of GAUC and nDCG@5, so including it
    # would double-count.
    sd = ((m['GAUC'] - BASELINE['GAUC']) + (m['nDCG@5'] - BASELINE['nDCG@5'])) / 2
    print(f'\n  score_dataset = mean(delta GAUC, delta nDCG@5) = {sd:+.4f}')

    if drift:
        print('\n  FAIL a metric drifted from the value reported in reports/RESULTS.md')
        return 1
    if is_submission:
        print('  matches the reported result to within 5e-5')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
