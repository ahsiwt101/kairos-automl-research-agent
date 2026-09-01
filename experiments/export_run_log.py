"""Render a ledger into the per-iteration run log required by Deliverable 3.

Required per iteration: the hypothesis and why, the code diff applied, the resulting
metrics (GAUC / nDCG@5), and any error / recovery events. Plus a summary of manual
interventions, tokens, wall-clock and iterations used.

Everything here is READ from runs/*/ledger.jsonl - nothing is recomputed or re-worded, so
the log cannot drift from what the run actually did.
"""
import json, os, sys

def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]

def render(run_dir, out_path, title, notes=''):
    led = os.path.join(run_dir, 'ledger.jsonl')
    entries = load(led)
    errs = {}
    ep = os.path.join(run_dir, 'ledger_errors.jsonl')
    if os.path.exists(ep):
        for e in load(ep):
            errs.setdefault(e.get('iteration'), []).append(e)

    L = [f'# {title}', '']
    if notes:
        L += [notes, '']
    n_int = sum(int(e.get('interventions') or 0) for e in entries)
    L += ['## Summary', '',
          f'- **Iterations:** {len(entries)} (cap 50)',
          f'- **Manual interventions:** {n_int}']
    last = entries[-1] if entries else {}
    b = last.get('budget') or {}
    if b:
        L += [f"- **Wall-clock:** {b.get('wall_clock_s','?')} s",
              f"- **Tokens (in / out):** {b.get('tokens_in','?'):,} / {b.get('tokens_out','?'):,}"]
    L += ['- **GPU-hours:** 0 (CPU only)', '']

    for e in entries:
        h, o = e.get('hypothesis') or {}, e.get('outcome') or {}
        it = e.get('iteration')
        L += [f"## Iteration {it} — {e.get('decision','?').upper()}", '',
              f"**Family:** `{h.get('family','?')}`  ",
              f"**Hypothesis:** {h.get('statement','')}", '',
              f"**Mechanism (why it should work):** {h.get('mechanism','')}", '']
        pred = h.get('prediction')
        if pred:
            hit = o.get('prediction_hit')
            mark = {True: 'HIT', False: 'MISS', None: 'unverifiable'}.get(hit, str(hit))
            L += [f"**Falsifiable prediction:** `{json.dumps(pred)}` → **{mark}**", '']
        vp, vg, vn = o.get('valid_primary'), o.get('valid_gauc'), o.get('valid_ndcg')
        if vp == vp and vp is not None:
            L += ['**Metrics (validation):**', '',
                  '| GAUC | nDCG@5 | primary | Δ vs incumbent |',
                  '|---|---|---|---|',
                  f"| {vg:.4f} | {vn:.4f} | **{vp:.4f}** | {o.get('delta_vs_incumbent', 0):+.4f} |", '']
        else:
            L += ['**Metrics:** none — iteration produced no validation score.', '']
        if e.get('reason'):
            L += [f"**Decision rationale:** {e['reason']}", '']
        for er in errs.get(it, []):
            L += [f"**Error / recovery — `{er.get('kind','?')}`**  ",
                  f"{er.get('detail','')}  ",
                  f"*Recovery:* {er.get('recovery','')}", '']
        diff = e.get('code_diff') or ''
        if diff:
            L += ['<details><summary>Code applied this iteration</summary>', '',
                  '```python', diff.strip(), '```', '', '</details>', '']
        L += ['---', '']

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    open(out_path, 'w').write('\n'.join(L))
    print(f'  {out_path}: {len(entries)} iterations, {n_int} interventions')

PURE_NOTES = (
    'Per-iteration log required by Deliverable 3, generated from the run ledger by\n'
    '`experiments/export_run_log.py`. This is the campaign that produced the submitted\n'
    '`submission.csv`. Convergence rule declared before the run per FAQ 2.9.1:\n'
    'eps = 0.002, N = 5, minimum-iteration floor = 10. The run ended on the self-imposed\n'
    '150k token budget rather than on that rule — see `reports/RESULTS.md`.')

ONEK_NOTES = (
    'Bonus benchmark. The same agent, unchanged, pointed at KuaiRand-1k — trained on 1k\'s\n'
    'own splits only, per FAQ 2.9.2. Single-seed by design; see `reports/RESULTS_1K.md`\n'
    'for why, and for the current status of this run.')

if __name__ == '__main__':
    # Defaults are the SUBMITTED campaign. This previously pointed at runs/kairos_live,
    # which is a different (earlier) run - regenerating would have silently replaced the
    # submitted Deliverable 3 log with the wrong campaign's iterations.
    args = sys.argv[1:]
    if args:
        render(args[0], args[1] if len(args) > 1 else 'reports/ITERATION_LOG.md',
               args[2] if len(args) > 2 else 'KAIROS run log')
        raise SystemExit(0)

    render('runs/kairos_submission_repro', 'reports/ITERATION_LOG.md',
           'KAIROS run log — KuaiRand-Pure (required benchmark, submitted campaign)',
           PURE_NOTES)
    if os.path.exists('runs/kairos_1k/ledger.jsonl'):
        render('runs/kairos_1k', 'reports/ITERATION_LOG_1K.md',
               'KAIROS run log — KuaiRand-1k (bonus benchmark, transfer probe)',
               ONEK_NOTES)
