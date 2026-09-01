# Archived: first KuaiRand-1k attempt

The first 1k campaign used `runs/kairos_1k/` as its workdir, and the current campaign
reuses that directory, so this attempt's artifacts were overwritten in place. What
survives:

- `cand1_meta.json` — the one candidate metadata blob recoverable from the pre-rewrite
  git history (415.1 s, `mode='scores'`)
- `../live_1k_attempt1.log` — the attempt's console output, committed separately

That attempt rejected every large-gain candidate. The cause was diagnosed afterwards and
was not what it looked like: prewarm did not cover the *confirmation* fold, so each
backtest confirmation rebuilt two windowed FMs over 11.7M rows inside the candidate
sandbox and blew its timeout. It was misread at the time as the verifier being
intrinsically too expensive on this variant. See `reports/RESULTS_1K.md`.
