## Why this run, and not the higher-scoring one

*Hand-written note, appended verbatim by `experiments/results_table.py`. Every figure it
cites is derived in the generated tables above.*

An earlier campaign (`runs/kairos_agent_submission`) reached validation 0.6034 and test
0.5988 — 0.0005 above this one, which is well inside the official baseline's own 5-seed
standard deviation of 0.0008.

This campaign was chosen **before either was scored on test**, on four grounds:

1. Its prior contains no test-derived content. The earlier run's prior cited hidden-test
   scores in four places and withdrew a technique *because its test score fell*, which is
   test-derived input to model selection.
2. Both of its accepted candidates were independently backtest-confirmed (gaps -0.0001 and
   -0.0012 against a 0.035 threshold, 0.053 clear of the honest ceiling). The earlier run's
   accepted candidate never was: confirmation only fired on implausible-looking gains, and
   +0.0018 did not qualify. Run after the fact
   (`experiments/confirm_archived.py`), that candidate **does** pass — backtest_a valid
   0.5966 / test 0.5976, gap -0.0010, 0.052 clear of the ceiling. So the prior's claim was
   true in substance but unverified when made. The process failure stands regardless of the
   answer: we asserted a check we had not run, and the harness now confirms every accepted
   candidate rather than only suspicious ones.
3. Ten iterations with two accepts shows a trajectory; three iterations does not.
4. The validation difference is 0.0004 — half a standard deviation, not a real difference.

Switching to the other run *after* seeing that it scored 0.0005 higher would be exactly the
selection-on-test this project exists to argue against. The commitment was made first and
is honoured here; that is what makes the claim checkable rather than rhetorical.
