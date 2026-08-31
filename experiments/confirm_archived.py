"""Run backtest confirmation on the ARCHIVED submitted candidate, after the fact.

That candidate was accepted on validation alone: confirmation fired only on
implausible-looking gains, and its +0.0018 did not qualify - while PRIOR_PURE told the
agent every run that it had "passed backtest confirmation". The claim is corrected in the
prior; this establishes what the answer actually is, so the record is complete rather than
merely un-false.
"""
import sys, json, os; sys.path.insert(0, '.')
from kairos.agent.loop import Kairos

# Source the code from the LEDGER, not from the candidate directory: the sandbox .py
# files are working artifacts and were stripped when the repo history was rewritten to
# remove large blobs. The ledger's code_diff is the durable record - which is precisely
# why Deliverable 3 asks for the diff per iteration.
_led = [json.loads(l) for l in open('runs/kairos_agent_submission/ledger.jsonl') if l.strip()]
_accepted = [e for e in _led if e['decision'] == 'accept']
assert len(_accepted) == 1, f'expected one accepted candidate, found {len(_accepted)}'
src = _accepted[0]['code_diff']
print(f"  recovered accepted candidate from ledger iteration {_accepted[0]['iteration']} "
      f"({len(src)} chars)")

class _Prop:
    code = src

class _NullProposer:
    """Satisfies the constructor; no call is made - only _backtest_confirm runs."""
    tokens_in = tokens_out = 0
    by_model = {}


# Build the REAL object rather than a hand-rolled stub. Guessing which attributes
# _backtest_confirm touches turned into whack-a-mole (data, fold, seeds, ...); constructing
# it properly cannot drift from the implementation. Caches are already warm, so prewarm is
# a set of cheap np.loads.
k = Kairos(_NullProposer(), fold_name='official', workdir='runs/_confirm_archived',
           seeds=(0, 1, 2), prewarm=(), confirm_fold='backtest_a')

ok, detail = k._backtest_confirm(_Prop())
verdict = {True: 'CONFIRMED', False: 'DISCONFIRMED', None: 'UNVERIFIABLE'}[ok]
print(f'\n  archived submitted candidate -> {verdict}')
print(f'  {detail}')
json.dump({'verdict': verdict, 'detail': detail},
          open('runs/archived_backtest_confirm.json', 'w'), indent=1)
