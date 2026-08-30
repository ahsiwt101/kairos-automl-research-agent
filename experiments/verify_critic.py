"""The critic must catch a prediction that does not follow from its mechanism.

Targets the measured failure: the agent's prediction hit-rate is 0 of 2, partly because
"primary increases" is true of any improvement and therefore tests nothing about the
specific mechanism claimed.
"""
import sys; sys.path.insert(0,'.')
from kairos.agent.proposer import TwoStageProposer

p = TwoStageProposer()

VACUOUS = {
    'statement': 'Add duration-relative features to correct systematic over-ranking of '
                 'long videos within a user list',
    'mechanism': 'Diagnostics show duration_decile is the largest single source of GAUC '
                 'inversion loss (0.2946 total, worst at deciles 4 and 6), meaning the '
                 'model systematically mis-orders items by video length.',
    'prediction': {'diagnostic': 'primary', 'direction': 'increase'},
}
COHERENT = {
    'statement': 'Add duration-relative features to correct over-ranking of long videos',
    'mechanism': 'duration_decile is the largest source of GAUC inversion loss (0.2946), '
                 'so correcting duration mis-ordering should reduce exactly that loss.',
    'prediction': {'diagnostic': 'inversion_loss_duration', 'direction': 'decrease'},
}

print("=== case 1: VACUOUS prediction (mechanism is about duration, predicts 'primary') ===")
out = p._criticise(dict(VACUOUS))
print(f"  coheres      : {p.last_critique.get('coheres')}")
print(f"  reason       : {p.last_critique.get('reason','')[:170]}")
print(f"  prediction   : {VACUOUS['prediction']}  ->  {out['prediction']}")
assert p.last_critique.get('coheres') is False, "critic must reject a vacuous prediction"
assert out['prediction'] != VACUOUS['prediction'], "critic must substitute a sharper one"
print("  PASS: caught and replaced")

print("\n=== case 2: COHERENT prediction (mechanism and diagnostic match) ===")
out2 = p._criticise(dict(COHERENT))
print(f"  coheres      : {p.last_critique.get('coheres')}")
print(f"  prediction   : {out2['prediction']}")
assert out2['prediction']['diagnostic'] == 'inversion_loss_duration', \
    "critic must not damage an already-coherent prediction"
print("  PASS: left intact")

print(f"\ncritic cost: {p.tokens_in} in / {p.tokens_out} out tokens for 2 checks")
