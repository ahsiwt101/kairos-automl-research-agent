"""The candidate sandbox must bound MEMORY as well as time.

Without this, a candidate that over-allocates gets the PARENT killed by the OS (the parent
holds the columnar cache and every prewarmed signal, so it is the largest process), and the
run vanishes with no traceback and no ledger entry. This test pins the contract: an
over-allocating candidate comes back as an ordinary, readable, repairable failure.
"""
import sys
sys.path.insert(0, '.')
from kairos.agent.sandbox import run_candidate

# The bomb must be CONTRACT-VALID (right row count) and must actually TOUCH the memory.
# A first version returned a wrong-length array and np.zeros' lazy allocation meant the
# row-count assertion caught it before the cap ever fired - the test passed for the wrong
# reason and proved nothing. This version returns exactly ctx.data.n rows and writes to
# every page, so the only thing that can stop it is the cap.
BOMB = """
def build(ctx):
    import numpy as np
    n = ctx.data.n
    cols = int(3.0 * (1024 ** 3) / 8 / n) + 1      # ~3 GB of float64, vs a 2 GB cap
    X = np.ones((n, cols), dtype=np.float64)       # ones() writes; zeros() may not
    X += 1.0                                        # force the pages resident
    # Hold the pages while doing allowlisted work. A real over-allocating candidate stays
    # resident for its whole fit; this bomb would otherwise allocate and exit inside one
    # poll interval and the watchdog would never observe it. (`import time` is correctly
    # refused by the static allowlist, so the wait is spent on arithmetic instead.)
    for _ in range(40):
        X[::997] += 1.0
    return X, [f'c{i}' for i in range(cols)]
"""

OK = """
def build(ctx):
    import numpy as np
    n = ctx.data.n
    return np.zeros((n, 1), dtype=np.float32), ['zero']
"""

r = run_candidate(BOMB, workdir='runs/_memguard_test', timeout=180, mem_limit_gb=2)
assert not r['ok'], 'over-allocating candidate was reported as ok'
blob = (r.get('error', '') + r.get('stage', '')).lower()
# must fail for the MEMORY reason specifically, not incidentally on some other check
assert 'memory' in blob or 'alloc' in blob or 'cannot allocate' in blob, \
    f"failed, but not on memory - the cap may not be firing: {r}"
print(f"  [PASS] allocation bomb contained: stage={r['stage']}, parent alive")
print(f"         error tail: {r.get('error','').strip().splitlines()[-1][:90]}")

r2 = run_candidate(OK, workdir='runs/_memguard_test2', timeout=600, mem_limit_gb=6)
assert r2['ok'], f"a modest candidate was wrongly killed by the cap: {r2}"
print("  [PASS] a modest candidate still runs under the same cap")
print("\nMEMORY-GUARD TESTS PASS")
