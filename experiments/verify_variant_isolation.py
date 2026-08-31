"""No two dataset variants may share a cache file.

The columnar cache was made variant-aware early; the DERIVED-signal caches (fm_signal,
refit, din, mf, cf, expert, aux) were not, and their paths were fixed strings. A 1k run
therefore np.load-ed Pure's 1,436,609-row signal into an 11,713,045-row problem. This test
pins both halves of the contract: Pure's paths are unchanged (so no cached work is thrown
away), and no variant's path collides with Pure's.
"""
import os, subprocess, sys, json

MODULES = [('kairos.kernel.baseline_signal', ['CACHE', 'AUX_CACHE_DIR']),
           ('kairos.kernel.refit_signal',    ['CACHE_DIR']),
           ('kairos.kernel.din_signal',      ['CACHE_DIR']),
           ('kairos.kernel.mf_signal',       ['CACHE_DIR']),
           ('kairos.kernel.cf_signal',       ['CACHE_DIR']),
           ('kairos.kernel.expert_signal',   ['CACHE_DIR'])]

PROBE = """
import sys, json; sys.path.insert(0, '.')
import importlib
out = {}
from kairos.kernel.dataset import CACHE_DIR as COLUMNAR
out['columnar'] = COLUMNAR
for mod, attrs in %r:
    m = importlib.import_module(mod)
    for a in attrs:
        out[mod + '.' + a] = getattr(m, a)
print(json.dumps(out))
"""


def paths_for(variant):
    env = dict(os.environ, KAIROS_VARIANT=variant, PYTHONWARNINGS='ignore')
    r = subprocess.run([sys.executable, '-c', PROBE % MODULES],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-500:]
    return json.loads(r.stdout.strip().splitlines()[-1])


pure, k1 = paths_for('pure'), paths_for('1k')

# 1. Pure's paths must be exactly what they always were - no cache invalidated.
expected = {'columnar': './runs/cache',
            'kairos.kernel.baseline_signal.CACHE': 'runs/fm_signal.npy',
            'kairos.kernel.baseline_signal.AUX_CACHE_DIR': 'runs/aux_cache',
            'kairos.kernel.refit_signal.CACHE_DIR': 'runs/refit_cache',
            'kairos.kernel.din_signal.CACHE_DIR': 'runs/din_cache',
            'kairos.kernel.mf_signal.CACHE_DIR': 'runs/mf_cache',
            'kairos.kernel.cf_signal.CACHE_DIR': 'runs/cf_cache',
            'kairos.kernel.expert_signal.CACHE_DIR': 'runs/expert_cache'}
for k, v in expected.items():
    assert pure[k] == v, f"Pure path moved: {k} = {pure[k]!r}, expected {v!r}"
print(f"  [PASS] all {len(expected)} Pure cache paths unchanged")

# 2. No 1k path may equal any Pure path.
collisions = [k for k in pure if pure[k] == k1[k]]
assert not collisions, f"variants share a cache path: {collisions}"
print(f"  [PASS] all {len(pure)} 1k paths differ from Pure's")

# 3. Every key present in both, none silently dropped.
assert set(pure) == set(k1), 'variant probes disagree on which caches exist'
print("  [PASS] both variants expose the same cache set")
# 4. A cache whose length does not match the dataset must be REFUSED, not returned.
#    Path discipline alone did not hold: the same keying mistake recurred at four sites
#    (fixed paths, then paths keyed by fold name). A length check at the point of load is
#    the invariant that binds without relying on anyone remembering the convention.
import numpy as np, tempfile
sys.path.insert(0, '.')
from kairos.kernel.dataset import load_cached, LeakageError

with tempfile.TemporaryDirectory() as td:
    p = os.path.join(td, 'wrong_length.npy')
    np.save(p, np.zeros(1_436_609, dtype=np.float32))       # a Pure-sized signal
    try:
        load_cached(p, 11_713_045, 'fm signal')             # loaded into a 1k-sized run
        raise AssertionError('a cross-variant cache was accepted')
    except LeakageError as e:
        assert '1,436,609' in str(e) and '11,713,045' in str(e), f'unhelpful message: {e}'
    print("  [PASS] a cross-variant cache is refused with both row counts named")

    ok = os.path.join(td, 'right_length.npy')
    np.save(ok, np.zeros(500, dtype=np.float32))
    assert load_cached(ok, 500, 'fm signal') is not None, 'a valid cache was refused'
    assert load_cached(os.path.join(td, 'absent.npy'), 500) is None, 'missing cache must be None'
    print("  [PASS] a matching cache loads and a missing one returns None")

print("\nVARIANT-ISOLATION TESTS PASS")
