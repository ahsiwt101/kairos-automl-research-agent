
import sys, json, time, traceback
sys.path.insert(0, '/Users/twishamehta/tiktok/kuairand-starter-kit')
import numpy as np
def main():
    import importlib.util
    spec = importlib.util.spec_from_file_location('candidate', '/Users/twishamehta/tiktok/kuairand-starter-kit/runs/arm_greedy/cand1/candidate.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    from kairos.agent.context import make_context
    ctx = make_context('official')
    t0 = time.time()
    built = m.build(ctx)
    cfg = {}
    if len(built) == 3:
        X, names, cfg = built
    else:
        X, names = built
    X = np.asarray(X, dtype=np.float32)
    assert X.ndim == 2, f"build() must return a 2-D matrix, got shape {X.shape}"
    assert X.shape[0] == ctx.data.n, (
        f"build() returned {X.shape[0]} rows but the log has {ctx.data.n}; "
        f"features must be aligned to ALL rows in data order")
    assert len(names) == X.shape[1], (
        f"{len(names)} names for {X.shape[1]} columns")
    assert np.isfinite(X).all(), "feature matrix contains NaN or Inf"
    np.save('/Users/twishamehta/tiktok/kuairand-starter-kit/runs/arm_greedy/cand1/X.npy', X)
    json.dump({'names': list(names), 'seconds': round(time.time()-t0,1),
               'train_cfg': dict(cfg or {})}, open('/Users/twishamehta/tiktok/kuairand-starter-kit/runs/arm_greedy/cand1/meta.json', 'w'))
    print("OK")
try:
    main()
except Exception:
    traceback.print_exc(); sys.exit(1)
