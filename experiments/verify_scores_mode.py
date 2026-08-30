"""Contract test for scores-mode: a candidate trains its own model and blends it with
ctx.baseline_score via rank fusion - the exact strategy that won by hand (+0.0030) and that
feature-matrix mode structurally cannot express.
"""
import sys, textwrap; sys.path.insert(0,'.')
import numpy as np
from kairos.agent.sandbox import run_candidate, static_check
from kairos.agent.evaluate_candidate import evaluate

SRC = textwrap.dedent('''
import numpy as np
import lightgbm as lgb

def build(ctx):
    d = ctx.data
    tr = ctx.fold.idx["train"]
    ytr = d.y_raw[tr].astype(np.float32)

    U, V = ctx.mf_factors(16)
    cf, cf_cnt = ctx.cf_score()
    X = np.concatenate([U, V, cf.reshape(-1,1), cf_cnt.reshape(-1,1)], 1).astype(np.float32)

    scores = np.zeros(d.n, dtype=np.float64)
    for seed in (0, 1):
        m = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=31,
                           min_data_in_leaf=200, verbose=-1, seed=seed, num_threads=4),
                     lgb.Dataset(X[tr], label=ytr), num_boost_round=80)
        scores += m.predict(X) / 2

    def wrank(s, users):
        s = np.asarray(s, float)
        o = np.lexsort((np.arange(len(s)), -s, users)); u = users[o]
        st = np.flatnonzero(np.r_[True, u[1:] != u[:-1]]); sz = np.diff(np.r_[st, len(u)])
        seg = np.repeat(np.arange(len(st)), sz)
        p = 1.0 - (np.arange(len(u)) - st[seg]) / np.maximum(sz[seg]-1, 1)
        r = np.empty(len(s)); r[o] = p; return r

    r_gb = wrank(scores, d.user_id)
    r_fm = wrank(ctx.baseline_score, d.user_id)
    final = 0.5*r_fm + 0.5*r_gb
    ctx.check(final.reshape(-1,1), ["blended_rank"])
    return final.reshape(-1,1).astype(np.float32), ["blended_rank"], {"mode": "scores"}
''').strip()

static_check(SRC)
print("static gate: PASS (lightgbm import allowed, no torch)")
res = run_candidate(SRC, workdir='runs/_scores_mode_test')
assert res['ok'], res.get('error')
print(f"sandbox run: OK ({res['seconds']}s), train_cfg={res['train_cfg']}")

X = np.load(res['X_path'])
r = evaluate(X, 'official', train_cfg=res['train_cfg'])
print(f"scores-mode eval: valid primary {r['valid_primary']:.4f} "
      f"(FM alone ~0.600, GBDT-on-MF/CF alone weaker, blend should beat both)")
assert r['objective'] == 'scores' and r['best_iter'] is None
print("PASS: scores-mode candidate trained its own model and blended with baseline_score")
