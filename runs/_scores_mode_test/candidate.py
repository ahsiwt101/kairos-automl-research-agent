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