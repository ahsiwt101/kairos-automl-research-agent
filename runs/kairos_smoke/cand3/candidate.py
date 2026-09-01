import numpy as np
def build(ctx):
    d = ctx.data
    y = d.y_raw.astype(np.float64)
    hz = ctx.window_horizons(d.date.astype(np.int64), ctx.OFFICIAL_WINDOWS)
    lab = np.ones(d.n, dtype=bool)
    cols, names = [], []
    for nm, keys in (('item', d.video_id.astype(np.int64)),
                     ('user', d.user_id.astype(np.int64))):
        l_, p_ = ctx.frozen_prefix(keys, d.date.astype(np.int64), y, lab, hz)
        cols.append(ctx.smoothed_rate(p_, l_, 0.33, 20.0)); names.append(nm+'_rate')
        cols.append(np.log1p(l_)); names.append(nm+'_logn')
    return np.stack(cols, 1).astype(np.float32), names