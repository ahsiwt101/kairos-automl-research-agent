"""A pool of pipelines spanning the leaky<->honest axis, buildable on any fold.

The point of the pool is not to find the best model.  It is to contain candidates that
differ in HOW MUCH they exploit within-window label feedback, so that selection rules can
be compared on their ability to tell those apart from genuine improvements.
"""
import numpy as np
from kairos.kernel.causal import (causal_prefix, frozen_prefix, window_horizons,
                                  smoothed_rate)
from kairos.kernel.frozenfeat import windows_for_fold, _dayindex

FAMILIES = ('item', 'author', 'user', 'user_author', 'user_item', 'user_tab', 'user_dur',
            'item_tab')


def _keys(d, fam, edges):
    def pair(a, b):
        return np.unique(np.stack([a, b], 1), axis=0, return_inverse=True)[1].astype(np.int64)
    v = d.video_id.astype(np.int64); uid = d.user_id.astype(np.int64)
    tab = d.col('tab').astype(np.int64)
    vb = d.col('author_id', 'vb')
    author = np.where(v < len(vb), vb[np.minimum(v, len(vb) - 1)], -1).astype(np.int64)
    durb = np.searchsorted(edges, d.col('duration_ms').astype(np.float64)).astype(np.int64)
    return {'item': v, 'author': author, 'user': uid,
            'user_author': pair(uid, author), 'user_item': pair(uid, v),
            'user_tab': pair(uid, tab), 'user_dur': pair(uid, durb),
            'item_tab': pair(v, tab)}[fam], durb


def build_candidate_matrix(d, fold_spec, mode, families, alpha=20.0):
    """mode='causal'  -> streaming prefix at a single horizon  (the natural, LEAKY choice)
       mode='frozen'  -> window-frozen aggregates              (deployment-faithful)"""
    date = d.date.astype(np.int64)
    y = d.y_raw.astype(np.float64)
    tr_hi = fold_spec['train'][1]
    tr_mask = date <= tr_hi
    prior = float(y[tr_mask].mean())
    edges = np.quantile(d.col('duration_ms').astype(np.float64)[tr_mask],
                        np.linspace(0, 1, 11)[1:-1])
    horizon = fold_spec['valid'][1]
    wins = windows_for_fold(fold_spec)
    hz = window_horizons(date, wins)

    cols, names = [], []
    for fam in families:
        keys, durb = _keys(d, fam, edges)
        if mode == 'causal':
            lab = (date <= horizon)
            _, l_, p_ = causal_prefix(keys, d.time_ms, y, lab)
        else:
            l_, p_ = frozen_prefix(keys, date, y, np.ones(d.n, bool), hz)
        cols.append(smoothed_rate(p_, l_, prior, alpha)); names.append(f'{fam}_rate')
        cols.append(np.log1p(l_));                        names.append(f'{fam}_logn')
    dur = d.col('duration_ms').astype(np.float64)
    _, durb = _keys(d, 'item', edges)
    cols += [np.log1p(dur), durb.astype(np.float64), d.col('tab').astype(np.float64),
             d.col('hourmin').astype(np.float64) // 100]
    names += ['log_duration', 'dur_bucket', 'tab', 'hour']
    if mode == 'frozen':
        cols.append((_dayindex(date) - _dayindex(hz)).astype(np.float64))
        names.append('staleness_days')
    return np.stack(cols, 1).astype(np.float32), names, hz


# Entries are (name, mode, families, objective, train_group).
# The objective axis is included deliberately.  A binary-objective pool shows the leak
# inflating validation without changing the test score much, which makes selection look
# harmless.  lambdarank over per-day groups is an entirely standard choice for a ranking
# task, and it is where the leak turns destructive - so leaving it out would understate
# the risk, not avoid bias.
POOL_V2 = [
    ('causal_all_bin',   'causal', FAMILIES,                    'binary',     None),
    ('causal_ui_bin',    'causal', ('user','item','user_item','user_author'), 'binary', None),
    ('causal_all_lmr',   'causal', FAMILIES,                    'lambdarank', 'user_day'),
    ('causal_ui_lmr',    'causal', ('user','item','user_item','user_author'), 'lambdarank','user_day'),
    ('causal_user_lmr',  'causal', ('user','item','user_tab'),  'lambdarank', 'user_day'),
    ('frozen_all_bin',   'frozen', FAMILIES,                    'binary',     None),
    ('frozen_ui_bin',    'frozen', ('user','item','user_item','user_author'), 'binary', None),
    ('frozen_all_lmr',   'frozen', FAMILIES,                    'lambdarank', 'user_day'),
    ('frozen_ui_lmr',    'frozen', ('user','item','user_item','user_author'), 'lambdarank','user_day'),
    ('frozen_item_bin',  'frozen', ('item','author','item_tab'),'binary',     None),
]

POOL = [
    ('causal_all',      'causal', FAMILIES),
    ('causal_user',     'causal', ('user', 'item', 'user_tab')),
    ('causal_ui',       'causal', ('user', 'item', 'user_item', 'user_author')),
    ('causal_item',     'causal', ('item', 'author', 'item_tab')),
    ('frozen_all',      'frozen', FAMILIES),
    ('frozen_user',     'frozen', ('user', 'item', 'user_tab')),
    ('frozen_ui',       'frozen', ('user', 'item', 'user_item', 'user_author')),
    ('frozen_item',     'frozen', ('item', 'author', 'item_tab')),
    ('frozen_nouser',   'frozen', ('item', 'author', 'item_tab', 'user_dur')),
    ('causal_nouser',   'causal', ('item', 'author', 'item_tab', 'user_dur')),
]
