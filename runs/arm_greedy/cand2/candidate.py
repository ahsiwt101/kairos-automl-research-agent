import numpy as np
def build(ctx):
    from kairos.kernel.candidates import build_candidate_matrix
    X, names, hz = build_candidate_matrix(ctx.data, ctx.fold.spec, 'causal', ('item', 'author', 'user', 'user_author', 'user_item', 'user_tab', 'user_dur', 'item_tab'))
    return X, names, {'objective': 'lambdarank', 'group': 'user_day'}