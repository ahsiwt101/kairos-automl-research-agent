
def build(ctx):
    import numpy as np
    n = ctx.data.n
    return np.zeros((n, 1), dtype=np.float32), ['zero']
