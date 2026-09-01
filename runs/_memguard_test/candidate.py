
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
