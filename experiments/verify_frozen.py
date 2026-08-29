import sys; sys.path.insert(0,'.')
import numpy as np
from kairos.kernel.causal import frozen_prefix, window_horizons

keys = np.array([1,1,1,1,2,2])
date = np.array([1,2,3,4,1,3])
y    = np.array([1,0,1,1,1,0], dtype=float)
lab  = np.ones(6, bool)
# every row freezes at day 2 -> sees only days 1..2 of its own key
hz = np.full(6, 2, dtype=np.int64)
l,p = frozen_prefix(keys, date, y, lab, hz)
assert list(l)==[2,2,2,2,1,1], l
assert list(p)==[1,1,1,1,1,1], p
print("PASS: frozen aggregates ignore rows past the row's horizon, include its own window")

# a row must NOT see its list-mates: horizon before the window start
hz = np.full(6, 0, dtype=np.int64)
l,p = frozen_prefix(keys, date, y, lab, hz)
assert list(l)==[0]*6 and list(p)==[0]*6
print("PASS: horizon before all data yields zero history (no forward peek)")

rng=np.random.default_rng(0); N=5000
keys=rng.integers(0,40,N); date=rng.integers(1,30,N)
y=(rng.random(N)<.3).astype(float); lab=rng.random(N)<.85
hz=rng.integers(0,30,N)
l,p=frozen_prefix(keys,date,y,lab,hz)
bad=0
for i in rng.choice(N,300,replace=False):
    m=(keys==keys[i])&(date<=hz[i])
    if l[i]!=lab[m].sum() or p[i]!=(y[m]*lab[m]).sum(): bad+=1
assert bad==0, f"{bad} mismatches"
print("PASS: matches brute force on 300 random probes")

d=np.array([20220410,20220425,20220502])
h=window_horizons(d,[(20220408,20220421,20220407),(20220422,20220428,20220421),
                     (20220429,20220508,20220428)])
assert list(h)==[20220407,20220421,20220428]
print("PASS: window_horizons assigns each split its own frozen horizon")
