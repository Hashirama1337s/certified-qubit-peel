"""Validate the triangle-pruning on a large level: enumerate the COMPLETE lower
chain of conv{(B_M, A_M)} (no f-based pruning; every segment is split until it
is certified a hull edge) and take max f over all its vertices."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import sys
import json
import math
import time
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exact_split import spectrum, Level, fmt

import numpy as np
d, seed = int(sys.argv[1]), int(sys.argv[2])
lev = int(sys.argv[3]) if len(sys.argv) > 3 else 0   # 0 = level 1 (N=d)
lam = spectrum(d, seed)
mt = json.load(open(os.path.join(HERE, 'match_d%d_s%d.json' % (d, seed))))
for _ in range(lev):
    prs = mt['N%d' % len(lam)]
    lam = np.array(sorted((lam[i] + lam[j]) / 2 for i, j in prs))
lv = Level(lam, log=lambda *a: None)
t0 = time.time()
n = lv.n
pr, A, B, _ = lv.oracle(1, 0)
left = (B, A)
pr = [(i, i + n) for i in range(n)]
A2, B2 = lv.AB(pr)
right = (B2, A2)
verts = [left, right]
stack = [(left, right)]
while stack:
    (Bu, Au), (Bv, Av) = stack.pop()
    P, Q = Bv - Bu, Av - Au
    g = math.gcd(P, Q)
    P //= g
    Q //= g
    _, A3, B3, val = lv.oracle(P, Q)
    if val < P * Au - Q * Bu:
        w = (B3, A3)
        verts.append(w)
        stack.append(((Bu, Au), w))
        stack.append((w, (Bv, Av)))
        if len(verts) % 200 == 0:
            print('verts', len(verts), 'calls', len(lv.calls), round(time.time() - t0, 1), flush=True)
best = max(verts, key=lambda v: Fraction(v[0] * v[0], v[1]))
R = n - Fraction(best[0] ** 2, best[1])
out = dict(d=d, seed=seed, N=lv.N, n_chain_vertices=len(verts), n_oracle=len(lv.calls),
           R_full_chain=fmt(R), Eq_full_chain=fmt(Fraction(best[1], best[0] << lv.K)),
           time_s=round(time.time() - t0, 1))
print(out)
json.dump(out, open(os.path.join(HERE, 'fullchain_d%d_s%d_N%d.json' % (d, seed, lv.N)), 'w'), indent=1)
