"""Spot-check the exact MWPM oracle (LP-dual edge elimination + sparse blossom)
against networkx blossom on the FULL complete graph (exact integer weights)."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import sys
import json
import math
import time
import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exact_split import spectrum, Level

out = []
for d, seed in ((128, 31), (256, 31)):
    lam = spectrum(d, seed)
    lv = Level(lam, log=lambda *a: None)
    res = lv.split(verbose=False)
    A, B = lv.AB(res['pairs'])
    dirs = [(B, 2 * A), (1, 0), (B, A), (B, 4 * A)]
    for P, Q in dirs:
        g = math.gcd(P, Q)
        P //= g
        Q //= g
        t = time.time()
        pr, A1, B1, v1 = lv.oracle(P, Q)
        t1 = time.time() - t
        t = time.time()
        c = [P * g2 - Q * gg for gg, g2 in zip(lv.G, lv.G2)]
        W0 = max(c) + 1
        Gx = nx.Graph()
        for e, (i, j) in enumerate(zip(lv.iu, lv.ju)):
            Gx.add_edge(i, j, weight=W0 - c[e])
        mate = nx.max_weight_matching(Gx, maxcardinality=True)
        pr2 = sorted(tuple(sorted(p)) for p in mate)
        v2 = sum(c[lv.eidx(i, j)] for i, j in pr2)
        t2 = time.time() - t
        rec = dict(d=d, seed=seed, P_over_Q=float(P) / max(1, float(Q)),
                   same_value=(v1 == v2), same_matching=(pr == pr2),
                   t_oracle=round(t1, 2), t_full_blossom=round(t2, 2))
        print(rec, flush=True)
        out.append(rec)
json.dump(out, open(os.path.join(HERE, 'oracle_spotcheck.json'), 'w'), indent=1)
