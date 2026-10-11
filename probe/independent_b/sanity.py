"""Independent exact checks for small N.
  brute_all(lam):  enumerates EVERY perfect matching (N<=16: 2,027,025 for N=16).
  dyck_all(lam):   enumerates every NON-NESTING perfect matching (Catalan(n));
                   exact because the R-minimiser M* minimises sum(g^2 - s g)
                   (s = 2 Eq*) and a nested pair can be uncrossed with a strict
                   decrease of that strictly convex linear objective.
Both use float64 for the scan, keep all candidates within 1e-9 relative of the
best, then rank the candidates in exact rational arithmetic.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import sys
import json
import time
from fractions import Fraction
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exact_split import spectrum, Level, fmt


def exact_R(lam, pairs):
    n = len(pairs)
    A = Fraction(0)
    B = Fraction(0)
    for i, j in pairs:
        g = Fraction(float(lam[j])) - Fraction(float(lam[i]))
        A += g * g
        B += g
    return n - B * B / A, A / B


def brute_all(lam):
    lam = [float(x) for x in lam]
    N = len(lam)
    used = [False] * N
    pairs = []
    best = [-1.0]
    cands = []
    count = [0]

    def rec(A, B):
        try:
            i = used.index(False)
        except ValueError:
            count[0] += 1
            f = B * B / A
            if f >= best[0] * (1 - 1e-9):
                if f > best[0]:
                    best[0] = f
                cands.append((f, list(pairs)))
            return
        used[i] = True
        li = lam[i]
        for j in range(i + 1, N):
            if not used[j]:
                used[j] = True
                g = lam[j] - li
                pairs.append((i, j))
                rec(A + g * g, B + g)
                pairs.pop()
                used[j] = False
        used[i] = False

    rec(0.0, 0.0)
    cands = [c for c in cands if c[0] >= best[0] * (1 - 1e-9)]
    ex = sorted((exact_R(lam, p)[0], p) for _, p in cands)
    return ex[0][1], count[0], len(cands)


def dyck_words(n):
    """All Dyck words of length 2n as int masks (bit p set = position p opens)."""
    h = n  # split into two halves of length n each
    pats = np.arange(1 << h, dtype=np.int64)
    bits = ((pats[:, None] >> np.arange(h)) & 1).astype(np.int8)
    steps = 2 * bits - 1
    # prefixes: partial sums >= 0
    ps = np.cumsum(steps, axis=1)
    pre_ok = np.all(ps >= 0, axis=1)
    pre_bal = ps[:, -1]
    # suffixes: starting at balance b, partial sums b + cumsum >= 0, end 0
    words = []
    for b in range(0, h + 1, 2 if h % 2 == 0 else 1):
        pre = pats[pre_ok & (pre_bal == b)]
        if len(pre) == 0:
            continue
        suf_ok = np.all(b + ps >= 0, axis=1) & (b + ps[:, -1] == 0)
        suf = pats[suf_ok]
        if len(suf) == 0:
            continue
        w = (suf[None, :] << h) | pre[:, None]
        words.append(w.ravel())
    return np.concatenate(words)


def dyck_all(lam, chunk=2_000_000):
    lam = np.asarray(lam, dtype=np.float64)
    N = len(lam)
    n = N // 2
    W = dyck_words(n)
    tot = len(W)
    best_f = -1.0
    cand_words = []
    for s in range(0, tot, chunk):
        w = W[s:s + chunk]
        m = len(w)
        openpos = np.zeros((m, n), dtype=np.int16)
        no = np.zeros(m, dtype=np.int64)
        nc = np.zeros(m, dtype=np.int64)
        A = np.zeros(m)
        B = np.zeros(m)
        ar = np.arange(m)
        for p in range(N):
            isop = ((w >> p) & 1).astype(bool)
            io = ar[isop]
            openpos[io, no[io]] = p
            no[io] += 1
            ic = ar[~isop]
            o = openpos[ic, nc[ic]]
            g = lam[p] - lam[o]
            A[ic] += g * g
            B[ic] += g
            nc[ic] += 1
        f = B * B / A
        fm = f.max()
        if fm > best_f:
            best_f = fm
        keep = f >= best_f * (1 - 1e-9)
        cand_words.extend(w[keep].tolist())
    # decode candidates and rank exactly
    ex = []
    for wd in cand_words:
        q = []
        pairs = []
        for p in range(N):
            if (wd >> p) & 1:
                q.append(p)
            else:
                pairs.append((q.pop(0), p))
        R, Eq = exact_R(lam, pairs)
        ex.append((R, sorted(pairs)))
    ex.sort()
    return ex[0][1], tot, len(cand_words)


if __name__ == '__main__':
    out = {}
    for d in (16, 32):
        lam = spectrum(d, 5)
        rec = dict(d=d, seed=5)
        t = time.time()
        lv = Level(lam)
        res = lv.split(verbose=False)
        rec['main'] = dict(R=res['R_str'], Eq=res['Eq_str'], pairs=res['pairs'],
                           n_oracle=res['n_oracle'], time=round(time.time() - t, 2))
        if d == 16:
            t = time.time()
            p, cnt, nc = brute_all(lam)
            R, Eq = exact_R(lam, p)
            rec['brute_all'] = dict(R=fmt(R), Eq=fmt(Eq), pairs=p, n_matchings=cnt,
                                    n_candidates=nc, time=round(time.time() - t, 2))
        t = time.time()
        p, cnt, nc = dyck_all(lam)
        R, Eq = exact_R(lam, p)
        rec['dyck_all'] = dict(R=fmt(R), Eq=fmt(Eq), pairs=p, n_matchings=cnt,
                               n_candidates=nc, time=round(time.time() - t, 2))
        same = all(rec[k]['pairs'] == rec['main']['pairs'] for k in rec
                   if isinstance(rec[k], dict))
        rec['all_agree'] = same
        print(json.dumps({k: (v if not isinstance(v, dict) else
                              {kk: vv for kk, vv in v.items() if kk != 'pairs'})
                          for k, v in rec.items()}, indent=1))
        out['d%d_seed5' % d] = rec
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'sanity_out.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
