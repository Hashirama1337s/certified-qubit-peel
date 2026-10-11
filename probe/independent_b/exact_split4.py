"""BigLevel4: duals for the fractional-matching LP from the bipartite double cover.

The degree-constrained LP  min c.x, x(delta(v)) = 1, x >= 0  equals one half of
the assignment problem with costs C[i,j] = c_ij (i != j), C[i,i] = +big: a
fixed-point-free permutation sigma symmetrises to a fractional perfect matching
of half its cost, and conversely.  scipy's linear_sum_assignment gives sigma;
assignment potentials (v by Bellman-Ford/SPFA on the reduced graph, u_i =
C[i,sigma(i)] - v_sigma(i)) give LP duals y_i = (u_i + v_i)/2 with
y_i + y_j <= c_ij.  These floats only steer the search: the exact phase
(integer duals, exact repair to r_e >= 0, rigorous float screening, exact
min-cost perfect matching on surviving edges) is identical to exact_split3.
If sigma has an odd cycle (LP fractional) we fall back to the HiGHS+cuts path.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import math
import time
from fractions import Fraction

import numpy as np
from scipy.optimize import linear_sum_assignment

from exact_split3 import BigLevel3, spectrum, fmt, DeadlineHit  # noqa: F401


class BigLevel4(BigLevel3):
    def __init__(self, lam, log=print):
        super().__init__(lam, log=log)
        self.n_fallback = 0
        self.cut_rounds = 50
        self.last_s = None

    def _highs(self, P, Q, tag):
        out = super().oracle(P, Q)
        self.calls[-1]['dual_source'] = tag
        self.last_s = Q / P if P else None
        return out

    def oracle(self, P, Q, k0=None):
        t0 = time.time()
        N = self.N
        if Q == 0:
            return super().oracle(P, Q)
        s_now = Q / P
        if self.last_s is not None and abs(math.log(s_now / self.last_s)) < 0.02:
            return self._highs(P, Q, 'highs_warm')
        K = self.K
        iu, ju = self.iu_np, self.ju_np
        p = float(P)
        q = float(Q) / 2.0 ** K
        gf = self.gf
        craw = p * gf * gf - q * gf
        sig = math.frexp(float(np.abs(craw).max()))[1]
        scale = 2.0 ** sig
        cf = craw / scale
        C = np.empty((N, N))
        C[iu, ju] = cf
        C[ju, iu] = cf
        np.fill_diagonal(C, 10.0)
        _, sg = linear_sum_assignment(C)
        # cycle structure
        seen = np.zeros(N, dtype=bool)
        cycles = []
        for i in range(N):
            if not seen[i]:
                cyc = []
                j = i
                while not seen[j]:
                    seen[j] = True
                    cyc.append(j)
                    j = int(sg[j])
                cycles.append(cyc)
        odd_cycles = any(len(c) % 2 == 1 for c in cycles)
        if odd_cycles and not getattr(self, 'bound_ok', False):
            self.n_fallback += 1
            return self._highs(P, Q, 'highs_fallback')
        # potentials: v_j <= v_k + W[k, j],  k = sigma(i)
        inv = np.empty(N, dtype=np.int64)
        inv[sg] = np.arange(N)
        W = C[inv, :] - C[inv, sg[inv]][:, None]
        del C
        d = np.zeros(N)
        ch = np.arange(N)
        it = 0
        while len(ch):
            it += 1
            cand = (d[ch][:, None] + W[ch]).min(axis=0)
            nd = np.minimum(d, cand)
            ch = np.nonzero(nd < d - 1e-16)[0]
            d = nd
            if it > 20 * N:
                break
        del W
        v = d
        cdiag = cf[np.array([self.eidx(i, int(sg[i])) for i in range(N)])]
        u = cdiag - v[sg]
        y = 0.5 * (u + v)
        t_lp = time.time() - t0
        # pairs0 from sigma: 2-cycles directly; even cycles -> cheaper alternation
        pairs0 = []
        if odd_cycles:
            cycles = []
        for cyc in cycles:
            if len(cyc) == 2:
                pairs0.append((min(cyc), max(cyc)))
            else:
                L = len(cyc)
                m1 = [(cyc[k], cyc[k + 1]) for k in range(0, L, 2)]
                m2 = [(cyc[k + 1], cyc[(k + 2) % L]) for k in range(0, L, 2)]
                s1 = sum(cf[self.eidx(a, b)] for a, b in m1)
                s2 = sum(cf[self.eidx(a, b)] for a, b in m2)
                pairs0.extend((min(a, b), max(a, b)) for a, b in (m1 if s1 <= s2 else m2))
        pairs0.sort()
        # ---------------- exact phase (no cuts) ----------------
        rf = (cf - y[iu]) - y[ju]
        T = (p * gf * gf + q * np.abs(gf)) / scale + np.abs(y[iu]) + np.abs(y[ju])
        margin = 1e-13 * (T + 1.0)
        shift = 2 * K + sig
        two_shift = Fraction(2) ** shift
        Y = [math.floor(Fraction(float(val)) * two_shift) for val in y]
        cex = {}

        def cexact(e):
            val = cex.get(e)
            if val is None:
                g = self.Gx(e)
                val = P * g * g - Q * g
                cex[e] = val
            return val

        def rexact(e):
            i, j = self.ij(e)
            return cexact(e) - Y[i] - Y[j]

        S0 = np.nonzero(rf <= margin)[0].tolist()
        dfc = [0] * N
        nneg = 0
        for e in S0:
            r = rexact(e)
            if r < 0:
                nneg += 1
                i, j = self.ij(e)
                hh = (-r + 1) // 2
                if hh > dfc[i]:
                    dfc[i] = hh
                if hh > dfc[j]:
                    dfc[j] = hh
        if nneg:
            Y = [a_ - b_ for a_, b_ in zip(Y, dfc)]
        for e in S0:
            assert rexact(e) >= 0
        LB = sum(Y)
        if odd_cycles:
            # certified: every perfect matching M has P*A(M) - Q*B(M) >= LB
            self.calls.append(dict(t=time.time() - t0, t_lp=t_lp, t_nx=0.0, bf_iter=it,
                                   dual_source='assignment_bound', nS0=len(S0), nneg=nneg,
                                   lp_integral=False, ncand=0, need=0, used_nx=False))
            return 'BOUND', LB
        UB0 = 0
        for i, j in pairs0:
            UB0 += cexact(self.eidx(i, j))
        gap = UB0 - LB
        assert gap >= 0
        gap_up = float(Fraction(gap) / two_shift) * (1 + 1e-12) + 1e-300
        cand = np.nonzero(rf - margin <= gap_up)[0].tolist()
        need = [e for e in cand if rexact(e) <= gap]
        t1 = time.time()
        pairs = self._nx_match(need, cex)
        t_nx = time.time() - t1
        assert pairs is not None
        UB = 0
        for i, j in pairs:
            UB += cexact(self.eidx(i, j))
        assert UB <= UB0
        A, B = self.AB(pairs)
        assert P * A - Q * B == UB
        self.calls.append(dict(t=time.time() - t0, t_lp=t_lp, t_nx=t_nx, bf_iter=it,
                               dual_source='assignment', nS0=len(S0), nneg=nneg,
                               lp_integral=all(len(c) == 2 for c in cycles),
                               ncand=len(cand), need=len(need), used_nx=True))
        return pairs, A, B, UB


def _clip(poly, P, Q, L):
    """Clip convex polygon [(B, A)] (Fractions) by half-plane P*A - Q*B >= L."""
    out = []
    n = len(poly)
    for k in range(n):
        a = poly[k]
        b = poly[(k + 1) % n]
        va = P * a[1] - Q * a[0] - L
        vb = P * b[1] - Q * b[0] - L
        if va >= 0:
            out.append(a)
        if (va >= 0) != (vb >= 0):
            t = va / (va - vb)
            out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
    return out


def _poly_fmax(poly):
    best = None
    for Bq, Aq in poly:
        if Aq <= 0:
            return None
        v = Bq * Bq / Aq
        if best is None or v > best:
            best = v
    return best


class BigLevel5(BigLevel4):
    """BigLevel4 + split() that, when the LP at a segment direction is fractional,
    first tries to close the triangle with the certified LP lower bound
    (valid inequality P*A - Q*B >= LB for every perfect matching)."""

    def exact_oracle(self, P, Q):
        self.bound_ok = False
        return BigLevel4.oracle(self, P, Q)

    def split(self, deadline=None, verbose=True, progress_every=25):
        t0 = time.time()
        n = self.n
        pts = []
        extra = {}   # (id(U), id(V)) -> list of (P, Q, L) certified lines

        def add_point(pairs, A, B, P, Q):
            for p_ in pts:
                if p_['A'] == A and p_['B'] == B:
                    return p_, False
            p_ = dict(A=A, B=B, P=P, Q=Q, pairs=pairs, f=Fraction(B * B, A))
            pts.append(p_)
            pts.sort(key=lambda z: (z['B'], z['A']))
            return p_, True

        def region(U, V):
            Pu, Qu, Pv, Qv = U['P'], U['Q'], V['P'], V['Q']
            cu = Pu * U['A'] - Qu * U['B']
            cv = Pv * V['A'] - Qv * V['B']
            D = Qu * Pv - Pu * Qv
            if D == 0:
                return 'edge'
            Aq = Fraction(Qu * cv - Qv * cu, D)
            Bq = Fraction(Pu * cv - Pv * cu, D)
            poly = [(Fraction(U['B']), Fraction(U['A'])), (Fraction(V['B']), Fraction(V['A'])),
                    (Bq, Aq)]
            for (P_, Q_, L_) in extra.get((id(U), id(V)), []):
                poly = _clip(poly, P_, Q_, L_)
            return _poly_fmax(poly)

        done = set()
        timed_out = False
        nbound = [0, 0]
        try:
            pr, A, B, _ = self.exact_oracle(1, 0)
            add_point(pr, A, B, 1, 0)
            pr = [(i, i + n) for i in range(n)]
            A, B = self.AB(pr)
            add_point(pr, A, B, 0, 1)
            best = max(pts, key=lambda z: z['f'])
            for _ in range(30):
                if deadline and time.time() > deadline:
                    raise DeadlineHit
                P, Q = best['B'], 2 * best['A']
                g = math.gcd(P, Q)
                pr, A, B, _ = self.exact_oracle(P // g, Q // g)
                p_, new = add_point(pr, A, B, P // g, Q // g)
                if p_['f'] > best['f']:
                    best = p_
                else:
                    break
            if verbose:
                self.log("  N=%d after descent: %d oracle calls, R=%s, %.0fs" % (
                    self.N, len(self.calls), fmt(n - best['f'], 14), time.time() - t0))
            while True:
                best = max(pts, key=lambda z: z['f'])
                cand = []
                for U, V in zip(pts[:-1], pts[1:]):
                    key = (id(U), id(V))
                    if key in done:
                        continue
                    fq = region(U, V)
                    if fq == 'edge' or (fq is not None and fq <= best['f']):
                        done.add(key)
                        continue
                    cand.append((float('inf') if fq is None else float(fq), U, V))
                if not cand:
                    break
                if deadline and time.time() > deadline:
                    raise DeadlineHit
                cand.sort(key=lambda z: -z[0])
                _, U, V = cand[0]
                key = (id(U), id(V))
                P = V['B'] - U['B']
                Q = V['A'] - U['A']
                assert P > 0 and Q > 0
                g = math.gcd(P, Q)
                P //= g
                Q //= g
                val_seg = P * U['A'] - Q * U['B']
                tried_bound = any(l[0] == P and l[1] == Q for l in extra.get(key, []))
                if not tried_bound:
                    self.bound_ok = True
                    res = BigLevel4.oracle(self, P, Q)
                    self.bound_ok = False
                    if res[0] == 'BOUND':
                        LB = res[1]
                        nbound[0] += 1
                        if LB >= val_seg:
                            done.add(key)      # nothing strictly below segment UV
                            nbound[1] += 1
                        else:
                            extra.setdefault(key, []).append((P, Q, LB))
                        continue
                    pr, A, B, val = res
                else:
                    pr, A, B, val = self.exact_oracle(P, Q)
                if val < val_seg:
                    p_, new = add_point(pr, A, B, P, Q)
                    assert new
                else:
                    done.add(key)
                if verbose and len(self.calls) % progress_every == 0:
                    self.log("    calls=%d open=%d best R=%s bounds=%s %.0fs" % (
                        len(self.calls), len(cand), fmt(n - best['f'], 14), nbound,
                        time.time() - t0))
        except DeadlineHit:
            timed_out = True
        best = max(pts, key=lambda z: z['f'])
        fmax = best['f']
        inf_open = False
        n_open = 0
        for U, V in zip(pts[:-1], pts[1:]):
            if (id(U), id(V)) in done:
                continue
            fq = region(U, V)
            if fq == 'edge':
                continue
            if fq is None:
                inf_open = True
                continue
            if fq > best['f']:
                n_open += 1
                fmax = max(fmax, fq)
        A, B = best['A'], best['B']
        R = n - best['f']
        Eq = Fraction(A, B * (1 << self.K))
        return dict(N=self.N, n=n, R=R, Eq=Eq, R_str=fmt(R, 15), Eq_str=fmt(Eq, 15),
                    pairs=best['pairs'], n_chain_points=len(pts), n_oracle=len(self.calls),
                    time=time.time() - t0, timed_out=timed_out, n_open_triangles=n_open,
                    R_lower_bound=None if inf_open else fmt(n - fmax, 15),
                    R_upper_bound=fmt(R, 15), oracle_info=self.calls,
                    n_bound_queries=nbound[0], n_bound_closed_as_edge=nbound[1])
