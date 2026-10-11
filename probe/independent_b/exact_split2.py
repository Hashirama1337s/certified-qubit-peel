"""Scalable variant of exact_split.Level for large N (e.g. N=2048, 2.1M edges).

Same hull search and the same exactness argument; the only change is in the
oracle: reduced costs of ALL edges are first evaluated in float64, and an edge
is excluded only if its float reduced cost clears a rigorous rounding margin.
Exact integer arithmetic is used for every edge that is not cleared.

Float error bound (u = 2^-53): with gf = fl(lam_j - lam_i) = g(1+d0),
p = fl(P), q = fl(Q)/2^K, craw = p*gf*gf - q*gf, cf = craw/2^sig and
rf = (cf - y_i) - y_j, one has
   |rf - r_exact/2^shift| <= 8u * T_e + 2*2^-shift,
   T_e = (P g^2 + Q' g)/2^sig + |y_i| + |y_j|.
We use margin_e = 1e-13 * (T_e(float) + 1), i.e. > 100x that bound.
  * rf_e > margin_e            => exact r_e > 0   (no repair needed there)
  * rf_e - margin_e > gap_up   => exact r_e > gap (edge cannot be in an optimum)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import math
import time
from fractions import Fraction

import numpy as np
import networkx as nx
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

from exact_split import spectrum, to_exact_ints, fmt  # noqa: F401


class DeadlineHit(Exception):
    pass


class BigLevel:
    def __init__(self, lam, log=print):
        lam = np.asarray(lam, dtype=np.float64)
        assert np.all(np.diff(lam) > 0)
        self.lam = lam
        self.N = N = len(lam)
        assert N % 2 == 0
        self.n = N // 2
        self.K, self.L = to_exact_ints(lam)
        iu, ju = np.triu_indices(N, 1)
        self.iu_np = iu.astype(np.int64)
        self.ju_np = ju.astype(np.int64)
        self.gf = lam[ju] - lam[iu]
        self.log = log
        self.calls = []

    def eidx(self, i, j):
        if i > j:
            i, j = j, i
        N = self.N
        return i * N - i * (i + 1) // 2 + (j - i - 1)

    def ij(self, e):
        return int(self.iu_np[e]), int(self.ju_np[e])

    def Gx(self, e):
        i, j = self.ij(e)
        return self.L[j] - self.L[i]

    def AB(self, pairs):
        A = 0
        B = 0
        for i, j in pairs:
            g = self.L[max(i, j)] - self.L[min(i, j)]
            A += g * g
            B += g
        return A, B

    def _nx_match(self, sel, cexact):
        N = self.N
        G = nx.Graph()
        G.add_nodes_from(range(N))
        W0 = max(cexact[e] for e in sel) + 1
        for e in sel:
            w = W0 - cexact[e]
            assert type(w) is int and w > 0
            i, j = self.ij(e)
            G.add_edge(i, j, weight=w)
        mate = nx.max_weight_matching(G, maxcardinality=True)
        if len(mate) != self.n:
            return None
        return sorted(tuple(sorted(p)) for p in mate)

    def _lp(self, cf, E):
        N = self.N
        m = len(E)
        rows = np.concatenate([self.iu_np[E], self.ju_np[E]])
        cols = np.concatenate([np.arange(m), np.arange(m)])
        Aeq = csr_matrix((np.ones(2 * m), (rows, cols)), shape=(N, m))
        res = linprog(cf[E], A_eq=Aeq, b_eq=np.ones(N), bounds=(0, None),
                      method='highs',
                      options=dict(dual_feasibility_tolerance=1e-10,
                                   primal_feasibility_tolerance=1e-10))
        assert res.status == 0, res.message
        return res.x, res.eqlin.marginals

    def oracle(self, P, Q, k0=10):
        t0 = time.time()
        N = self.N
        K = self.K
        L = self.L
        iu, ju = self.iu_np, self.ju_np
        p = float(P)
        q = float(Q) / 2.0 ** K
        gf = self.gf
        craw = p * gf * gf - q * gf
        sig = math.frexp(float(np.abs(craw).max()))[1]
        scale = 2.0 ** sig
        cf = craw / scale
        # candidate columns: k0 cheapest edges per vertex + adjacent pairs
        Cm = np.full((N, N), np.inf)
        Cm[iu, ju] = cf
        Cm[ju, iu] = cf
        kk = min(k0, N - 1)
        nb = np.argpartition(Cm, kk - 1, axis=1)[:, :kk]
        del Cm
        ii = np.repeat(np.arange(N), kk)
        jj = nb.ravel()
        a = np.minimum(ii, jj)
        b = np.maximum(ii, jj)
        inE = np.zeros(len(gf), dtype=bool)
        inE[a * N - a * (a + 1) // 2 + (b - a - 1)] = True
        adj = np.array([self.eidx(k, k + 1) for k in range(0, N, 2)])
        inE[adj] = True
        cg = 0
        while True:
            cg += 1
            E = np.nonzero(inE)[0]
            x, y = self._lp(cf, E)
            rf = (cf - y[iu]) - y[ju]
            viol = np.nonzero((rf < -1e-9) & ~inE)[0]
            if len(viol) == 0 or cg > 50:
                break
            vo = viol[np.argsort(rf[viol])][:8 * N]
            inE[vo] = True
        t_lp = time.time() - t0
        # rigorous float margin
        T = (p * gf * gf + q * np.abs(gf)) / scale + np.abs(y[iu]) + np.abs(y[ju])
        margin = 1e-13 * (T + 1.0)
        shift = 2 * K + sig
        two_shift = Fraction(2) ** shift
        Y = [math.floor(Fraction(float(v)) * two_shift) for v in y]
        cex = {}

        def cexact(e):
            v = cex.get(e)
            if v is None:
                g = self.Gx(e)
                v = P * g * g - Q * g
                cex[e] = v
            return v

        # exact repair over all edges not certified positive by the float test
        S0 = np.nonzero(rf <= margin)[0].tolist()
        dfc = [0] * N
        nneg = 0
        for e in S0:
            i, j = self.ij(e)
            re = cexact(e) - Y[i] - Y[j]
            if re < 0:
                nneg += 1
                h = (-re + 1) // 2
                if h > dfc[i]:
                    dfc[i] = h
                if h > dfc[j]:
                    dfc[j] = h
        if nneg:
            Y = [a_ - b_ for a_, b_ in zip(Y, dfc)]
        for e in S0:
            i, j = self.ij(e)
            assert cexact(e) - Y[i] - Y[j] >= 0
        LB = sum(Y)
        # upper bound from a perfect matching
        integral = bool(np.all((x < 1e-7) | (x > 1 - 1e-7)))
        pairs0 = None
        if integral:
            Em = E[x > 0.5]
            if len(Em) == self.n:
                pairs0 = sorted(self.ij(e) for e in Em.tolist())
                if sorted(v for pr in pairs0 for v in pr) != list(range(N)):
                    pairs0 = None
        if pairs0 is None:
            m = 3 * N
            sel = set(np.argpartition(rf, m - 1)[:m].tolist())
            sel.update(adj.tolist())
            for e in sel:
                cexact(e)
            pairs0 = self._nx_match(sel, cex)
        UB0 = 0
        for i, j in pairs0:
            UB0 += cexact(self.eidx(i, j))
        gap = UB0 - LB
        assert gap >= 0
        gap_up = float(Fraction(gap) / two_shift) * (1 + 1e-12) + 1e-300
        cand = np.nonzero(rf - margin <= gap_up)[0].tolist()
        need = []
        for e in cand:
            i, j = self.ij(e)
            if cexact(e) - Y[i] - Y[j] <= gap:
                need.append(e)
        t_nx = time.time()
        pairs = self._nx_match(need, cex)
        t_nx = time.time() - t_nx
        assert pairs is not None
        UB = 0
        for i, j in pairs:
            UB += cexact(self.eidx(i, j))
        assert UB <= UB0
        A, B = self.AB(pairs)
        assert P * A - Q * B == UB
        info = dict(t=time.time() - t0, t_lp=t_lp, t_nx=t_nx, cg=cg, ncols=int(inE.sum()),
                    nS0=len(S0), nneg=nneg, lp_integral=integral, ncand=len(cand),
                    need=len(need))
        self.calls.append(info)
        return pairs, A, B, UB

    def f(self, A, B):
        return Fraction(B * B, A)

    def split(self, deadline=None, verbose=True, progress_every=25):
        t0 = time.time()
        n = self.n
        pts = []

        def add_point(pairs, A, B, P, Q):
            for p_ in pts:
                if p_['A'] == A and p_['B'] == B:
                    return p_, False
            p_ = dict(A=A, B=B, P=P, Q=Q, pairs=pairs, f=self.f(A, B))
            pts.append(p_)
            pts.sort(key=lambda z: (z['B'], z['A']))
            return p_, True

        def apex_f(U, V):
            Pu, Qu, Pv, Qv = U['P'], U['Q'], V['P'], V['Q']
            cu = Pu * U['A'] - Qu * U['B']
            cv = Pv * V['A'] - Qv * V['B']
            D = Qu * Pv - Pu * Qv
            if D == 0:
                return 'edge'
            Aq = Fraction(Qu * cv - Qv * cu, D)
            Bq = Fraction(Pu * cv - Pv * cu, D)
            if Aq <= 0:
                return None  # +inf
            return Bq * Bq / Aq

        done = set()
        timed_out = False
        try:
            pr, A, B, _ = self.oracle(1, 0)
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
                pr, A, B, _ = self.oracle(P // g, Q // g)
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
                    fq = apex_f(U, V)
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
                P = V['B'] - U['B']
                Q = V['A'] - U['A']
                assert P > 0 and Q > 0
                g = math.gcd(P, Q)
                P //= g
                Q //= g
                val_seg = P * U['A'] - Q * U['B']
                pr, A, B, val = self.oracle(P, Q)
                if val < val_seg:
                    p_, new = add_point(pr, A, B, P, Q)
                    assert new
                else:
                    done.add((id(U), id(V)))
                if verbose and len(self.calls) % progress_every == 0:
                    self.log("    calls=%d open=%d best R=%s  %.0fs" % (
                        len(self.calls), len(cand), fmt(n - best['f'], 14), time.time() - t0))
        except DeadlineHit:
            timed_out = True
        best = max(pts, key=lambda z: z['f'])
        # certified bounds: R* <= n - f(best) (achieved); R* >= n - max(f(best), open apexes)
        fmax = best['f']
        inf_open = False
        n_open = 0
        if len(pts) >= 2:
            for U, V in zip(pts[:-1], pts[1:]):
                if (id(U), id(V)) in done:
                    continue
                fq = apex_f(U, V)
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
        out = dict(N=self.N, n=n, R=R, Eq=Eq, R_str=fmt(R, 15), Eq_str=fmt(Eq, 15),
                   pairs=best['pairs'], n_chain_points=len(pts), n_oracle=len(self.calls),
                   time=time.time() - t0, timed_out=timed_out, n_open_triangles=n_open,
                   R_lower_bound=None if inf_open else fmt(n - fmax, 15),
                   R_upper_bound=fmt(R, 15), oracle_info=self.calls)
        return out
