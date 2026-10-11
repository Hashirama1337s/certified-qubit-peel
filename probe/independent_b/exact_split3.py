"""BigLevel3: exact_split2.BigLevel with a persistent, warm-started HiGHS LP
(scipy's bundled HiGHS object) + odd-set (blossom) cuts.

Exactness argument is unchanged, extended by cut duals:
  for odd S, x(delta(S)) >= 1 is valid for every perfect matching, so with
  integer Y (free) and Z_S >= 0 and exact reduced costs
     r_e = c_e - Y_i - Y_j - sum_{S: e in delta(S)} Z_S >= 0  (all edges),
  every perfect matching M has c(M) >= sum Y + sum Z + sum_{e in M} r_e,
  hence an optimum only uses edges with r_e <= UB0 - LB.  Edges are excluded
  by a float64 test only if they clear a margin > 100x the rounding-error
  bound; all others are evaluated in exact integers.  If the surviving edge
  set is exactly one perfect matching it is the unique (hence optimal)
  candidate; otherwise networkx's exact integer blossom solves it.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import math
import time
from fractions import Fraction

import numpy as np
from scipy.optimize._highspy._core import _Highs, HighsModelStatus

from exact_split2 import BigLevel, spectrum, fmt, DeadlineHit  # noqa: F401


class BigLevel3(BigLevel):
    def __init__(self, lam, log=print, k0=10):
        super().__init__(lam, log=log)
        N = self.N
        self.k0 = k0
        self.adj = np.array([self.eidx(k, k + 1) for k in range(0, N, 2)])
        self.pool_cap = 12 * N
        self.cut_rounds = 12
        self.cut_cap = N // 2
        self.keep = np.zeros(0, dtype=np.int64)       # tight columns of last call
        self.rebuilds = 0
        self._new_model([])

    def _new_model(self, cuts):
        N = self.N
        h = _Highs()
        h.setOptionValue('output_flag', False)
        h.setOptionValue('dual_feasibility_tolerance', 1e-10)
        h.setOptionValue('primal_feasibility_tolerance', 1e-10)
        h.addRows(N, np.ones(N), np.ones(N), 0, np.zeros(0, np.int32),
                  np.zeros(0, np.int32), np.zeros(0))
        self.h = h
        self.inf = h.getInfinity()
        self.pool = np.zeros(0, dtype=np.int64)      # column j -> edge id
        self.in_pool = np.zeros(len(self.gf), dtype=bool)
        self.cuts = []                                # list of bool[N] masks
        for mem in cuts:
            self._add_cut(mem)

    # ---- LP model maintenance ----
    def _add_cols(self, E, cf):
        E = np.asarray(E, dtype=np.int64)
        E = E[~self.in_pool[E]]
        if len(E) == 0:
            return 0
        iu, ju = self.iu_np[E], self.ju_np[E]
        cols = [[a, b] for a, b in zip(iu.tolist(), ju.tolist())]
        for k, mem in enumerate(self.cuts):
            cr = np.nonzero(mem[iu] != mem[ju])[0]
            for t in cr.tolist():
                cols[t].append(self.N + k)
        starts = np.zeros(len(E), dtype=np.int32)
        idx = []
        pos = 0
        for t, cl in enumerate(cols):
            starts[t] = pos
            idx.extend(cl)
            pos += len(cl)
        idx = np.array(idx, dtype=np.int32)
        m = len(E)
        self.h.addCols(m, cf[E], np.zeros(m), np.full(m, self.inf), len(idx), starts, idx,
                       np.ones(len(idx)))
        self.pool = np.concatenate([self.pool, E])
        self.in_pool[E] = True
        return m

    def _add_cut(self, mem):
        P = self.pool
        cr = np.nonzero(mem[self.iu_np[P]] != mem[self.ju_np[P]])[0].astype(np.int32)
        self.h.addRows(1, np.array([1.0]), np.array([self.inf]), len(cr),
                       np.array([0], dtype=np.int32), cr, np.ones(len(cr)))
        self.cuts.append(mem)

    def _solve(self):
        self.h.run()
        st = self.h.getModelStatus()
        assert st == HighsModelStatus.kOptimal, str(st)
        s = self.h.getSolution()
        x = np.array(s.col_value)
        rd = np.array(s.row_dual)
        return x, rd[:self.N], rd[self.N:]

    def _odd_components(self, x):
        """Vertex sets of odd connected components of the fractional support."""
        P = self.pool
        fr = (x > 1e-7) & (x < 1 - 1e-7)
        if not fr.any():
            return [], True
        N = self.N
        parent = list(range(N))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        for e in P[fr].tolist():
            a, b = find(int(self.iu_np[e])), find(int(self.ju_np[e]))
            if a != b:
                parent[a] = b
        verts = set(self.iu_np[P[fr]].tolist()) | set(self.ju_np[P[fr]].tolist())
        comp = {}
        for v in verts:
            comp.setdefault(find(v), []).append(v)
        odd = []
        for vs in comp.values():
            if len(vs) % 2 == 1:
                mem = np.zeros(N, dtype=bool)
                mem[vs] = True
                odd.append(mem)
        return odd, False

    def _nx_match(self, sel, cexact):
        """Exact min-cost perfect matching on edge set `sel`:
        1. a vertex of degree 1 forces its edge (repeat);
        2. a remaining component that is a cycle has exactly two perfect
           matchings -> compare exactly;
        3. anything else -> networkx integer blossom per component."""
        N = self.N
        adjl = {}
        for e in sel:
            i, j = self.ij(e)
            adjl.setdefault(i, {})[j] = e
            adjl.setdefault(j, {})[i] = e
        if len(adjl) != N:
            return None
        forced = []
        stack = [v for v, nb in adjl.items() if len(nb) == 1]
        alive = set(adjl)
        while stack:
            v = stack.pop()
            if v not in alive:
                continue
            nb = adjl[v]
            if len(nb) == 0:
                return None
            if len(nb) != 1:
                continue
            (u, e), = nb.items()
            forced.append((min(u, v), max(u, v)))
            alive.discard(u)
            alive.discard(v)
            for w in (u, v):
                for x in list(adjl[w]):
                    del adjl[x][w]
                    if x in alive and len(adjl[x]) == 1:
                        stack.append(x)
                    if x in alive and len(adjl[x]) == 0:
                        return None
                adjl[w] = {}
        if not alive:
            return sorted(forced)
        rest_edges = set()
        for v in alive:
            for u, e in adjl[v].items():
                rest_edges.add(e)
        # cycles
        parent = {v: v for v in alive}

        def fnd(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        for e in rest_edges:
            i, j = self.ij(e)
            a, b = fnd(i), fnd(j)
            if a != b:
                parent[a] = b
        comps = {}
        for v in alive:
            comps.setdefault(fnd(v), []).append(v)
        out = list(forced)
        hard = []
        for vs in comps.values():
            if len(vs) % 2:
                return None
            if all(len(adjl[v]) == 2 for v in vs):
                # walk the cycle
                start = vs[0]
                order = [start]
                prev, cur = None, start
                while True:
                    nxt = [u for u in adjl[cur] if u != prev]
                    if prev is None:
                        nxt = nxt[:1]
                    u = nxt[0]
                    if u == start:
                        break
                    order.append(u)
                    prev, cur = cur, u
                assert len(order) == len(vs)
                L = len(order)
                m1 = [(order[k], order[k + 1]) for k in range(0, L, 2)]
                m2 = [(order[k + 1], order[(k + 2) % L]) for k in range(0, L, 2)]
                c1 = sum(cexact[adjl[a][b]] for a, b in m1)
                c2 = sum(cexact[adjl[a][b]] for a, b in m2)
                mm = m1 if c1 <= c2 else m2
                out.extend((min(a, b), max(a, b)) for a, b in mm)
            else:
                hard.extend(e for v in vs for u, e in adjl[v].items() if v < u)
        if hard:
            sub = self._nx_match_raw(hard, cexact, set(v for e in hard for v in self.ij(e)))
            if sub is None:
                return None
            out.extend(sub)
        out = sorted(out)
        assert sorted(v for p in out for v in p) == list(range(N))
        return out

    def _nx_match_raw(self, sel, cexact, verts):
        """networkx integer blossom per connected component on vertex set verts."""
        import networkx as nx
        N = self.N
        sel = list(sel)
        parent = list(range(N))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        for e in sel:
            i, j = self.ij(e)
            a, b = find(i), find(j)
            if a != b:
                parent[a] = b
        W0 = max(cexact[e] for e in sel) + 1
        comps = {}
        for e in sel:
            i, j = self.ij(e)
            comps.setdefault(find(i), []).append((i, j, W0 - cexact[e]))
        covered = set()
        pairs = []
        for edges in comps.values():
            G = nx.Graph()
            for i, j, w in edges:
                assert type(w) is int and w > 0
                G.add_edge(i, j, weight=w)
            mate = nx.max_weight_matching(G, maxcardinality=True)
            if 2 * len(mate) != G.number_of_nodes():
                return None
            for a, b in mate:
                pairs.append((min(a, b), max(a, b)))
            covered.update(G.nodes())
        if covered != set(verts):
            return None
        return sorted(pairs)

    def oracle(self, P, Q, k0=None):
        t0 = time.time()
        N = self.N
        if Q == 0:
            # min sum g^2: h(g)=g^2 strictly increasing & convex on g>0 => nested
            # pairs uncross (no increase) and crossing pairs separate (strict
            # decrease), so the unique minimiser is the adjacent pairing.
            pairs = [(k, k + 1) for k in range(0, N, 2)]
            A, B = self.AB(pairs)
            self.calls.append(dict(t=time.time() - t0, analytic=True, t_lp=0.0, t_nx=0.0,
                                   need=self.n, ncand=self.n, nS0=0, lp_integral=True,
                                   used_nx=False))
            return pairs, A, B, P * A - Q * B
        K = self.K
        iu, ju = self.iu_np, self.ju_np
        p = float(P)
        q = float(Q) / 2.0 ** K
        gf = self.gf
        craw = p * gf * gf - q * gf
        sig = math.frexp(float(np.abs(craw).max()))[1]
        scale = 2.0 ** sig
        cf = craw / scale
        # rebuild a bloated model (keeps only last call's tight columns and active cuts)
        if len(self.pool) > self.pool_cap or len(self.cuts) > self.cut_cap:
            self.rebuilds += 1
            act_cuts = [self.cuts[k] for k in getattr(self, 'last_active', [])
                        if k < len(self.cuts)]
            self._new_model(act_cuts)
            if len(self.keep):
                self._add_cols(self.keep, cf)
        # new costs on existing pool, then seed columns (k0 cheapest per vertex)
        if len(self.pool):
            self.h.changeColsCost(len(self.pool), np.arange(len(self.pool), dtype=np.int32),
                                  cf[self.pool])
        kk = min(self.k0, N - 1)
        Cm = np.full((N, N), np.inf)
        Cm[iu, ju] = cf
        Cm[ju, iu] = cf
        nb = np.argpartition(Cm, kk - 1, axis=1)[:, :kk]
        del Cm
        ii = np.repeat(np.arange(N), kk)
        jj = nb.ravel()
        a = np.minimum(ii, jj)
        b = np.maximum(ii, jj)
        seed = np.unique(np.concatenate([a * N - a * (a + 1) // 2 + (b - a - 1), self.adj]))
        self._add_cols(seed, cf)
        rounds = 0
        ncuts0 = len(self.cuts)
        integral = False
        while True:
            rounds += 1
            x, y, z = self._solve()
            zc = np.maximum(z, 0.0)
            rf = (cf - y[iu]) - y[ju]
            for k in np.nonzero(zc > 0)[0].tolist():
                mem = self.cuts[k]
                rf -= zc[k] * (mem[iu] != mem[ju])
            viol = np.nonzero((rf < -1e-9) & ~self.in_pool)[0]
            if len(viol) and rounds <= 60:
                vo = viol[np.argsort(rf[viol])][:8 * N]
                self._add_cols(vo, cf)
                continue
            odd, integral = self._odd_components(x)
            if odd and rounds <= self.cut_rounds:
                for mem in odd:
                    self._add_cut(mem)
                continue
            break
        t_lp = time.time() - t0
        # ---------------- exact phase ----------------
        act = np.nonzero(zc > 0)[0].tolist()
        self.last_active = act
        mk = min(len(rf), 4 * N)
        self.keep = np.argpartition(rf, mk - 1)[:mk]
        T = (p * gf * gf + q * np.abs(gf)) / scale + np.abs(y[iu]) + np.abs(y[ju]) \
            + float(zc.sum())
        margin = 1e-13 * (T + 1.0)
        shift = 2 * K + sig
        two_shift = Fraction(2) ** shift
        Y = [math.floor(Fraction(float(v)) * two_shift) for v in y]
        Z = {k: math.floor(Fraction(float(zc[k])) * two_shift) for k in act}
        cex = {}

        def cexact(e):
            v = cex.get(e)
            if v is None:
                g = self.Gx(e)
                v = P * g * g - Q * g
                cex[e] = v
            return v

        def rexact(e):
            i, j = self.ij(e)
            r = cexact(e) - Y[i] - Y[j]
            for k in act:
                mem = self.cuts[k]
                if mem[i] != mem[j]:
                    r -= Z[k]
            return r

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
        LB = sum(Y) + sum(Z.values())
        pairs0 = None
        if integral:
            Em = self.pool[x > 0.5]
            if len(Em) == self.n:
                pairs0 = sorted(self.ij(e) for e in Em.tolist())
                if sorted(v for pr in pairs0 for v in pr) != list(range(N)):
                    pairs0 = None
        if pairs0 is None:
            m = 3 * N
            sel = set(np.argpartition(rf, m - 1)[:m].tolist())
            sel.update(self.adj.tolist())
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
        need = [e for e in cand if rexact(e) <= gap]
        deg = np.zeros(N, dtype=np.int64)
        for e in need:
            i, j = self.ij(e)
            deg[i] += 1
            deg[j] += 1
        t_nx = 0.0
        if np.all(deg == 1):
            pairs = sorted(self.ij(e) for e in need)   # forced: unique candidate
            used_nx = False
        else:
            t1 = time.time()
            pairs = self._nx_match(need, cex)
            t_nx = time.time() - t1
            used_nx = True
        assert pairs is not None
        UB = 0
        for i, j in pairs:
            UB += cexact(self.eidx(i, j))
        assert UB <= UB0
        A, B = self.AB(pairs)
        assert P * A - Q * B == UB
        info = dict(t=time.time() - t0, t_lp=t_lp, t_nx=t_nx, rounds=rounds, rebuilds=self.rebuilds,
                    ncols=len(self.pool), ncuts=len(self.cuts), new_cuts=len(self.cuts) - ncuts0,
                    nS0=len(S0), nneg=nneg, lp_integral=integral, ncand=len(cand),
                    need=len(need), used_nx=used_nx)
        self.calls.append(info)
        return pairs, A, B, UB
