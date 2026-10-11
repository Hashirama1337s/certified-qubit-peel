"""Implementation B: exact global minimum of R(M) = n - B(M)^2 / A(M) over all
perfect matchings M of a sorted spectrum (independent implementation).

Method (see report):
  * f(M) = B^2/A is a convex function of the point (B, A), so its maximum over
    the finite point set {(B_M, A_M)} is attained at a vertex of the convex hull,
    and (tangent argument) at a vertex of the lower chain whose supporting
    direction minimises A - s*B for some s in (0, inf).
  * Every query "minimise P*A - Q*B" (P, Q >= 0) is a min-weight perfect matching
    with edge cost P*g^2 - Q*g. It is solved EXACTLY in integer arithmetic:
    levels are converted to exact integers L_i = lam_i * 2^K; LP duals from HiGHS
    (fractional matching LP) are rounded to integers and repaired so every
    reduced cost r_e = c_e - y_u - y_v >= 0 exactly; then any perfect matching
    M satisfies c(M) = sum(y) + sum_{e in M} r_e, so an optimum can only use
    edges with r_e <= UB - sum(y). The restricted graph is solved by networkx's
    blossom algorithm with Python-int weights (exact integer arithmetic, with
    networkx's own verifyOptimum()).
  * Hull exploration (Eisner-Severance style) with exact Fractions; a triangle
    between consecutive known chain vertices is pruned only if f at its apex
    is <= the incumbent (f is convex, so its max over the triangle is at a
    vertex of the triangle).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import math
import time
from fractions import Fraction
from decimal import Decimal, getcontext

import numpy as np
import networkx as nx
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

getcontext().prec = 60


def spectrum(d, seed):
    rng = np.random.default_rng(seed)
    X = (rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))) / np.sqrt(d)
    H = 0.5 * (X + X.conj().T)
    H -= np.trace(H).real / d * np.eye(d)
    return np.linalg.eigvalsh(H)


def to_exact_ints(lam):
    """lam_i = L_i / 2**K exactly (float64 values are dyadic rationals)."""
    nums, ks = [], []
    for x in lam:
        p, q = float(x).as_integer_ratio()
        k = q.bit_length() - 1
        assert q == (1 << k)
        nums.append(p)
        ks.append(k)
    K = max(ks)
    return K, [p << (K - k) for p, k in zip(nums, ks)]


def fmt(fr, sig=15):
    """Format a Fraction to `sig` significant digits."""
    d = Decimal(fr.numerator) / Decimal(fr.denominator)
    return format(d, '.%dg' % sig)


class Level:
    def __init__(self, lam, log=print):
        lam = np.asarray(lam, dtype=np.float64)
        assert np.all(np.diff(lam) > 0), "levels must be strictly increasing"
        self.lam = lam
        self.N = N = len(lam)
        assert N % 2 == 0
        self.n = N // 2
        self.K, self.L = to_exact_ints(lam)
        iu, ju = np.triu_indices(N, 1)
        self.iu = iu.tolist()
        self.ju = ju.tolist()
        L = self.L
        self.G = [L[j] - L[i] for i, j in zip(self.iu, self.ju)]
        self.G2 = [g * g for g in self.G]
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

    def AB(self, pairs):
        A = 0
        B = 0
        for i, j in pairs:
            e = self.eidx(i, j)
            A += self.G2[e]
            B += self.G[e]
        return A, B

    def _nx_match(self, sel, c):
        N = self.N
        Gx = nx.Graph()
        Gx.add_nodes_from(range(N))
        W0 = max(c[e] for e in sel) + 1
        iu, ju = self.iu, self.ju
        for e in sel:
            w = W0 - c[e]
            assert type(w) is int and w > 0
            Gx.add_edge(iu[e], ju[e], weight=w)
        mate = nx.max_weight_matching(Gx, maxcardinality=True)
        if len(mate) != self.n:
            return None
        return sorted(tuple(sorted(p)) for p in mate)

    def _lp(self, cf, E):
        """Fractional perfect-matching LP restricted to edge ids E (np array)."""
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
        """Exact argmin over perfect matchings of sum_e (P*G_e^2 - Q*G_e).

        Exactness: integer duals Y with every exact reduced cost
        r_e = c_e - Y_i - Y_j >= 0 give c(M) = sum(Y) + sum_{e in M} r_e for
        every perfect matching M; with UB = cost of any perfect matching, an
        optimal matching uses only edges with r_e <= UB - sum(Y). networkx's
        blossom algorithm (integer weights => exact integer arithmetic and its
        own verifyOptimum) is then run on that edge set.
        """
        t0 = time.time()
        N = self.N
        G, G2, iu, ju = self.G, self.G2, self.iu, self.ju
        K = self.K
        # ---- float LP with column generation (only to obtain good duals) ----
        p = float(P)
        q = float(Q) / 2.0 ** K
        gf = self.gf
        craw = p * gf * gf - q * gf          # ~ c_e / 2^(2K)
        sig = math.frexp(float(np.abs(craw).max()))[1]
        cf = craw / 2.0 ** sig               # ~ c_e / 2^(2K+sig)
        Cm = np.full((N, N), np.inf)
        Cm[self.iu_np, self.ju_np] = cf
        Cm[self.ju_np, self.iu_np] = cf
        kk = min(k0, N - 1)
        nb = np.argpartition(Cm, kk - 1, axis=1)[:, :kk]
        ii = np.repeat(np.arange(N), kk)
        jj = nb.ravel()
        a = np.minimum(ii, jj)
        b = np.maximum(ii, jj)
        Eset = set((a * N - a * (a + 1) // 2 + (b - a - 1)).tolist())
        for k in range(0, N, 2):
            Eset.add(self.eidx(k, k + 1))
        cg_rounds = 0
        inE = np.zeros(len(gf), dtype=bool)
        inE[list(Eset)] = True
        while True:
            cg_rounds += 1
            E = np.nonzero(inE)[0]
            x, y = self._lp(cf, E)
            rf = cf - y[self.iu_np] - y[self.ju_np]
            viol = np.nonzero((rf < -1e-9) & ~inE)[0]
            if len(viol) == 0 or cg_rounds > 50:
                break
            # add the most violated edges (at most 8N per round)
            vo = viol[np.argsort(rf[viol])][:8 * N]
            inE[vo] = True
        Eset = E
        t_lp = time.time() - t0
        # ---- exact integer phase ----
        shift = 2 * K + sig
        Y = [math.floor(Fraction(float(v)) * (Fraction(2) ** shift)) for v in y]
        c = [P * g2 - Q * g for g, g2 in zip(G, G2)]
        r = [ce - Y[i] - Y[j] for ce, i, j in zip(c, iu, ju)]
        dfc = [0] * N
        nneg = 0
        for e, re in enumerate(r):
            if re < 0:
                nneg += 1
                h = (-re + 1) // 2
                i, j = iu[e], ju[e]
                if h > dfc[i]:
                    dfc[i] = h
                if h > dfc[j]:
                    dfc[j] = h
        if nneg:
            Y = [a_ - b_ for a_, b_ in zip(Y, dfc)]
            r = [ce - Y[i] - Y[j] for ce, i, j in zip(c, iu, ju)]
        assert min(r) >= 0
        LB = sum(Y)
        # ---- an upper bound UB0 from a perfect matching ----
        xm = x > 0.5
        integral = bool(np.all((x < 1e-7) | (x > 1 - 1e-7)))
        pairs0 = None
        if integral:
            Em = E[xm]
            if len(Em) == self.n:
                pairs0 = sorted((iu[e], ju[e]) for e in Em.tolist())
                cov = sorted(v for pr in pairs0 for v in pr)
                if cov != list(range(N)):
                    pairs0 = None
        if pairs0 is None:
            rfx = np.array([float(v) for v in r])
            m = min(len(r), 3 * N)
            sel = set(np.argpartition(rfx, m - 1)[:m].tolist())
            for k in range(0, N, 2):
                sel.add(self.eidx(k, k + 1))
            pairs0 = self._nx_match(sel, c)
        UB0 = sum(c[self.eidx(i, j)] for i, j in pairs0)
        gap = UB0 - LB
        assert gap >= 0
        need = [e for e, re in enumerate(r) if re <= gap]
        t_nx = time.time()
        pairs = self._nx_match(need, c)
        t_nx = time.time() - t_nx
        assert pairs is not None
        UB = sum(c[self.eidx(i, j)] for i, j in pairs)
        assert UB <= UB0
        A, B = self.AB(pairs)
        assert P * A - Q * B == UB
        info = dict(t=time.time() - t0, t_lp=t_lp, t_nx=t_nx, cg=cg_rounds,
                    ncols=len(Eset), nneg=nneg, lp_integral=integral,
                    need=len(need), gap_rel=float(Fraction(gap, max(1, abs(UB0)))))
        self.calls.append(info)
        return pairs, A, B, UB

    # ---------------- hull search -----------------
    def f(self, A, B):
        return Fraction(B * B, A)

    def split(self, verbose=True):
        t0 = time.time()
        n = self.n
        pts = []  # each: dict(A,B,P,Q,pairs)

        def add_point(pairs, A, B, P, Q):
            for p in pts:
                if p['A'] == A and p['B'] == B:
                    return p, False
            p = dict(A=A, B=B, P=P, Q=Q, pairs=pairs, f=self.f(A, B))
            pts.append(p)
            pts.sort(key=lambda q: (q['B'], q['A']))
            return p, True

        # left end: global min of A (support direction (1,0))
        pr, A, B, _ = self.oracle(1, 0)
        add_point(pr, A, B, 1, 0)
        # right end: max B (all pairs cross the median), min A among those:
        # sorted pairing (i, i+n) by rearrangement; support (0,1): B <= Bmax
        pr = [(i, i + n) for i in range(n)]
        A, B = self.AB(pr)
        add_point(pr, A, B, 0, 1)
        best = max(pts, key=lambda q: q['f'])
        # Dinkelbach-type descent for an incumbent (each step is an exact
        # oracle call, so every point found is a lower-chain point with support)
        for _ in range(30):
            P, Q = best['B'], 2 * best['A']
            g = math.gcd(P, Q)
            P //= g
            Q //= g
            pr, A, B, _ = self.oracle(P, Q)
            p, new = add_point(pr, A, B, P, Q)
            if p['f'] > best['f']:
                best = p
            else:
                break
        if verbose:
            self.log("  N=%d after descent: %d oracle calls, R=%s" % (
                self.N, len(self.calls), fmt(n - best['f'], 14)))
        # triangle refinement
        done = set()  # frozenset of (id(U), id(V)) confirmed
        while True:
            best = max(pts, key=lambda q: q['f'])
            cand = []
            for U, V in zip(pts[:-1], pts[1:]):
                key = (id(U), id(V))
                if key in done:
                    continue
                Pu, Qu, Pv, Qv = U['P'], U['Q'], V['P'], V['Q']
                cu = Pu * U['A'] - Qu * U['B']
                cv = Pv * V['A'] - Qv * V['B']
                D = Qu * Pv - Pu * Qv
                if D == 0:
                    done.add(key)
                    continue
                Aq = Fraction(Qu * cv - Qv * cu, D)
                Bq = Fraction(Pu * cv - Pv * cu, D)
                if Aq <= 0:
                    fq = None  # +inf
                else:
                    fq = Bq * Bq / Aq
                if fq is not None and fq <= best['f']:
                    done.add(key)
                    continue
                cand.append((float('inf') if fq is None else float(fq), U, V))
            if not cand:
                break
            cand.sort(key=lambda t: -t[0])
            _, U, V = cand[0]
            P = V['B'] - U['B']
            Q = V['A'] - U['A']
            assert P > 0 and Q > 0
            g = math.gcd(P, Q)
            P //= g
            Q //= g
            val_seg = P * U['A'] - Q * U['B']
            assert val_seg == P * V['A'] - Q * V['B']
            pr, A, B, val = self.oracle(P, Q)
            if val < val_seg:
                p, new = add_point(pr, A, B, P, Q)
                assert new
            else:
                done.add((id(U), id(V)))
        best = max(pts, key=lambda q: q['f'])
        A, B = best['A'], best['B']
        R = n - best['f']
        Eq = Fraction(A, B * (1 << self.K))
        # cross-check with plain float64 evaluation
        g = np.array([self.lam[j] - self.lam[i] for i, j in best['pairs']])
        R_float = n - g.sum() ** 2 / (g ** 2).sum()
        # NB spec: (lam_i + lam_j)/2 in float64
        means = sorted((self.lam[i] + self.lam[j]) / 2 for i, j in best['pairs'])
        out = dict(N=self.N, n=n, R=R, Eq=Eq, R_str=fmt(R, 15), Eq_str=fmt(Eq, 15),
                   R_float=float(R_float), pairs=best['pairs'],
                   n_chain_points=len(pts), n_oracle=len(self.calls),
                   time=time.time() - t0, rest=np.array(means),
                   oracle_info=self.calls,
                   chain=[dict(B=fmt(Fraction(p['B'], 1 << self.K), 15),
                               A=fmt(Fraction(p['A'], 1 << (2 * self.K)), 15),
                               R=fmt(n - p['f'], 15)) for p in pts])
        return out


def peel(lam, log=print, stop_after=None):
    levels = []
    cur = np.asarray(lam, dtype=np.float64)
    while len(cur) > 2:
        lv = Level(cur, log=log)
        out = lv.split()
        log("N=%4d  Eq=%s  R*=%s  oracle=%d  chain=%d  t=%.1fs" % (
            out['N'], out['Eq_str'], out['R_str'], out['n_oracle'],
            out['n_chain_points'], out['time']))
        levels.append(out)
        cur = out['rest']
        if stop_after is not None and len(levels) >= stop_after:
            return levels, None
    final = float(cur[1] - cur[0])
    return levels, final
