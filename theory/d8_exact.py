"""Door 8: certified exact minimum-weight perfect matching of sorted levels with pair cost (g - delta)^2, g = |lam_j - lam_i|,
and the exact one-qubit peeling of arXiv 2606.10004 built on it.

Oracle (certified_matching): Edmonds' perfect-matching LP on a sparse edge set (degree equalities + odd-set cuts), with
column generation over ALL N(N-1)/2 pairs (reduced costs computed exactly in numpy), and a final reduced-cost-fixing MILP.
Certificate: dual (y, z) feasible for every pair to tolerance eps (min reduced cost >= -eps), lower bound
L = sum y + sum z (|S|-1)/2, an integer perfect matching of cost V, and either V - L <= tol or V is the HiGHS-optimal MILP
over every edge whose reduced cost is <= V - L (no edge outside that set can occur in a matching of cost < V, because
c(M) = L + sum_{e in M} rc_e + nonnegative cut-slack terms). Reported: V, L, gap, min reduced cost, #edges, #cuts, route.

Peeling (2606.10004 Sec. 2.2-2.4 as specified in theory/D8-SPEC.md): with the qubit frozen at A = (-q, +q), the exact
optimum over the rest spectrum B of the sorted-level loss sum_i (lam_i - sort({B_j -+ q})_i)^2 is
min_M sum_pairs (g - 2q)^2 / 2 with B_pair = pair mean. Their selection argmin_q loss/Tr-weight (loss / (2 q^2)) is, over
all q and all M, min_M (n - B^2/A) with E_q = 2q = A/B (A = sum g^2, B = sum g, n = #pairs): the 'Cao ratio', solved by
Dinkelbach + the 1-D chord branch-and-bound of qubit_exact.solve_k. The next level's spectrum is the fitted B (pair means).
Run from probe/:  py ../theory/d8_exact.py VALIDATE | PEEL <seed> [d]"""
import sys, os, json, time, numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
from scipy.sparse import csc_matrix, coo_matrix
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qubit_exact as QE

EPS_RC = 1e-10
SEED_SCAN = 16
REBUILD_COLS = 40          # rebuild the warm LP when it holds more than REBUILD_COLS * N columns
STATS = []


def pair_cost(lam, i, j, delta):
    return (np.abs(lam[j] - lam[i]) - delta) ** 2


def initial_edges(lam, delta, k_near=3, k_far=3):
    N = len(lam); E = set()
    for i in range(N):
        for j in range(i + 1, min(N, i + 1 + k_near)): E.add((i, j))
        for target in (lam[i] + delta, lam[i] - delta):
            p = int(np.searchsorted(lam, target))
            for j in range(max(0, p - k_far), min(N, p + k_far + 1)):
                if j != i: E.add((min(i, j), max(i, j)))
    for i in range(0, N - 1, 2): E.add((i, i + 1))            # a feasible perfect matching is always present
    return E


def solve_lp(lam, delta, edges, cuts):
    N = len(lam); E = np.array(sorted(edges)); m = len(E)
    c = pair_cost(lam, E[:, 0], E[:, 1], delta)
    rows = np.concatenate([E[:, 0], E[:, 1]]); cols = np.concatenate([np.arange(m), np.arange(m)])
    Aeq = csc_matrix((np.ones(2 * m), (rows, cols)), shape=(N, m))
    if cuts:
        r, cc = [], []
        for k, S in enumerate(cuts):
            inS = np.zeros(N, bool); inS[list(S)] = True
            idx = np.nonzero(inS[E[:, 0]] & inS[E[:, 1]])[0]; r += [k] * len(idx); cc += list(idx)
        Aub = csc_matrix((np.ones(len(r)), (r, cc)), shape=(len(cuts), m)); bub = np.array([(len(S) - 1) / 2 for S in cuts])
    else: Aub, bub = None, None
    res = linprog(c, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=np.ones(N), bounds=(0, None), method="highs-ds",
                  options=dict(dual_feasibility_tolerance=1e-10, primal_feasibility_tolerance=1e-10))
    assert res.status == 0, res.message
    y = res.eqlin.marginals; z = res.ineqlin.marginals if cuts else np.zeros(0)
    return E, c, res.x, y, z, res.fun


def reduced_costs_all(lam, delta, y, z, cuts, block=512):
    """Exact reduced cost of every pair i < j; returns the global minimum and the list of the most negative pairs."""
    N = len(lam); member = np.zeros((len(cuts), N), bool)
    for k, S in enumerate(cuts): member[k, list(S)] = True
    mins = []; worst = []
    for a in range(0, N, block):
        I = np.arange(a, min(N, a + block))
        G = np.abs(lam[None, :] - lam[I, None]); rc = (G - delta) ** 2 - y[I, None] - y[None, :]
        for k in range(len(cuts)):
            if z[k] != 0: rc -= z[k] * (member[k, I][:, None] & member[k][None, :])
        tri = np.arange(N)[None, :] > I[:, None]; rc = np.where(tri, rc, np.inf)
        mins.append(rc.min())
        neg = np.argwhere(rc < -EPS_RC)
        if len(neg): worst.append(np.column_stack([I[neg[:, 0]], neg[:, 1], rc[neg[:, 0], neg[:, 1]]]))
    W = np.vstack(worst) if worst else np.zeros((0, 3))
    return float(min(mins)), W


def edges_below(lam, delta, y, z, cuts, thr, block=512, cap=400000):
    N = len(lam); member = np.zeros((len(cuts), N), bool)
    for k, S in enumerate(cuts): member[k, list(S)] = True
    out = []
    for a in range(0, N, block):
        I = np.arange(a, min(N, a + block))
        G = np.abs(lam[None, :] - lam[I, None]); rc = (G - delta) ** 2 - y[I, None] - y[None, :]
        for k in range(len(cuts)):
            if z[k] != 0: rc -= z[k] * (member[k, I][:, None] & member[k][None, :])
        tri = np.arange(N)[None, :] > I[:, None]
        sel = np.argwhere(tri & (rc <= thr)); out.append(np.column_stack([I[sel[:, 0]], sel[:, 1]]))
        if sum(len(o) for o in out) > cap: return None
    return np.vstack(out)


def odd_components(E, x, N):
    frac = np.nonzero((x > 1e-7) & (x < 1 - 1e-7))[0]
    parent = list(range(N))
    def f(a):
        while parent[a] != a: parent[a] = parent[parent[a]]; a = parent[a]
        return a
    nodes = set()
    for e in frac:
        u, v = E[e]; nodes.add(u); nodes.add(v); parent[f(u)] = f(v)
    comps = {}
    for u in nodes: comps.setdefault(f(u), []).append(u)
    return [tuple(sorted(c)) for c in comps.values() if len(c) % 2 == 1 and len(c) >= 3]


def milp_on(lam, delta, E):
    N = len(lam); m = len(E); c = pair_cost(lam, E[:, 0], E[:, 1], delta)
    rows = np.concatenate([E[:, 0], E[:, 1]]); cols = np.concatenate([np.arange(m), np.arange(m)])
    A = csc_matrix((np.ones(2 * m), (rows, cols)), shape=(N, m))
    res = milp(c, constraints=LinearConstraint(A, 1, 1), integrality=np.ones(m), bounds=Bounds(0, 1),
               options=dict(mip_rel_gap=0.0, disp=False))
    assert res.status == 0, res.message
    x = np.round(res.x); sel = E[x > 0.5]
    return [tuple(map(int, p)) for p in sel], float(c[x > 0.5].sum())


def certified_matching(lam, delta, max_rounds=200):
    t0 = time.time(); N = len(lam); lam = np.asarray(lam, float)
    edges = initial_edges(lam, delta); cuts = []; rounds = 0
    while True:
        rounds += 1; assert rounds <= max_rounds, "no convergence"
        E, c, x, y, z, fun = solve_lp(lam, delta, edges, cuts)
        rcmin, W = reduced_costs_all(lam, delta, y, z, cuts)
        if len(W):                                                       # column generation: only edges not yet in the LP
            W = W[[(int(i), int(j)) not in edges for i, j, _ in W]] if len(W) < 2_000_000 else W
            if len(W):
                order = np.argsort(W[:, 2])[: 8 * N]
                for i, j, _ in W[order]: edges.add((int(i), int(j)))
                continue
        # dual objective, corrected by the residual dual infeasibility: every perfect matching has N/2 edges, so
        # c(M) >= L + (N/2) * min(0, rcmin) is a valid lower bound even if HiGHS left rc slightly negative.
        L = float(y.sum() + (z * np.array([(len(S) - 1) / 2 for S in cuts])).sum()) if cuts else float(y.sum())
        L = L + (N / 2) * min(0.0, rcmin)
        if np.all((x < 1e-7) | (x > 1 - 1e-7)):                          # integral LP optimum over the full graph
            pairs = [tuple(map(int, p)) for p in E[x > 0.5]]; V = float(c[x > 0.5].sum()); route = "LP-integral"
            break
        new = [S for S in odd_components(E, x, N) if S not in cuts]
        if new and len(cuts) < 4 * N: cuts += new; continue
        pairs, V = milp_on(lam, delta, E)                                # reduced-cost fixing
        E2 = edges_below(lam, delta, y, z, cuts, V - L + 1e-9)
        assert E2 is not None, "reduced-cost-fixing set too large"
        pairs, V = milp_on(lam, delta, E2); route = f"RCF-MILP({len(E2)} edges)"
        break
    pairs = sorted(pairs); assert len(pairs) == N // 2 and len({u for p in pairs for u in p}) == N
    g = np.array([abs(lam[j] - lam[i]) for i, j in pairs]); Vchk = float(((g - delta) ** 2).sum())
    cert = dict(N=N, V=Vchk, L=L, gap=Vchk - L, rcmin=rcmin, edges=len(edges), cuts=len(cuts), rounds=rounds, route=route,
                secs=round(time.time() - t0, 3))
    TOL = 1e-9 * max(1.0, abs(Vchk)) + 1e-9          # certified acceptance: V - L <= TOL (L already absorbs rc noise)
    cert["tol"] = TOL
    assert rcmin >= -1e-8 and (Vchk - L <= TOL or route.startswith("RCF")), cert
    STATS.append(cert)
    return pairs, float(g.dot(g)), float(g.sum())


def exact_cao_split(lam):
    """Exact Cao-ratio split: min over perfect matchings of n - B^2/A; returns E_q = A/B, pairs, ratio, rest = pair means."""
    lam = np.sort(np.asarray(lam, float)); n = len(lam) // 2
    QE.best_matching = warm_matching                                      # inject the certified (warm-started) oracle
    cache = {}
    M0 = warm_matching(lam, 0.0); cache[0.0] = M0
    # Seed: a coarse delta scan gives a near-optimal incumbent, so Dinkelbach starts close and the chord
    # branch-and-bound prunes early. (Does not affect exactness: every candidate is still certified and the
    # B&B still covers [0, spread].)
    spread = lam[-1] - lam[0]
    for dl in np.linspace(0.05, 0.95, SEED_SCAN) * spread:
        if dl not in cache: cache[dl] = warm_matching(lam, dl)
    c = min(M[1] / M[2] ** 2 for M in cache.values() if M[2] > 0)
    for _ in range(80):                                                   # Dinkelbach on A - c B^2
        Mr, vr, _ = QE.solve_k(lam, 1.0 / c, cache)
        if vr >= -1e-13 * max(1.0, Mr[1]): break
        c = Mr[1] / Mr[2] ** 2
    best = min((M for M in cache.values() if M[2] > 0), key=lambda M: M[1] / M[2] ** 2)
    pairs, A, B = best
    rest = np.sort(np.array([(lam[i] + lam[j]) / 2 for i, j in pairs]))
    return dict(E_q=A / B, ratio=n - B * B / A, A=A, B=B, n=n), pairs, rest, len(cache)


def validate():
    import networkx as nx
    rng = np.random.default_rng(20261010); worst = 0.0; rows = 0
    for N in (8, 16, 32, 64, 128, 256):
        for t in range(12 if N <= 64 else 4):
            lam = np.sort(rng.standard_normal(N)); spread = lam[-1] - lam[0]
            for delta in rng.uniform(0, spread, 3):
                p1, A1, B1 = certified_matching(lam, delta); v1 = A1 - 2 * delta * B1
                p2, A2, B2 = NXREF(lam, delta)
                v2 = A2 - 2 * delta * B2; worst = max(worst, (v1 - v2) / max(1.0, abs(v2))); rows += 1
        print(f"N={N}: cases {rows}, worst (certified - networkx)/scale = {worst:.2e}; last cert {STATS[-1]}", flush=True)
    print("VALIDATE:", "PASS" if worst <= 1e-9 else "FAIL", f"worst {worst:.2e} over {rows} cases")


NXREF = QE.best_matching                                                  # the original networkx blossom oracle


def peel(lam, max_levels=None):
    out = []; spec = np.sort(lam); lvl = 0
    while len(spec) > 2 and (max_levels is None or lvl < max_levels):
        t0 = time.time(); nst = len(STATS)
        info, pairs, rest, ncache = exact_cao_split(spec)
        info.update(level=lvl, d=len(spec), oracle_calls=len(STATS) - nst, secs=round(time.time() - t0, 1),
                    max_gap=max(s["gap"] for s in STATS[nst:]), min_rc=min(s["rcmin"] for s in STATS[nst:]),
                    routes=sorted(set(s["route"].split("(")[0] for s in STATS[nst:])))
        out.append(info); print(json.dumps(info), flush=True)
        spec = rest; lvl += 1
    if len(spec) == 2: out.append(dict(level=lvl, d=2, E_q=float(spec[1] - spec[0]), final=True))
    return out


def gue_spectrum(d, seed):
    """2606.10004 convention (D8-SPEC sec 1): <|H_ij|^2> = 1/d, traceless. Own RNG stream (their exact matrix is not reproducible)."""
    rng = np.random.default_rng(seed)
    X = (rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))) / np.sqrt(d); H = 0.5 * (X + X.conj().T)
    H -= np.trace(H).real / d * np.eye(d)
    return np.linalg.eigvalsh(H)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "VALIDATE": validate()
    elif mode == "PEEL":
        seed = int(sys.argv[2]); d = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
        lam = gue_spectrum(d, seed); res = peel(lam)
        json.dump(dict(seed=seed, d=d, levels=res, oracle_stats=STATS), open(os.path.join(QE.t1.OUT, f"d8_peel_d{d}_s{seed}.json"), "w"))


# ---------------------------------------------------------------------------------------------------------------------------
# Warm-started oracle: one persistent HiGHS LP per spectrum (scipy's bundled HiGHS API). A change of delta is a cost update;
# columns found by pricing are appended and the simplex restarts from the current basis; odd-set cuts are valid for every
# delta and stay. Certificate exactly as in certified_matching.
from scipy.optimize._highspy import _core as HC
_WARM = {}


class WarmMatcher:
    def __init__(self, lam):
        self.lam = np.asarray(lam, float); self.N = N = len(lam)
        self.h = h = HC._Highs()
        for k, v in (("output_flag", False), ("primal_feasibility_tolerance", 1e-10), ("dual_feasibility_tolerance", 1e-10),
                     ("presolve", "off"), ("solver", "simplex"), ("simplex_scale_strategy", 0)):
            h.setOptionValue(k, v)
        h.addRows(N, np.ones(N), np.ones(N), 0, np.zeros(N, np.int32), np.zeros(0, np.int32), np.zeros(0))
        self.cols = []; self.colset = set(); self.cuts = []; self.member = np.zeros((0, N), bool); self.delta = None

    def _add_cols(self, pairs, delta):
        pairs = [p for p in pairs if p not in self.colset]
        if not pairs: return 0
        starts, idx = [], []
        for (u, v) in pairs:
            starts.append(len(idx)); r = [u, v]
            if len(self.cuts):
                r += list(np.nonzero(self.member[:, u] & self.member[:, v])[0] + self.N)
            idx += r
        P = np.array(pairs); cost = pair_cost(self.lam, P[:, 0], P[:, 1], delta)
        self.h.addCols(len(pairs), cost, np.zeros(len(pairs)), np.full(len(pairs), np.inf), len(idx),
                       np.array(starts, np.int32), np.array(idx, np.int32), np.ones(len(idx)))
        self.cols += pairs; self.colset.update(pairs); return len(pairs)

    def _add_cuts(self, sets):
        for S in sets:
            inS = np.zeros(self.N, bool); inS[list(S)] = True
            C = np.array(self.cols); cidx = np.nonzero(inS[C[:, 0]] & inS[C[:, 1]])[0].astype(np.int32)
            self.h.addRows(1, np.array([-np.inf]), np.array([(len(S) - 1) / 2]), len(cidx), np.array([0], np.int32), cidx, np.ones(len(cidx)))
            self.cuts.append(tuple(S)); self.member = np.vstack([self.member, inS[None, :]])

    def solve(self, delta, max_rounds=400):
        t0 = time.time(); N = self.N; lam = self.lam
        if not self.cols: self._add_cols(sorted(initial_edges(lam, delta)), delta)
        else:
            self._add_cols(sorted(initial_edges(lam, delta)), delta)
            C = np.array(self.cols); self.h.changeColsCost(len(C), np.arange(len(C), dtype=np.int32), pair_cost(lam, C[:, 0], C[:, 1], delta))
        rounds = 0; route = None; self._cr = 0
        while True:
            rounds += 1; assert rounds <= max_rounds, "no convergence"
            self.h.run(); st = self.h.getModelStatus()
            if self.h.modelStatusToString(st) != "Optimal":
                self.h.clearSolver(); self.h.run(); st = self.h.getModelStatus()
                if self.h.modelStatusToString(st) != "Optimal": raise RuntimeError("HiGHS status " + self.h.modelStatusToString(st))
                self.status_retries = getattr(self, "status_retries", 0) + 1
            sol = self.h.getSolution(); x = np.array(sol.col_value); rd = np.array(sol.row_dual)
            y = rd[:N]; z = rd[N:]
            rcmin, W = reduced_costs_all(lam, delta, y, z, self.cuts)
            if len(W):
                W = W[np.argsort(W[:, 2])]
                new = []; per = np.zeros(N, int)
                for i, j, _ in W:                                    # at most 6 new columns per node per round
                    i, j = int(i), int(j)
                    if (i, j) in self.colset or per[i] >= 6 or per[j] >= 6: continue
                    new.append((i, j)); per[i] += 1; per[j] += 1
                    if len(new) >= 4 * N: break
                if new: self._add_cols(new, delta); continue
            L = float(y.sum() + (z * np.array([(len(S) - 1) / 2 for S in self.cuts])).sum()) if self.cuts else float(y.sum())
            L += (N / 2) * min(0.0, rcmin)
            C = np.array(self.cols); c = pair_cost(lam, C[:, 0], C[:, 1], delta)
            if np.all((x < 1e-7) | (x > 1 - 1e-7)):
                pairs = [tuple(map(int, p)) for p in C[x > 0.5]]
                g = np.array([abs(lam[j] - lam[i]) for i, j in pairs]); Vt = float(((g - delta) ** 2).sum())
                if Vt - L > 1e-9 * max(1.0, abs(Vt)) + 1e-9 and not getattr(self, "_cold", False):
                    # HiGHS can return 'Optimal' after a warm start with a residual dual infeasibility (seen: 4.4e-9 with
                    # tolerance 1e-10). Discard the basis once and re-solve cold to get clean duals for the certificate.
                    self._cold = True; self.h.clearSolver(); self.cold_resolves = getattr(self, "cold_resolves", 0) + 1; continue
                if Vt - L > 1e-9 * max(1.0, abs(Vt)) + 1e-9:
                    # still not closed (rc noise x N/2 exceeds the tolerance when V is tiny): prove optimality by
                    # reduced-cost fixing instead; edges with rc > V - L cannot occur in a matching cheaper than V.
                    self._cold = False
                    E2 = edges_below(lam, delta, y, z, self.cuts, Vt - L + 1e-12); assert E2 is not None, "RCF set too large"
                    pairs, V = milp_on(lam, delta, E2); route = f"RCF-MILP({len(E2)} edges)"; break
                self._cold = False; route = "LP-integral"; break
            newcuts = [S for S in odd_components(C, x, N) if S not in self.cuts]
            cut_rounds = getattr(self, '_cr', 0)
            if newcuts and cut_rounds < 25: self._cr = cut_rounds + 1; self._add_cuts(newcuts); continue
            self._cr = 0
            # Upper bound from any perfect matching (no MILP over all LP columns): the best of the matchings this LP returned at
            # other deltas, and the adjacent pairing, evaluated at the current delta. Exactness is unchanged: the MILP below runs
            # over every edge with reduced cost <= V_ub - L, which contains every matching cheaper than V_ub.
            cands = list(getattr(self, "prev", [])) + [[(i, i + 1) for i in range(0, N - 1, 2)]]
            def cost_of(P): P = np.array(P); return float(pair_cost(lam, P[:, 0], P[:, 1], delta).sum())
            V_ub = min(cost_of(P) for P in cands)
            E2 = edges_below(lam, delta, y, z, self.cuts, V_ub - L + 1e-9)
            if E2 is None:                                                  # bound too loose: old route
                pairs, V = milp_on(lam, delta, C)
                E2 = edges_below(lam, delta, y, z, self.cuts, V - L + 1e-9); assert E2 is not None, "RCF set too large"
            pairs, V = milp_on(lam, delta, E2); route = f"RCF-MILP({len(E2)} edges)"; break
        pairs = sorted(pairs); assert len(pairs) == N // 2 and len({u for p in pairs for u in p}) == N
        g = np.array([abs(lam[j] - lam[i]) for i, j in pairs]); V = float(((g - delta) ** 2).sum())
        TOL = 1e-9 * max(1.0, abs(V)) + 1e-9
        cert = dict(N=N, V=V, L=L, gap=V - L, rcmin=rcmin, edges=len(self.cols), cuts=len(self.cuts), rounds=rounds, route=route,
                    secs=round(time.time() - t0, 3), tol=TOL, warm=True, cold_resolves=getattr(self, "cold_resolves", 0))
        assert rcmin >= -1e-8 and (V - L <= TOL or route.startswith("RCF")), cert
        STATS.append(cert)
        self.prev = (list(getattr(self, 'prev', [])) + [pairs])[-40:]
        return pairs, float(g.dot(g)), float(g.sum())


def warm_matching(lam, delta):
    key = (len(lam), float(lam[0]), float(lam[-1]), float(np.sum(lam * lam)))
    if key not in _WARM or len(_WARM[key].cols) > REBUILD_COLS * len(lam):
        if len(_WARM) > 4: _WARM.clear()
        _WARM[key] = WarmMatcher(lam)
    try:
        return _WARM[key].solve(delta)
    except RuntimeError:
        _WARM[key] = WarmMatcher(lam)                       # fresh LP once
        try: return _WARM[key].solve(delta)
        except RuntimeError: return certified_matching(lam, delta)   # cold scipy route (validated 168/168)
