"""BAR-T2 E1: exact minimal-interaction QUBIT split (2 x n grid) by parametric min-weight perfect matching.
min C1 * ||lam||^2 = (1/2) min_M (A_M - B_M^2/n), A = sum g^2, B = sum g over the n pairs (g = |gap|); optimum is a vertex of
the lower convex hull of {(B_M, A_M)}, found by sweeping delta in min_M sum (g - delta)^2  (Eisner-Severance).
Run from probe/:  py ../theory/qubit_exact.py E1a | E1b"""
import sys, os, json, time, numpy as np, networkx as nx
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "probe"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t1


def best_matching(lam, delta):
    """Exact min-weight perfect matching for pair cost (|lam_i - lam_j| - delta)^2 (blossom via networkx)."""
    N = len(lam); G = nx.Graph(); big = (lam.max() - lam.min() + abs(delta)) ** 2 + 1.0
    for i in range(N):
        for j in range(i + 1, N):
            G.add_edge(i, j, weight=big - (abs(lam[j] - lam[i]) - delta) ** 2)
    M = nx.max_weight_matching(G, maxcardinality=True)
    pairs = [tuple(sorted(e)) for e in M]
    assert len(pairs) == N // 2
    g = np.array([abs(lam[j] - lam[i]) for i, j in pairs]); return pairs, g.dot(g), g.sum()


def hull_sweep(lam, tol=1e-12):
    """All lower-hull vertices of {(B_M, A_M)} via parametric search on delta in [0, spread]."""
    spread = lam.max() - lam.min(); verts = {}
    def q(delta):
        pairs, A, B = best_matching(lam, delta); key = tuple(sorted(pairs)); verts[key] = (A, B, pairs); return A, B, key
    lo, hi = q(0.0), q(spread); stack = [(0.0, lo, spread, hi)]; nq = 2
    while stack:
        dl, (Al, Bl, kl), dh, (Ah, Bh, kh) = stack.pop()
        if kl == kh or abs(Bh - Bl) < 1e-15: continue
        ds = (Ah - Al) / (2 * (Bh - Bl))                    # lines A - 2 d B tie here
        if not (dl < ds < dh): continue
        Am, Bm, km = q(ds); nq += 1
        if (Am - 2 * ds * Bm) < (Al - 2 * ds * Bl) - tol * max(1.0, abs(Al)):
            stack.append((dl, (Al, Bl, kl), ds, (Am, Bm, km))); stack.append((ds, (Am, Bm, km), dh, (Ah, Bh, kh)))
    return verts, nq


def solve_k(lam, k, cache, tol=1e-13):
    """Exact min over matchings of (A - B^2/k) = min_delta [h(delta) + k delta^2], h(delta) = min_M (A_M - 2 delta B_M)
    concave piecewise linear. 1-D branch-and-bound: on [d1, d2], h >= chord (concavity), so the bound is
    min_delta k delta^2 + chord(delta); if both ends have the same optimal M, h is exactly that line there (prune)."""
    spread = lam.max() - lam.min()
    def q(d):
        if d not in cache: cache[d] = best_matching(lam, d)
        return cache[d]
    def val(M): return M[1] - M[2] ** 2 / k
    inc = None
    def upd(M):
        nonlocal inc
        if inc is None or val(M) < val(inc): inc = M
    M0, M1 = q(0.0), q(spread); upd(M0); upd(M1); stack = [(0.0, M0, spread, M1)]; nodes = 0
    while stack:
        d1, Ma, d2, Mb = stack.pop(); nodes += 1
        if sorted(Ma[0]) == sorted(Mb[0]): continue
        h1, h2 = Ma[1] - 2 * d1 * Ma[2], Mb[1] - 2 * d2 * Mb[2]
        s = (h2 - h1) / (d2 - d1)                                     # chord slope
        dstar = min(max(-s / (2 * k), d1), d2)                         # argmin of k d^2 + h1 + s (d - d1)
        lb = k * dstar ** 2 + h1 + s * (dstar - d1)
        if lb >= val(inc) - tol * max(1.0, abs(val(inc))): continue
        dt = (Mb[1] - Ma[1]) / (2 * (Mb[2] - Ma[2])) if abs(Mb[2] - Ma[2]) > 1e-15 else 0.5 * (d1 + d2)
        dm = dt if d1 + 1e-12 < dt < d2 - 1e-12 else 0.5 * (d1 + d2)
        Mm = q(dm); upd(Mm)
        if dm - d1 < 1e-13 or d2 - dm < 1e-13: continue
        stack.append((d1, Ma, dm, Mm)); stack.append((dm, Mm, d2, Mb))
    return inc, val(inc), nodes


def exact_qubit(lam):
    """Exact C1 qubit split (k = n) and exact Cao-ratio optimum via Dinkelbach on top of solve_k (k = 1/c)."""
    n = len(lam) // 2; cache = {}
    best, v, nodes = solve_k(lam, n, cache)
    cost = 0.5 * v / (lam ** 2).sum()
    c = min(M[1] / M[2] ** 2 for M in cache.values() if M[2] > 0)
    for _ in range(50):                                             # Dinkelbach: find M with A - c B^2 < 0, else c optimal
        Mr, vr, _ = solve_k(lam, 1.0 / c, cache)
        if vr >= -1e-13 * max(1.0, Mr[1]): break
        c = Mr[1] / Mr[2] ** 2
    ratio = min((M for M in cache.values() if M[2] > 0), key=lambda M: M[1] / M[2] ** 2)
    return cost, (best[1], best[2], best[0]), (ratio[1], ratio[2], ratio[0]), len(set(tuple(sorted(M[0])) for M in cache.values())), len(cache)


def grid_from_pairs(pairs, lam_len):
    """2 x n rank grid: row 0 = lower member, row 1 = upper, columns sorted by pair mean (an assignment; gauge-irrelevant)."""
    pairs = sorted(pairs, key=lambda p: p[0] + p[1]); n = len(pairs)
    return np.array([[p[0] for p in pairs], [p[1] for p in pairs]])


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "E1a":
        Y = t1.syt_all(2, 8); worst = 0.0; rows = []
        import census as C
        for kind in ("GUE", "chain"):
            for k in range(100 if kind == "GUE" else 20):
                lam = np.sort(C.spectra(kind, 2, 8, k)); c_enum = t1.syt_min(lam, 2, 8, Y)[0]
                c_ex, *_ = exact_qubit(lam); diff = abs(c_ex - c_enum); worst = max(worst, diff); rows.append(dict(kind=kind, k=k, enum=c_enum, exact=c_ex))
        print(f"E1a: max |matching - enumeration| = {worst:.2e} (relative to ||lam||^2) over {len(rows)} spectra -> {'PASS' if worst <= 1e-12 else 'FAIL'}")
        json.dump(rows, open(os.path.join(t1.OUT, "E1a.json"), "w"))
    elif mode == "E1b":
        import impl_a as A, torch
        rows = []
        for ch in range(10):
            st = A.setup(1000 + ch, 2000 + ch, 6); lam = st["lam"].cpu().numpy().real; t0 = time.time()
            cost, best, ratio, nv, nq = exact_qubit(lam)
            grid = grid_from_pairs(best[2], len(lam)); E = lam[grid]; L, R = t1.fit(E)
            c_grid = (R ** 2).sum() / (lam ** 2).sum()
            # instrument check: impl_a cost1 at the assignment unitary with dA = 2, dB = 32
            U = st["V"][:, torch.as_tensor(grid.reshape(-1), device=A.DEV)]
            c_inst = A.cost1(U.conj().T @ st["H"] @ U, st["H_fro2"], 2, 32).item()
            iso = bool(np.array_equal(np.argsort(E.reshape(-1)), np.argsort(L.reshape(-1))))
            n = len(lam) // 2; delta = best[1] / n
            r = dict(chain=ch, exact_cost=cost, grid_cost=float(c_grid), instrument_cost=c_inst, order_isomorphic=iso,
                     qubit_splitting_delta=float(delta), mean_level_spacing=float((lam.max() - lam.min()) / (len(lam) - 1)),
                     ratio_opt_value=float(n * ratio[0] / ratio[1] ** 2 - 1), ratio_opt_delta=float(ratio[1] / n),
                     hull_vertices=nv, matchings_solved=nq, secs=round(time.time() - t0, 1))
            rows.append(r); print(json.dumps(r), flush=True)
        json.dump(rows, open(os.path.join(t1.OUT, "E1b_bb.json"), "w"), indent=0)
