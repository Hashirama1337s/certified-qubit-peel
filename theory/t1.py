"""BAR-T1 tests (sealed bar: theory/BAR-T1.md). Run from probe/:  py ../theory/t1.py <T12|T3|T4|T5|T6|COUNT> [args]"""
import sys, os, json, time, math, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "probe"))
import impl_a as A
from torch.autograd.functional import hessian
from r5 import gauge_basis

DEV, CT, RT = A.DEV, A.CT, A.RT
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"); os.makedirs(OUT, exist_ok=True)


# ---------- combinatorics ----------
def syt_all(m, n):
    """All standard Young tableaux of rectangular shape m x n as rank grids (cell -> rank of eigenvalue, 0 = smallest)."""
    out = []; T = -np.ones((m, n), int); rowlen = [0] * m
    def rec(k):
        if k == m * n: out.append(T.copy()); return
        for a in range(m):
            b = rowlen[a]
            if b < n and (a == 0 or rowlen[a - 1] > b):
                T[a, b] = k; rowlen[a] += 1; rec(k + 1); rowlen[a] -= 1; T[a, b] = -1
    rec(0); return np.array(out)


def syt_random(m, n, rng):
    """Uniform random SYT of shape m x n (Greene-Nijenhuis-Wilf hook walk)."""
    rowlen = [n] * m; T = np.zeros((m, n), int)
    for k in range(m * n - 1, -1, -1):
        cells = [(a, b) for a in range(m) for b in range(rowlen[a])]
        a, b = cells[rng.integers(len(cells))]
        while True:
            arm = rowlen[a] - b - 1; leg = sum(1 for a2 in range(a + 1, m) if rowlen[a2] > b)
            if arm + leg == 0: break
            h = rng.integers(arm + leg)
            if h < arm: b = b + 1 + h
            else: a = a + 1 + (h - arm)
        T[a, b] = k; rowlen[a] -= 1
    return T


def fit(E):
    """Two-way additive fit of grids E (..., m, n): L = mu + alpha_a + beta_b (= P_loc of diag(E)), R = E - L."""
    mu = E.mean((-1, -2), keepdims=True); L = E.mean(-1, keepdims=True) + E.mean(-2, keepdims=True) - mu
    return L, E - L


def xmin(E, L):
    """min over anti-diagonal cell pairs (a<a', b>b') of (L_i-L_j)(E_i-E_j); E, L (..., m, n)."""
    m, n = E.shape[-2:]; idx = [(a, b, a2, b2) for a in range(m) for a2 in range(a + 1, m) for b in range(n) for b2 in range(b)]
    if not idx: return np.full(E.shape[:-2], np.inf)
    a, b, a2, b2 = map(np.array, zip(*idx))
    v = (L[..., a, b] - L[..., a2, b2]) * (E[..., a, b] - E[..., a2, b2])
    return v.min(-1)


def rc_ok(E):
    """(R) and (C): every two rows / columns entrywise ordered."""
    m, n = E.shape
    for a in range(m):
        for a2 in range(a + 1, m):
            D = E[a] - E[a2]
            if not (np.all(D > 0) or np.all(D < 0)): return False
    for b in range(n):
        for b2 in range(b + 1, n):
            D = E[:, b] - E[:, b2]
            if not (np.all(D > 0) or np.all(D < 0)): return False
    return True


def predict(E, N):
    """D3: gauge-projected Hessian spectrum of cost1 at the assignment with grid E."""
    m, n = E.shape; L, _ = fit(E); ev = [0.0] * ((m - 1) * (n - 1))
    for a in range(m):
        for a2 in range(a + 1, m):
            for b in range(n):
                for b2 in range(n):
                    if b2 != b:
                        v = 2 * (L[a, b] - L[a2, b2]) * (E[a, b] - E[a2, b2]) / N; ev += [v, v]
    def block(D):
        k = len(D); M = D.mean() * np.diag(D) - np.outer(D, D) / k
        Qc = np.linalg.qr(np.c_[np.ones(k), np.eye(k)[:, :k - 1]])[0][:, 1:]
        return np.linalg.eigvalsh(Qc.T @ M @ Qc)
    for a in range(m):
        for a2 in range(a + 1, m):
            for v in 2 * block(E[a] - E[a2]) / N: ev += [v, v]
    for b in range(n):
        for b2 in range(b + 1, n):
            for v in 2 * block(E[:, b] - E[:, b2]) / N: ev += [v, v]
    return np.sort(np.array(ev))


# ---------- numerics on the real instrument (impl_a) ----------
def U_of(st, grid):
    """Assignment unitary: product cell g = a*dB + b  ->  eigenvector of rank grid[a, b]."""
    return st["V"][:, torch.as_tensor(grid.reshape(-1), device=DEV)]


def autograd_at(st, U0):
    d, dA, dB = st["d"], st["dA"], st["dB"]; H, N = st["H"], st["H_fro2"]
    def f(P):
        U = U0 @ A.expiK(A.herm(P)); return A.cost1(U.conj().T @ H @ U, N, dA, dB)
    P0 = torch.zeros(d, d, dtype=RT, device=DEV, requires_grad=True); val = f(P0)
    g = torch.autograd.grad(val, P0)[0].reshape(-1)
    Hs = hessian(f, torch.zeros(d, d, dtype=RT, device=DEV), vectorize=True).reshape(d * d, d * d)
    Q = gauge_basis(dA, dB); Qf, _ = torch.linalg.qr(Q, mode="complete"); C = Qf[:, Q.shape[1]:]
    g_ng = (g - Q @ (Q.T @ g)).norm().item()
    w = torch.linalg.eigvalsh(C.T @ ((Hs + Hs.T) / 2) @ C).cpu().numpy()
    return val.item(), g_ng, np.sort(w)


def chain_st(ch, n):
    return A.setup(1000 + ch, 2000 + ch, n)


def grid_E(st, grid):
    return st["lam"].cpu().numpy().real[grid]


def T12(n, chains, per):
    """T1 gradient and T2 Hessian-spectrum match at random-perm and random-SYT assignments."""
    dA = dB = 2 ** (n // 2); rows = []; rng = np.random.default_rng(99)
    for ch in range(chains):
        st = chain_st(ch, n); N = st["H_fro2"].item()
        for kind in ("perm", "syt"):
            for r in range(per):
                grid = rng.permutation(dA * dB).reshape(dA, dB) if kind == "perm" else syt_random(dA, dB, rng)
                E = grid_E(st, grid); t0 = time.time()
                val, g_ng, w = autograd_at(st, U_of(st, grid)); p = predict(E, N)
                diff = np.abs(w - p).max(); scale = np.abs(p).max()
                _, R = fit(E); c_pred = (R ** 2).sum() / N
                rec = dict(n=n, chain=ch, kind=kind, grad=g_ng, val=val, val_pred=c_pred, hess_maxdiff=diff, hess_scale=scale,
                           rel=diff / scale, n_neg=int((w < -1e-10).sum()), n_neg_pred=int((p < -1e-10).sum()),
                           n_zero_pred=int((np.abs(p) <= 1e-10).sum()), D4=bool(rc_ok(E) and xmin(E, fit(E)[0]) > 0),
                           secs=round(time.time() - t0, 1))
                rows.append(rec); print(json.dumps(rec), flush=True)
    ok1 = all(r["grad"] < 1e-9 for r in rows); ok2 = all(r["rel"] < 1e-8 for r in rows)
    print(f"T1 n={n}: max grad {max(r['grad'] for r in rows):.2e} -> {'PASS' if ok1 else 'FAIL'};  "
          f"T2: max rel Hessian diff {max(r['rel'] for r in rows):.2e} -> {'PASS' if ok2 else 'FAIL'}")
    json.dump(rows, open(f"{OUT}/T12_n{n}.json", "w"), indent=0)


def T3(chains, per_kind):
    """T3 (n=4): numerically strict local min (autograd, no negative eig) <=> D4."""
    n = 4; dA = dB = 4; rng = np.random.default_rng(7); agree = tot = 0; nmin = 0; rows = []
    for ch in range(chains):
        st = chain_st(ch, n)
        for kind in ("perm", "syt"):
            for r in range(per_kind):
                grid = rng.permutation(16).reshape(4, 4) if kind == "perm" else syt_random(4, 4, rng)
                E = grid_E(st, grid); _, _, w = autograd_at(st, U_of(st, grid))
                nz = np.sort(np.abs(w))[(dA - 1) * (dB - 1):]          # drop the 9 torus zeros
                is_min = bool((w > -1e-10).all() and nz.min() > 1e-10)
                d4 = bool(rc_ok(E) and xmin(E, fit(E)[0]) > 0)
                agree += (is_min == d4); tot += 1; nmin += is_min
                rows.append(dict(chain=ch, kind=kind, is_min=is_min, D4=d4))
        print(f"chain {ch}: running agreement {agree}/{tot}, local minima so far {nmin}", flush=True)
    print(f"T3: autograd-local-min <=> D4 in {agree}/{tot} -> {'PASS' if agree == tot else 'FAIL'}  (local minima found: {nmin})")
    json.dump(rows, open(f"{OUT}/T3.json", "w"))


def syt_min(lam, m, n, Y=None):
    """Exact global min of cost1 (D2) = min over SYT of the additive residual; also the number of strict local minima (D4)."""
    Y = syt_all(m, n) if Y is None else Y
    E = lam[Y]; L, R = fit(E); c = (R ** 2).sum((-1, -2)) / (lam ** 2).sum()
    xm = xmin(E, L); is_min = xm > 0
    if m == n: is_min_cls = is_min.sum() // 2             # transpose pairs are the same TPS up to A<->B swap
    else: is_min_cls = is_min.sum()
    k = int(np.argmin(c)); return c[k], Y[k], int(is_min_cls), c, is_min


def T4():
    Y = syt_all(4, 4); print("SYT 4x4:", len(Y))
    worst = None; lines = []
    for path, mk in (("results/run_n4_c1_I3.jsonl", lambda ch: A.setup(1000 + ch, 2000 + ch, 4)),):
        for l in open(path):
            r = json.loads(l); st = mk(r["chain"]); lam = st["lam"].cpu().numpy().real
            cmin, grid, nloc, _, _ = syt_min(lam, 4, 4, Y)
            # verify cmin with the instrument
            c_inst = A.cost1(U_of(st, grid).conj().T @ st["H"] @ U_of(st, grid), st["H_fro2"], 4, 4).item()
            gap = min(r["all_costs"]) - cmin
            lines.append(dict(src=path, chain=r["chain"], syt_min=cmin, syt_min_instrument=c_inst, optimiser_best=min(r["all_costs"]),
                              lattice=r["cost_at_plant"], n_local_minima=nloc, gap=gap))
            print(json.dumps(lines[-1]), flush=True)
            worst = gap if worst is None else min(worst, gap)
    print(f"T4: min over chains of (optimiser best - SYT min) = {worst:.3e} -> {'PASS' if worst > -1e-9 else 'FAIL (D2 killed)'}")
    json.dump(lines, open(f"{OUT}/T4.json", "w"), indent=0)


def T5():
    import probe as PR
    Y = syt_all(4, 4); rows = []
    for ch, eps in ((0, 0.05), (0, 1e-3), (1, 1e-3), (2, 1e-3)):
        st = PR.setup_decoupled(7000 + ch, 4, eps); lam = st["lam"].cpu().numpy().real
        cmin, grid, nloc, _, _ = syt_min(lam, 4, 4, Y)
        Hp = st["S"].conj().T @ st["H"] @ st["S"]; c_plant = A.cost1(Hp, st["H_fro2"], 4, 4).item()
        U = U_of(st, grid); D_raw = A.tps_distance(U, st["S"], 4, 4)
        rows.append(dict(chain=ch, eps=eps, syt_min=cmin, plant=c_plant, beats_or_equal=bool(cmin <= c_plant + 1e-15),
                         D_raw_phases=D_raw, n_local_minima=nloc))
        print(json.dumps(rows[-1]), flush=True)
    ok = all(r["beats_or_equal"] or r["D_raw_phases"] < 0.05 for r in rows)
    print(f"T5: {'PASS' if ok else 'FAIL'}")
    json.dump(rows, open(f"{OUT}/T5.json", "w"), indent=0)


def T6():
    rng = np.random.default_rng(5); rows = []
    for ch, n in ((0, 4), (1, 4), (0, 6)):
        st = chain_st(ch, n); dA = dB = st["dA"]
        grid = syt_random(dA, dB, rng) if ch == 0 else rng.permutation(dA * dB).reshape(dA, dB)
        U = U_of(st, grid); Hp = U.conj().T @ st["H"] @ U
        with torch.no_grad():
            c2 = A.cost2(Hp, dA, dB, st["H_op2"]).item()
            a = torch.zeros(dA, dtype=CT, device=DEV); a[1] = 1; b = torch.zeros(dB, dtype=CT, device=DEV); b[2] = 1
            c3 = A.cost3_at((st["lam"], st["V"]), U, a, b, dA, dB, st["K"], st["dt"]).item()
        rows.append(dict(n=n, chain=ch, cost2=c2, cost3_basis_state=c3)); print(json.dumps(rows[-1]), flush=True)
    ok = all(abs(r["cost2"]) < 1e-10 and abs(r["cost3_basis_state"]) < 1e-10 for r in rows)
    print(f"T6: {'PASS' if ok else 'FAIL'}"); json.dump(rows, open(f"{OUT}/T6.json", "w"), indent=0)


if __name__ == "__main__":
    m = sys.argv[1]
    if m == "T12": T12(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    elif m == "T3": T3(int(sys.argv[2]), int(sys.argv[3]))
    elif m == "T4": T4()
    elif m == "T5": T5()
    elif m == "T6": T6()
