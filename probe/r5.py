"""R5: is the lattice cut a critical point / local minimum of each criterion? Exact autograd gradient and Hessian at U = S,
gauge directions projected out. Bar: PROBE-BAR.md Amendment 5.  Usage: py r5.py <n> <chains> [crits=123]"""
import sys, json, math, time, numpy as np, torch
import impl_a as A
from torch.autograd.functional import hessian

DEV, CT, RT = A.DEV, A.CT, A.RT


def gauge_basis(dA, dB):
    """Orthonormal basis (as flattened real P-vectors) of gauge directions K = W_A (x) 1 + 1 (x) W_B, W Hermitian."""
    d = dA * dB; vecs = []
    def add(K):
        Pm = K.real + K.imag                      # herm(P) = K  (Re K symmetric, Im K antisymmetric)
        vecs.append(Pm.reshape(-1))
    LA, LB = A.ggm(dA), A.ggm(dB)                 # traceless; add the identity on A (global phase on B is the same direction)
    for L in LA: add(torch.kron(L, A.eye(dB)))
    for L in LB: add(torch.kron(A.eye(dA), L))
    add(A.eye(d))
    G = torch.stack(vecs).to(RT)                  # (dA^2 + dB^2 - 1, d^2)
    Q, _ = torch.linalg.qr(G.T)                   # orthonormal columns spanning the gauge subspace
    return Q


def crit_fn(st, crit, extras=None):
    """Returns f(P) (and for C3 f(P, a, b)) with U = S expm(i herm(P)); extras holds fixed CPO for C2."""
    S, H, dA, dB = st["S"], st["H"], st["dA"], st["dB"]
    if crit == 1:
        def f(P):
            U = S @ A.expiK(A.herm(P)); return A.cost1(U.conj().T @ H @ U, st["H_fro2"], dA, dB)
        return f
    if crit == 2:
        OA, OB = extras
        def f(P):
            U = S @ A.expiK(A.herm(P)); return A.schwinger(U.conj().T @ H @ U, OA, OB, dA, dB, st["H_op2"])[0]
        return f
    def f(P, a, b):
        U = S @ A.expiK(A.herm(P)); return A.cost3_at((st["lam"], st["V"]), U, a, b, dA, dB, st["K"], st["dt"])
    return f


def inner_c3(st):
    """(alpha, beta) at their optimum for U = S (8 starts, as in CRITERIA A4)."""
    S, dA, dB = st["S"], st["dA"], st["dB"]; best = None
    for s in range(8):
        rng = np.random.default_rng(s)
        a = torch.tensor(rng.standard_normal(dA) + 1j * rng.standard_normal(dA), dtype=CT, device=DEV, requires_grad=True)
        b = torch.tensor(rng.standard_normal(dB) + 1j * rng.standard_normal(dB), dtype=CT, device=DEV, requires_grad=True)
        opt = torch.optim.Adam([a, b], lr=0.05)
        for _ in range(400):
            opt.zero_grad(); L = A.cost3_at((st["lam"], st["V"]), S, a, b, dA, dB, st["K"], st["dt"]); L.backward(); opt.step()
        L = A.cost3_at((st["lam"], st["V"]), S, a, b, dA, dB, st["K"], st["dt"]).item()
        if best is None or L < best[0]: best = (L, a.detach() / a.detach().norm(), b.detach() / b.detach().norm())
    return best


def lanczos_extremes(fun, x0, Q, k=6, maxiter=400):
    """Smallest and largest eigenvalues of the gauge-projected Hessian of fun at x0 via Hessian-vector products (Lanczos).
    Returns (smallest k, largest k, number of negatives among the smallest k)."""
    from scipy.sparse.linalg import LinearOperator, eigsh
    m = x0.numel(); Pproj = lambda v: v - Q @ (Q.T @ v)
    def hvp(v_np):
        v = Pproj(torch.tensor(v_np, dtype=RT, device=DEV))
        x = x0.clone().requires_grad_(True); y = fun(x); g = torch.autograd.grad(y, x, create_graph=True)[0].reshape(-1)
        Hv = torch.autograd.grad(g @ v, x)[0].reshape(-1)
        return Pproj(Hv).detach().cpu().numpy()
    op = LinearOperator((m, m), matvec=hvp, dtype=np.float64)
    lo = eigsh(op, k=k, which="SA", maxiter=maxiter, tol=1e-6, return_eigenvectors=False)
    hi = eigsh(op, k=k, which="LA", maxiter=maxiter, tol=1e-6, return_eigenvectors=False)
    lo, hi = np.sort(lo), np.sort(hi)[::-1]
    return lo, hi, int((lo < -1e-9).sum())


def analyse_big(st, crit):
    """n >= 8: gradient exact; Hessian extremes by Lanczos (C2 on the min-Schwinger CPO branch, fixed; C3 with (alpha, beta)
    fixed at their inner optimum, stated as a proxy)."""
    d, dA, dB = st["d"], st["dA"], st["dB"]; Q = gauge_basis(dA, dB)
    P0 = torch.zeros(d, d, dtype=RT, device=DEV); note = "exact"
    if crit == 1:
        f = crit_fn(st, 1)
    elif crit == 2:
        LA, LB = A.ggm(dA), A.ggm(dB)
        with torch.no_grad():
            Hp = st["S"].conj().T @ st["H"] @ st["S"]; _, _, Hint = A.h_parts(Hp, dA, dB)
            best = None
            for s0 in range(16):
                _, OA_s, OB_s = A.cpo(Hint, LA, LB, dA, dB, starts=1) if s0 == 0 else (None, None, None)
                break
            _, OA, OB = A.cpo(Hint, LA, LB, dA, dB, starts=16)   # min commutator-norm CPO over 16 starts (branch choice stated)
        f = crit_fn(st, 2, (OA, OB)); note = "fixed CPO (16-start commutator min; min-Schwinger tie-break NOT applied at n>=8)"
    else:
        L0, a0, b0 = inner_c3(st); f3 = crit_fn(st, 3); f = lambda P: f3(P, a0, b0); note = "(alpha, beta) fixed at inner optimum (proxy Hessian)"
    val = f(P0).item(); Pv = P0.clone().requires_grad_(True); y = f(Pv); g = torch.autograd.grad(y, Pv)[0].reshape(-1)
    g_ng = (g - Q @ (Q.T @ g)).norm().item(); rel_g = g_ng / max(abs(val), 1e-300)
    lo, hi, nneg = lanczos_extremes(f, P0, Q)
    cls = "NOT CRITICAL" if rel_g >= 1e-6 else ("SADDLE" if nneg > 0 else "STRICT LOCAL MIN (among Lanczos extremes)")
    return dict(value=val, grad_norm_nongauge=g_ng, rel_grad=rel_g, n_neg=nneg, n_pos=None, min_eig=float(lo[0]), max_eig=float(hi[0]),
                n_eigs=None, cls=cls, hessian_note="Lanczos extremes: " + note, smallest_eigs=[float(x) for x in lo], largest_eigs=[float(x) for x in hi])


def analyse(st, crit):
    d, dA, dB = st["d"], st["dA"], st["dB"]; Q = gauge_basis(dA, dB)
    P0 = torch.zeros(d, d, dtype=RT, device=DEV)
    if crit == 2:
        # Enumerate CPO branches at the cut (degenerate), pick the min-Schwinger branch (stated tie-break), then the smooth
        # fixed-CPO gradient/Hessian on that branch. Also the "limit from outside": mean f(+-h v) at h = 1e-3, 16 directions.
        LA, LB = A.ggm(dA), A.ggm(dB)
        with torch.no_grad():
            Hp = st["S"].conj().T @ st["H"] @ st["S"]; _, _, Hint = A.h_parts(Hp, dA, dB)
            branches = []
            for s0 in range(64):
                fcomm, OA_s, OB_s = A.cpo(Hint, LA, LB, dA, dB, starts=1) if False else (None, None, None)
            # cpo() takes starts=n and loops seeds 0..n-1; to enumerate branches call it per seed via a thin wrapper
            def cpo_seed(seed):
                import impl_a as _A
                nA, nB = LA.shape[0], LB.shape[0]; IA, IB = _A.eye(dA), _A.eye(dB)
                LA_full = torch.stack([torch.kron(L, IB) for L in LA]); LB_full = torch.stack([torch.kron(IA, L) for L in LB])
                comm = lambda Ops: Hint @ Ops - Ops @ Hint
                quad = lambda Cs: ((Cs.reshape(Cs.shape[0], -1)).conj() @ (Cs.reshape(Cs.shape[0], -1)).T).real
                rng = np.random.default_rng(seed)
                a = torch.as_tensor(rng.standard_normal(nA), dtype=RT, device=DEV); a = a / a.norm()
                b = torch.as_tensor(rng.standard_normal(nB), dtype=RT, device=DEV); b = b / b.norm(); f_prev = None
                for it in range(200):
                    OB_ = torch.einsum("k,kij->ij", b.to(CT), LB_full); w, V = torch.linalg.eigh(quad(comm(LA_full @ OB_))); a = V[:, 0]
                    OA_ = torch.einsum("k,kij->ij", a.to(CT), LA_full); w, V = torch.linalg.eigh(quad(comm(OA_ @ LB_full))); b = V[:, 0]; f = w[0].item()
                    if f_prev is not None and abs(f_prev - f) <= 1e-12 * max(1.0, abs(f)): break
                    f_prev = f
                return f, torch.einsum("k,kij->ij", a.to(CT), LA), torch.einsum("k,kij->ij", b.to(CT), LB)
            for s0 in range(64):
                fcomm, OA_s, OB_s = cpo_seed(s0)
                Ssch = A.schwinger(Hp, OA_s, OB_s, dA, dB, st["H_op2"])[0].item()
                branches.append((Ssch, fcomm, OA_s, OB_s))
            branches.sort(key=lambda x: x[0]); fmin_comm = min(b[1] for b in branches)
            tied = [b for b in branches if b[1] <= fmin_comm * (1 + 1e-6) + 1e-12]     # branches that are genuine CPO minimisers
            vals = np.array([b[0] for b in tied]); n_distinct = int(len(np.unique(np.round(vals, 6))))
            Sbest, _, OA, OB = tied[0]
        f = crit_fn(st, 2, (OA, OB)); val = f(P0).item()
        Pv = P0.clone().requires_grad_(True); y = f(Pv); g_fixed = torch.autograd.grad(y, Pv)[0].reshape(-1)
        Hs = hessian(lambda P: f(P), P0).reshape(d * d, d * d)
        def full_c2(P):
            with torch.no_grad():
                U = st["S"] @ A.expiK(A.herm(P)); return A.cost2(U.conj().T @ st["H"] @ U, dA, dB, st["H_op2"], LA, LB).item()
        Pproj0 = torch.eye(d * d, dtype=RT, device=DEV) - Q @ Q.T; rng = np.random.default_rng(5); h = 1e-3; outside = []
        for _ in range(16):
            v = Pproj0 @ torch.tensor(rng.standard_normal(d * d), dtype=RT, device=DEV); v = v / v.norm()
            outside += [full_c2((h * v).reshape(d, d)), full_c2((-h * v).reshape(d, d))]
        g = g_fixed
        extra = dict(cpo_branches_tied=len(tied), schwinger_branch_min=float(vals.min()), schwinger_branch_max=float(vals.max()),
                     n_distinct_branch_values=n_distinct, limit_from_outside_mean=float(np.mean(outside)), limit_from_outside_min=float(np.min(outside)),
                     grad_norm_fixedcpo=float((Pproj0 @ g_fixed).norm()))
    elif crit == 1:
        f = crit_fn(st, 1); val = f(P0).item()
        Pv = P0.clone().requires_grad_(True); y = f(Pv); g = torch.autograd.grad(y, Pv)[0].reshape(-1)
        Hs = hessian(lambda P: f(P), P0).reshape(d * d, d * d)
    else:
        L0, a0, b0 = inner_c3(st); f = crit_fn(st, 3); val = L0
        # joint Hessian over (P, Re a, Im a, Re b, Im b); Schur complement onto P
        def joint(x):
            P = x[: d * d].reshape(d, d); ra, ia = x[d * d: d * d + dA], x[d * d + dA: d * d + 2 * dA]
            rb, ib = x[d * d + 2 * dA: d * d + 2 * dA + dB], x[d * d + 2 * dA + dB:]
            return f(P, torch.complex(ra, ia), torch.complex(rb, ib))
        x0 = torch.cat([P0.reshape(-1), a0.real, a0.imag, b0.real, b0.imag]).to(RT)
        xv = x0.clone().requires_grad_(True); y = joint(xv); gj = torch.autograd.grad(y, xv)[0]
        g = gj[: d * d]; g_inner = gj[d * d:]
        Hj = hessian(joint, x0); m = d * d
        Hpp, Hpq, Hqq = Hj[:m, :m], Hj[:m, m:], Hj[m:, m:]
        # project out the global-phase/scale null directions of (a, b) via pseudo-inverse
        Hs = Hpp - Hpq @ torch.linalg.pinv(Hqq, rtol=1e-8) @ Hpq.T
        val = L0; extra_grad = g_inner.norm().item()
    # gauge projection
    Pproj = torch.eye(d * d, dtype=RT, device=DEV) - Q @ Q.T
    g_ng = (Pproj @ g).norm().item(); H_ng = Pproj @ Hs @ Pproj
    w = torch.linalg.eigvalsh((H_ng + H_ng.T) / 2).cpu().numpy()
    # drop the gauge null space (exactly dim(Q) zero eigenvalues)
    k = Q.shape[1]; w_sorted = np.sort(np.abs(w)); thresh = w_sorted[k - 1] if k <= len(w) else 0.0
    w_ng = np.array([x for x in w if abs(x) > max(thresh, 1e-12)]) if k > 0 else w
    rel_g = g_ng / max(abs(val), 1e-300)          # for C2: the min-Schwinger CPO branch (smooth), as stated
    neg = int((w_ng < -1e-9).sum()); critical = rel_g < 1e-6
    cls = "NOT CRITICAL" if not critical else ("SADDLE" if neg > 0 else "STRICT LOCAL MIN")
    out = dict(value=val, grad_norm_nongauge=g_ng, rel_grad=rel_g, n_neg=neg, n_pos=int((w_ng > 1e-9).sum()),
                min_eig=float(w_ng.min()) if len(w_ng) else None, max_eig=float(w_ng.max()) if len(w_ng) else None,
                n_eigs=int(len(w_ng)), cls=cls, hessian_note=("min-Schwinger CPO branch, fixed" if crit == 2 else "exact"))
    if crit == 2: out.update(extra)
    return out


if __name__ == "__main__":
    n, chains = int(sys.argv[1]), int(sys.argv[2]); crits = [int(c) for c in (sys.argv[3] if len(sys.argv) > 3 else "123")]
    start = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    out = open(f"results/r5_n{n}.jsonl", "a")
    for ch in range(start, start + chains):
        st = A.setup(1000 + ch, 2000 + ch, n)
        for crit in crits:
            t0 = time.time(); r = (analyse_big if n >= 8 else analyse)(st, crit); r.update(chain=ch, n=n, crit=crit, secs=round(time.time() - t0, 1))
            out.write(json.dumps(r) + "\n"); out.flush()
            print(f"n={n} chain {ch} C{crit}: {r['cls']}  value={r['value']:.4g} rel_grad={r['rel_grad']:.2e} "
                  f"neg={r['n_neg']} pos={r['n_pos']}/{r['n_eigs']} min_eig={r['min_eig']:.3g} max_eig={r['max_eig']:.3g} {r['secs']}s", flush=True)
