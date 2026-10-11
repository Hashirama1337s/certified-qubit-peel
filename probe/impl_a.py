"""QMEREO implementation A (lead author). Normative text: CRITERIA.md (+ Amendment 1). torch, complex128, CUDA if present.
Modes:
  py impl_a.py EVAL <seed_H> <seed_S> <seed_U> <n>      -> "cost1 cost2 cost3 D(U,1)" (8 significant digits)
  py impl_a.py SELFTEST                                -> distance and derivative sanity checks (control I2 and more)
Library functions are used by probe.py (controls I3 and readouts R1-R3)."""
import sys, math, numpy as np, torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CT, RT = torch.complex128, torch.float64
I2 = np.eye(2, dtype=complex); PX = np.array([[0, 1], [1, 0]], complex)
PY = np.array([[0, -1j], [1j, 0]]); PZ = np.diag([1.0, -1.0]).astype(complex)


def kron_all(ops):
    out = np.array([[1.0 + 0j]])
    for o in ops:
        out = np.kron(out, o)
    return out


def site_op(op, i, n):
    return kron_all([op if k == i else I2 for k in range(n)])


def chain(seed, n):
    """Planted nearest-neighbour chain: bonds (Jx,Jy,Jz) in order, then fields (hx,hy,hz) per site."""
    rng = np.random.default_rng(seed); d = 2 ** n; H = np.zeros((d, d), complex)
    for i in range(n - 1):
        J = rng.standard_normal(3)
        for c, P in zip(J, (PX, PY, PZ)):
            H += c * (site_op(P, i, n) @ site_op(P, i + 1, n))
    for i in range(n):
        h = rng.standard_normal(3)
        for c, P in zip(h, (PX, PY, PZ)):
            H += c * site_op(P, i, n)
    return H


def haar(seed, d):
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    Q, R = np.linalg.qr(G); ph = np.diag(R) / np.abs(np.diag(R))
    return Q * ph[None, :]


def randK(seed, d):
    rng = np.random.default_rng(seed); G = rng.standard_normal((d, d)); return (G + G.T) / 2


def T(a):
    return torch.as_tensor(np.asarray(a), dtype=CT, device=DEV)


def eye(m):
    return torch.eye(m, dtype=CT, device=DEV)


def ptrace_B(rho, dA, dB):  # rho (..., d, d) -> (..., dA, dA)
    r = rho.reshape(*rho.shape[:-2], dA, dB, dA, dB)
    return r.diagonal(dim1=-3, dim2=-1).sum(-1)


def ptrace_A(rho, dA, dB):  # -> (..., dB, dB)
    r = rho.reshape(*rho.shape[:-2], dA, dB, dA, dB)
    return r.diagonal(dim1=-4, dim2=-2).sum(-1)


def h_parts(Hp, dA, dB):
    d = dA * dB; c = torch.trace(Hp) / d
    HA = ptrace_B(Hp, dA, dB) / dB; HB = ptrace_A(Hp, dA, dB) / dA
    Hint = Hp - torch.kron(HA, eye(dB)) - torch.kron(eye(dA), HB) + c * eye(d)
    return HA, HB, Hint


def expiK(K):  # K real symmetric (or Hermitian) tensor -> U = exp(iK)
    return torch.linalg.matrix_exp(1j * K.to(CT))


def herm(P):
    """Real d x d parameter matrix -> Hermitian K = sym(P) + i antisym(P) (covers all of u(d)); U = expiK(herm(P)) is unitary."""
    return (P + P.T) / 2 + 1j * (P - P.T) / 2


# ---------- C1 ----------
def cost1(Hp, H_fro2, dA, dB):
    return (torch.linalg.norm(h_parts(Hp, dA, dB)[2]) ** 2 / H_fro2).real


# ---------- C2 ----------
def ggm(m):
    """Orthonormal traceless Hermitian basis of M_m (Amendment A3 order): symmetric, antisymmetric, diagonal."""
    out = []
    for j in range(m):
        for k in range(j + 1, m):
            E = np.zeros((m, m), complex); E[j, k] = 1; out.append((E + E.T) / math.sqrt(2))
    for j in range(m):
        for k in range(j + 1, m):
            E = np.zeros((m, m), complex); E[j, k] = 1; out.append(-1j * (E - E.T) / math.sqrt(2))
    for l in range(1, m):
        D = np.zeros((m, m), complex)
        for j in range(l): D[j, j] = 1
        D[l, l] = -l; out.append(D / math.sqrt(l * (l + 1)))
    return T(np.stack(out))  # (m^2-1, m, m)


def cpo(Hint, LA, LB, dA, dB, starts=8):
    """Candidate pointer observable: minimise ||[Hint, OA (x) OB]||_F, ||OA||=||OB||=1, alternating exact minimisation."""
    nA, nB = LA.shape[0], LB.shape[0]
    IA, IB = eye(dA), eye(dB)
    LA_full = torch.stack([torch.kron(L, IB) for L in LA])  # (nA, d, d) basis for OA (x) 1
    LB_full = torch.stack([torch.kron(IA, L) for L in LB])
    def comm(Ops):
        return Hint @ Ops - Ops @ Hint
    def quad(Cs):  # Cs (k, d, d) -> real symmetric M[j,j'] = Re Tr(C_j^dag C_j')
        F = Cs.reshape(Cs.shape[0], -1)
        return (F.conj() @ F.T).real
    best = None
    for s in range(starts):
        rng = np.random.default_rng(s)
        a = torch.as_tensor(rng.standard_normal(nA), dtype=RT, device=DEV); a = a / a.norm()
        b = torch.as_tensor(rng.standard_normal(nB), dtype=RT, device=DEV); b = b / b.norm()
        f_prev = None
        for it in range(200):
            OB = torch.einsum("k,kij->ij", b.to(CT), LB_full)           # 1 (x) OB
            Cs = comm(LA_full @ OB)                                    # [Hint, L_j (x) OB] since L_j(x)1 . 1(x)OB
            w, V = torch.linalg.eigh(quad(Cs)); a = V[:, 0]
            OA = torch.einsum("k,kij->ij", a.to(CT), LA_full)
            Cs = comm(OA @ LB_full)
            w, V = torch.linalg.eigh(quad(Cs)); b = V[:, 0]; f = w[0].item()
            if f_prev is not None and abs(f_prev - f) <= 1e-12 * max(1.0, abs(f)):
                break
            f_prev = f
        if best is None or f < best[0]:
            best = (f, a.clone(), b.clone())
    f, a, b = best
    OA = torch.einsum("k,kij->ij", a.to(CT), LA); OB = torch.einsum("k,kij->ij", b.to(CT), LB)
    return f, OA, OB


def schwinger(Hp, OA, OB, dA, dB, H_op2):
    """Carroll-Singh Eqs. 27-30 with exact second derivatives at t = 0, averaged over the d_A peaked states."""
    wA, A = torch.linalg.eigh(OA)   # columns = eigenvectors |a_j>
    # CRITERIA Amendment 2: environment ready state = maximally mixed 1_B/d_B (phase average of the uniform superposition)
    projA = torch.einsum("aj,bj->jab", A, A.conj())                    # (dA, dA, dA): |a_j><a_j|
    rho = torch.stack([torch.kron(projA[j], eye(dB)) for j in range(dA)]) / dB   # (dA, d, d)
    def c(X): return Hp @ X - X @ Hp
    rho1 = -1j * c(rho); rho2 = -(c(c(rho)))
    rA, r1A, r2A = ptrace_B(rho, dA, dB), ptrace_B(rho1, dA, dB), ptrace_B(rho2, dA, dB)
    S_lin = -2 * (torch.einsum("jab,jba->j", r2A, rA) + torch.einsum("jab,jba->j", r1A, r1A)).real
    p = torch.einsum("ka,jab,bk->jk", A.conj().T, rA, A).real
    p1 = torch.einsum("ka,jab,bk->jk", A.conj().T, r1A, A).real
    p2 = torch.einsum("ka,jab,bk->jk", A.conj().T, r2A, A).real
    S_ptr = -2 * (p1 ** 2 + p * p2).sum(1)
    return (torch.maximum(S_lin, S_ptr).mean() / H_op2), S_lin, S_ptr


def cost2(Hp, dA, dB, H_op2, LA=None, LB=None):
    LA = ggm(dA) if LA is None else LA; LB = ggm(dB) if LB is None else LB
    _, _, Hint = h_parts(Hp, dA, dB)
    f, OA, OB = cpo(Hint, LA, LB, dA, dB)
    return schwinger(Hp, OA, OB, dA, dB, H_op2)[0]


# ---------- C3 ----------
def evolve_purity(Hp_eig, U, alpha, beta, dA, dB, K, dt_phys):
    """Purity of rho_A at t_k = k*dt_phys, k = 0..K, for psi0 = alpha (x) beta (normalised), H' = U^dag H U with H = V diag(lam) V^dag."""
    lam, V = Hp_eig
    a = alpha / alpha.norm(); b = beta / beta.norm()
    psi0 = torch.kron(a, b)
    t = torch.arange(K + 1, dtype=RT, device=DEV) * dt_phys
    phase = torch.exp(-1j * lam[None, :] * t[:, None])            # (K+1, d)
    v = V.conj().T @ (U @ psi0)                                   # V^dag U psi0
    psi_t = (U.conj().T @ (V @ (phase * v[None, :]).T)).T         # (K+1, d) rows: U^dag V e^{-i lam t} V^dag U psi0
    P = psi_t.reshape(K + 1, dA, dB)
    rA = torch.einsum("kab,kcb->kac", P, P.conj())
    return (rA.abs() ** 2).sum((1, 2)).real                       # Tr rho_A^2


def cost3_at(Hp_eig, U, alpha, beta, dA, dB, K, dt_phys):
    pur = evolve_purity(Hp_eig, U, alpha, beta, dA, dB, K, dt_phys)
    return (1.0 - pur).sum() / K                                  # (dt/T) sum_{k=0}^{K}, T = K dt


def cost3_inner(Hp_eig, U, dA, dB, K, dt_phys, starts=8, steps=200, lr=0.05):
    best = None
    for s in range(starts):
        rng = np.random.default_rng(s)
        a0 = rng.standard_normal(dA) + 1j * rng.standard_normal(dA)
        b0 = rng.standard_normal(dB) + 1j * rng.standard_normal(dB)
        a = torch.tensor(a0, dtype=CT, device=DEV, requires_grad=True)
        b = torch.tensor(b0, dtype=CT, device=DEV, requires_grad=True)
        opt = torch.optim.Adam([a, b], lr=lr)
        for _ in range(steps):
            opt.zero_grad(); L = cost3_at(Hp_eig, U, a, b, dA, dB, K, dt_phys); L.backward(); opt.step()
        with torch.no_grad():
            L = cost3_at(Hp_eig, U, a, b, dA, dB, K, dt_phys).item()
        if best is None or L < best:
            best = L
    return best


# ---------- distance ----------
def alg_basis(U, dA, dB, which="A"):
    d = dA * dB
    if which == "A":
        E = torch.zeros(dA * dA, dA, dA, dtype=CT, device=DEV); idx = torch.arange(dA * dA, device=DEV)
        E[idx, idx // dA, idx % dA] = 1
        full = torch.stack([torch.kron(e, eye(dB)) for e in E]) / math.sqrt(dB)
    else:
        E = torch.zeros(dB * dB, dB, dB, dtype=CT, device=DEV); idx = torch.arange(dB * dB, device=DEV)
        E[idx, idx // dB, idx % dB] = 1
        full = torch.stack([torch.kron(eye(dA), e) for e in E]) / math.sqrt(dA)
    Xs = U @ full @ U.conj().T
    return Xs.reshape(Xs.shape[0], d * d)


def tps_distance(U, V, dA, dB):
    for M in (U, V):
        err = (M.conj().T @ M - eye(M.shape[0])).abs().max().item()
        if err > 1e-8:
            raise ValueError(f"tps_distance: input is not unitary (|U^dag U - 1| = {err:.3g})")
    X = alg_basis(U, dA, dB, "A")
    def dist(Y, m):
        s = ((X.conj() @ Y.T).abs() ** 2).sum().real
        return math.sqrt(max(0.0, 2.0 * (1.0 - s.item() / m)))
    return min(dist(alg_basis(V, dA, dB, "A"), dA * dA), dist(alg_basis(V, dA, dB, "B"), dA * dA))


# ---------- setup ----------
def setup(seed_H, seed_S, n):
    d = 2 ** n; dA = dB = 2 ** (n // 2)
    Hp = chain(seed_H, n); S = haar(seed_S, d); H = S @ Hp @ S.conj().T
    Ht = T(H); lam, V = torch.linalg.eigh(Ht)
    return dict(H=Ht, S=T(S), lam=lam, V=V, dA=dA, dB=dB, d=d, H_fro2=(torch.linalg.norm(Ht) ** 2).real,
                H_op2=(lam.abs().max() ** 2), K=2 * d, dt=0.2 / lam.abs().max().item())


def eval_record(seed_H, seed_S, seed_U, n):
    st = setup(seed_H, seed_S, n); U = expiK(T(randK(seed_U, st["d"])))
    Hp = U.conj().T @ st["H"] @ U
    with torch.no_grad():
        c1 = cost1(Hp, st["H_fro2"], st["dA"], st["dB"]).item()
        c2 = cost2(Hp, st["dA"], st["dB"], st["H_op2"]).item()
    c3 = cost3_inner((st["lam"], st["V"]), U, st["dA"], st["dB"], st["K"], st["dt"])
    D = tps_distance(U, eye(st["d"]), st["dA"], st["dB"])
    return c1, c2, c3, D


def selftest():
    ok = True
    for n in (4, 6):
        d = 2 ** n; dA = dB = 2 ** (n // 2); I = eye(d)
        WA = T(haar(11, dA)); WB = T(haar(12, dB)); W = torch.kron(WA, WB)
        d0 = tps_distance(I, I, dA, dB); dl = tps_distance(W, I, dA, dB); dh = tps_distance(T(haar(13, d)), I, dA, dB)
        sw = tps_distance(T(np.eye(d)[np.arange(d).reshape(dA, dB).T.reshape(-1)]), I, dA, dB)  # swap A<->B
        print(f"n={n}: D(1,1)={d0:.2e} D(local,1)={dl:.2e} D(haar,1)={dh:.3f} D(swap,1)={sw:.2e}")
        ok &= d0 < 1e-6 and dl < 1e-6 and dh > 0.3 and sw < 1e-6  # tolerance per PROBE-BAR Amendment 1
        st = setup(1, 2, n); U = expiK(T(randK(3, d))); Hp = U.conj().T @ st["H"] @ U
        with torch.no_grad():
            _, _, Hint = h_parts(Hp, dA, dB)
            f, OA, OB = cpo(Hint, ggm(dA), ggm(dB), dA, dB)
            S, S_lin, S_ptr = schwinger(Hp, OA, OB, dA, dB, st["H_op2"])
        print(f"  CPO f={f:.4e} ||OA||={torch.linalg.norm(OA).item():.3f} S_lin>=0: {bool((S_lin > -1e-9).all())} Schwinger={S.item():.4e}")
        ok &= bool((S_lin > -1e-9).all())
        # H_int has vanishing partial traces
        ok &= ptrace_B(Hint, dA, dB).abs().max().item() < 1e-9 and ptrace_A(Hint, dA, dB).abs().max().item() < 1e-9
    print("SELFTEST", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    if sys.argv[1] == "EVAL":
        c1, c2, c3, D = eval_record(*map(int, sys.argv[2:6]))
        print(f"{c1:.8g} {c2:.8g} {c3:.8g} {D:.8g}")
    elif sys.argv[1] == "SELFTEST":
        selftest()
