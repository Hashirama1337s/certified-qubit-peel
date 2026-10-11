"""Door 8: re-implementation (from theory/D8-SPEC.md, sections 2-4) of the numerical one-qubit split of arXiv 2606.10004.
Not their code: written from the specification of their notebook (quantum_mereology, code_for_Wave-packets-from-the-spectrum).
Per q: A frozen at (-q, +q); minimise over B in R^{d/2} the sorted-level loss ||sort(A_i + B_j) - evals||^2 by scipy BFGS
(analytic gradient, tol 1e-10, gtol 1e-16, maxiter 1000), starting from pair means of a seeded random pairing.
Selection: argmin over the q grid of loss / sum(A^2) = loss / (2 q^2). Grid: coarse q = EA_max s / n_coarse, s = 1..n_coarse,
EA_max = 1.5 mean(upper half of evals); fine = linspace over +-50 coarse steps around the coarse winner with
2*50*refine + 1 points. Peel: next spectrum = sorted fitted B. Grid sizes are parameters (theirs: n_coarse 5000, refine 100)."""
import numpy as np
from scipy.optimize import minimize


def sorted_loss_grad(B, q, evals):
    d = len(evals); m = np.concatenate([B - q, B + q]); p = np.argsort(m, kind="stable")
    r = m[p] - evals; loss = float(r @ r)
    g_full = np.empty(d); g_full[p] = 2 * r
    gB = g_full[: d // 2] + g_full[d // 2:]
    return loss, gB


def fit_B(q, evals, seed):
    rng = np.random.RandomState(seed); dummy = evals.copy(); rng.shuffle(dummy)
    B0 = 0.5 * (dummy[::2] + dummy[1::2])
    res = minimize(sorted_loss_grad, B0, args=(q, evals), jac=True, method="BFGS", tol=1e-10, options=dict(gtol=1e-16, maxiter=1000))
    loss, _ = sorted_loss_grad(res.x, q, evals)
    return res.x, loss, res.nit


def best_split(evals, n_coarse=5000, refine=100, scan_width=50, log=None):
    evals = np.sort(evals); d = len(evals); EA_max = 1.5 * np.mean(evals[d // 2:])
    rows = []
    def run(q, seed):
        B, loss, nit = fit_B(q, evals, seed); ratio = (loss / d) / (2 * q * q)
        rows.append((q, loss / d, ratio, nit)); return B, ratio
    best = None
    for s in range(1, n_coarse + 1):
        q = EA_max * s / n_coarse; B, r = run(q, s)
        if best is None or r < best[0]: best = (r, q, B, s)
    dq = EA_max / n_coarse; qw = best[1]
    fine = np.linspace(max(dq * 1e-3, qw - scan_width * dq), qw + scan_width * dq, 2 * scan_width * refine + 1)
    for k, q in enumerate(fine):
        B, r = run(q, k + 1 + n_coarse)
        if r < best[0]: best = (r, q, B, -k)
    r, q, B, _ = best
    return dict(E_q=2 * q, q=q, ratio_scaled=r, ratio_cao=r * d, rest=np.sort(B), grid=np.array(rows))


# ---------------------------------------------------------------------------------------------------------------------------
# THEIRS-NB: line-for-line port of the paper's notebook as executed (code_for_Wave-packets-from-the-spectrum.ipynb):
# partition(evals, 2, y0=y0, optimizeA=False) -> PartitionerB; return_best_split; return_dofs. Read-only port; their code was
# not executed. Kept on purpose: sum-zero chart B[0] = -sum(x); gradient returns arr[1:] (no chain-rule term -arr[0]);
# initial guess drops the first pair mean; global np.random.seed(s) + np.random.shuffle; fine-only argmin;
# scan_width = N_steps // refining_factor; python list indexing of q_values (negative indices wrap).

def _nb_cost(x, q, spec):
    y = np.concatenate([[-np.sum(x)], x]); DB = len(y)
    AB = np.sort(np.repeat(np.array([-q, q]), DB) + np.tile(y, 2))
    return float(np.sum((AB - spec) ** 2))


def _nb_grad(x, q, spec):
    y = np.concatenate([[-np.sum(x)], x]); DB = len(y); A = np.array([-q, q])
    G1 = np.full(DB, 2) * 2 * y
    G2 = 2 * np.full(DB, np.sum(A))
    AB = np.repeat(A, DB) + np.tile(y, 2)
    p = np.argsort(AB); inv = np.empty_like(p); inv[p] = np.arange(len(p))
    S = np.reshape(spec[inv], (2, DB))
    G3 = -2 * S.sum(axis=0)
    return (G1 + G2 + G3)[1:]


def _nb_partition(evals, y0):
    spec = np.sort(evals); q = y0[1]
    x0 = y0[2:][1:]                                                    # initial_guess = get_x_from_y(y0[DA:])
    res = minimize(_nb_cost, x0, args=(q, spec), jac=_nb_grad, method="BFGS", tol=1e-10, options=dict(gtol=1e-16, maxiter=1000))
    B = np.concatenate([[-np.sum(res.x)], res.x]); EA = np.array([-q, q])
    full = np.concatenate([EA[0] + B, EA[1] + B])
    Eint = evals - np.sort(full)
    return EA, B, Eint, res.nit


def best_split_nb(evals, N_steps=5000, refining_factor=100, progress=None):
    d = len(evals); d_half = d // 2
    EA_max = 1.5 * np.mean(evals[d_half:]); q_values = []
    loss_c = np.zeros(N_steps); EA_c = np.zeros((N_steps, 2)); EB_c = np.zeros((N_steps, d_half)); rows = []
    for s in range(1, N_steps + 1):
        y0 = np.zeros(2 + d_half); q = EA_max * s / N_steps; q_values.append(q); y0[0] = -q; y0[1] = q
        np.random.seed(s); dummy = evals.copy(); np.random.shuffle(dummy); y0[2:] = 0.5 * (dummy[::2] + dummy[1::2])
        EA, EB, Eint, nit = _nb_partition(evals, y0); loss = np.sum(Eint ** 2) / d
        loss_c[s - 1] = loss; EA_c[s - 1] = EA; EB_c[s - 1] = EB; rows.append((q, loss, loss / np.sum(EA ** 2), nit, 0))
        if progress and s % 25 == 0: progress(f'NB coarse {s}/{N_steps}')
    ind_min = int(np.argmin(loss_c / np.sum(EA_c ** 2, axis=1)))
    scan_width = N_steps // refining_factor
    q_min = q_values[ind_min - scan_width]; q_max = q_values[ind_min + scan_width] if ind_min + scan_width < N_steps else q_values[-1]
    steps = 2 * scan_width * refining_factor + 1; qf = np.linspace(q_min, q_max, steps)
    loss_f = np.zeros(steps); EA_f = np.zeros((steps, 2)); EB_f = np.zeros((steps, d_half))
    for s in range(steps):
        y0 = np.zeros(2 + d_half); q = qf[s]; y0[0] = -q; y0[1] = q
        np.random.seed(s + 1 + N_steps); dummy = evals.copy(); np.random.shuffle(dummy); y0[2:] = 0.5 * (dummy[::2] + dummy[1::2])
        EA, EB, Eint, nit = _nb_partition(evals, y0); loss = np.sum(Eint ** 2) / d
        loss_f[s] = loss; EA_f[s] = EA; EB_f[s] = EB; rows.append((q, loss, loss / np.sum(EA ** 2), nit, 1))
        if progress and s % 50 == 0: progress(f'NB fine {s}/{steps}')
    k = int(np.argmin(loss_f / np.sum(EA_f ** 2, axis=1)))
    q = EA_f[k][1]; r_scaled = loss_f[k] / np.sum(EA_f[k] ** 2)
    return dict(E_q=2 * q, q=q, ratio_scaled=r_scaled, ratio_cao=r_scaled * d, rest=np.sort(EB_f[k]), grid=np.array(rows),
                wrapped=bool(ind_min - scan_width < 0), upper_clamped=bool(ind_min + scan_width >= N_steps), ind_min=ind_min)
