"""BAR-D8 Amendment 1: T3s (spike-local BFGS around the exact optimum q*) and T3w (exact spike width), level 1.
Usage (from probe/): py ../theory/d8_spike.py d seed   -> results/d8/spike_d{d}_s{seed}.json"""
import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d8_exact as D, d8_theirs as T, qubit_exact as QE

d, seed = int(sys.argv[1]), int(sys.argv[2]); OUT = os.path.join(QE.t1.OUT, "d8")
run = json.load(open(os.path.join(OUT, f"run_d{d}_s{seed}.json")))
lam = np.sort(D.gue_spectrum(d, seed)); n = d // 2
ex = run["exact_on_theirs"][0]; qstar = ex["exact_E_q"] / 2; rstar = ex["exact_ratio"]
t0 = time.time(); u = np.linspace(-0.02, 0.02, 401); best = np.inf; fits = []
for k, uu in enumerate(u):
    q = qstar * (1 + uu); B, loss, nit = T.fit_B(q, lam, 100000 + k); r = loss / (2 * q * q)   # Cao-scale ratio: loss / (2 q^2)
    fits.append((q, loss, r, nit)); best = min(best, r)
t_bfgs = time.time() - t0
t0 = time.time(); w = np.linspace(-0.02, 0.02, 41); ex_r = []
for uu in w:
    q = qstar * (1 + uu); p, A, Bs = D.warm_matching(lam, 2 * q); V = A - 4 * q * Bs + n * 4 * q * q; ex_r.append((q, (V / 2) / (2 * q * q)))
t_ex = time.time() - t0
qs = np.array([a for a, b in ex_r]); rs = np.array([b for a, b in ex_r]); thr = 2 * rstar
inside = rs <= thr
# width by linear interpolation of the crossing points on each side of q*
i0 = int(np.argmin(np.abs(qs - qstar)))
def cross(step):
    i = i0
    while 0 <= i + step < len(qs) and rs[i + step] <= thr: i += step
    if not (0 <= i + step < len(qs)): return qs[i]
    a, b = rs[i], rs[i + step]; f = (thr - a) / (b - a) if b != a else 0.0
    return qs[i] + f * (qs[i + step] - qs[i])
width = float(cross(+1) - cross(-1))
EA_max = 1.5 * np.mean(lam[d // 2:]); step = EA_max / 5000
reading = "grid effect" if best <= rstar * (1 + 1e-3) else ("optimiser effect" if best > rstar * (1 + 1e-2) else "mixed")
res = dict(d=d, seed=seed, qstar=qstar, exact_ratio=rstar, best_bfgs_near=best, rel=(best - rstar) / rstar, T3s=reading,
           width=width, their_coarse_step=step, width_over_step=width / step, edge_hit=bool(inside[0] or inside[-1]),
           secs_bfgs=round(t_bfgs, 1), secs_exact=round(t_ex, 1), exact_curve=ex_r, bfgs_fits=fits)
json.dump(res, open(os.path.join(OUT, f"spike_d{d}_s{seed}.json"), "w"), default=float)
print(json.dumps({k: v for k, v in res.items() if k not in ("exact_curve", "bfgs_fits")}, default=float))
