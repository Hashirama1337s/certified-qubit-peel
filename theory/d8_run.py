"""BAR-D8 production runner (sealed bar theory/BAR-D8.md). One (d, seed) per process; results/d8/run_d{d}_s{seed}.json, log .log.
Arms: THEIRS-R peel (d8_theirs, reduced grid 200 + 201), EXACT split on each THEIRS-R level input (T2), EXACT peel (T5/T6),
level-1 per-q check at 41 THEIRS-R grid points (T1). Writes partial JSON after every stage."""
import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d8_exact as D, d8_theirs as T, qubit_exact as QE

d, seed = int(sys.argv[1]), int(sys.argv[2])
OUT = os.path.join(QE.t1.OUT, "d8"); os.makedirs(OUT, exist_ok=True)
JP = os.path.join(OUT, f"run_d{d}_s{seed}.json"); LOG = open(os.path.join(OUT, f"run_d{d}_s{seed}.log"), "a", buffering=1)
def say(*a): print(time.strftime("%H:%M:%S"), *a, file=LOG, flush=True)
R = dict(d=d, seed=seed, bar="BAR-D8", grid=dict(n_coarse=200, scan_width=10, refine=10), theirs=[], exact_on_theirs=[], exact=[], t1=[])
def save(): json.dump(R, open(JP, "w"), default=float)
BESSEL = 4 * np.pi / np.array([3.8317059702, 7.0155866698, 10.1734681351, 13.3236919363, 16.4706300509])

def factorization_fraction(lam, Eqs):
    """Model spectrum of the peeled qubits: all sums of +-E_q/2 (+ mean); f = ||lam - model||^2 / ||lam||^2 on sorted levels."""
    model = np.array([0.0])
    for E in Eqs: model = np.concatenate([model - E / 2, model + E / 2])
    model = np.sort(model) + lam.mean(); return float(((np.sort(lam) - model) ** 2).sum() / (lam ** 2).sum())

lam = D.gue_spectrum(d, seed); say("start", d, seed, "spread", lam[-1] - lam[0])
# ---- THEIRS-R peel with per-level exact comparison on the same input (T2, T3, T4)
spec = lam.copy(); lvl = 0; grid1 = None
while len(spec) > 2:
    t0 = time.time(); th = T.best_split(spec, n_coarse=200, refine=10, scan_width=10)
    if lvl == 0: grid1 = th["grid"]
    t1 = time.time(); n0 = len(D.STATS); ex, pairs, rest, nc = D.exact_cao_split(spec); S = D.STATS[n0:]
    row = dict(level=lvl, dim=len(spec), theirs_E_q=th["E_q"], theirs_ratio=th["ratio_cao"], exact_E_q=ex["E_q"], exact_ratio=ex["ratio"],
               rel_gap=(th["ratio_cao"] - ex["ratio"]) / ex["ratio"], rel_Eq=abs(th["E_q"] - ex["E_q"]) / ex["E_q"],
               exact_spike_dist=float(np.min(np.abs(ex["E_q"] - BESSEL))), theirs_secs=round(t1 - t0, 1), exact_secs=round(time.time() - t1, 1),
               oracle_calls=len(S), max_gap=max(s["gap"] for s in S), T2_pass=bool(ex["ratio"] <= th["ratio_cao"] + 1e-9))
    R["theirs"].append(dict(level=lvl, E_q=th["E_q"], ratio=th["ratio_cao"])); R["exact_on_theirs"].append(row); save(); say("THEIRS-R level", json.dumps(row))
    spec = th["rest"]; lvl += 1
R["theirs"].append(dict(level=lvl, E_q=float(spec[1] - spec[0]), final=True))
# ---- EXACT peel (T5, T6)
spec = lam.copy(); lvl = 0
while len(spec) > 2:
    t0 = time.time(); n0 = len(D.STATS); ex, pairs, rest, nc = D.exact_cao_split(spec); S = D.STATS[n0:]
    R["exact"].append(dict(level=lvl, dim=len(spec), E_q=ex["E_q"], ratio=ex["ratio"], secs=round(time.time() - t0, 1), oracle_calls=len(S),
                           max_gap=max(s["gap"] for s in S), routes=sorted(set(s["route"][:3] for s in S))))
    save(); say("EXACT level", json.dumps(R["exact"][-1])); spec = rest; lvl += 1
R["exact"].append(dict(level=lvl, E_q=float(spec[1] - spec[0]), final=True))
Eth = [r["E_q"] for r in R["theirs"]]; Eex = [r["E_q"] for r in R["exact"]]
R["T5"] = dict(f_theirs=factorization_fraction(lam, Eth), f_exact=factorization_fraction(lam, Eex))
R["T5"].update(sqrt_f_theirs=R["T5"]["f_theirs"] ** 0.5, sqrt_f_exact=R["T5"]["f_exact"] ** 0.5)
a, b = np.sort(Eth), np.sort(Eex); R["T6"] = dict(max_rel_diff=float(np.max(np.abs(a - b) / b)), sorted_theirs=a.tolist(), sorted_exact=b.tolist())
save(); say("T5", json.dumps(R["T5"])); say("T6", json.dumps(R["T6"]))
# ---- T1: per-q optimality at level 1 on 41 THEIRS-R grid points
g = grid1; coarse = g[:200]; fine = g[200:]
qbest = fine[np.argmin(fine[:, 2]), 0]; fsel = fine[np.argsort(np.abs(fine[:, 0] - qbest))[:21]]
pts = np.vstack([coarse[4::5], fsel])
for q, loss_scaled, ratio, nit in pts:
    p, A, B = D.warm_matching(np.sort(lam), 2 * q); n = d // 2
    Vq = A - 4 * q * B + n * 4 * q * q; loss_exact = Vq / 2; loss_bfgs = loss_scaled * d
    R["t1"].append(dict(q=q, loss_exact=loss_exact, loss_bfgs=loss_bfgs, rel=(loss_bfgs - loss_exact) / max(1.0, loss_exact),
                        T1_pass=bool(loss_exact <= loss_bfgs + 1e-9 * max(1.0, loss_bfgs))))
save()
v = [r for r in R["t1"]]; say("T1", f"{sum(r['T1_pass'] for r in v)}/{len(v)} pass; BFGS above exact by >1e-6 rel at {sum(r['rel'] > 1e-6 for r in v)} points; max rel {max(r['rel'] for r in v):.3e}")
say("DONE")
