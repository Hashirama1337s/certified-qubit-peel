"""BAR-D8 Amendment 3: staged d = 2048 runner (EXACT + THEIRS-NB). Partial JSON after every stage; progress in the log.
Usage (from probe/): py ../theory/d8_run2048.py seed  -> results/d8/r2048_s{seed}.json / .log"""
import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d8_exact as D, d8_theirs as T, qubit_exact as QE

seed = int(sys.argv[1]); d = 2048; OUT = os.path.join(QE.t1.OUT, "d8")
JP = os.path.join(OUT, f"r2048_s{seed}.json"); LOG = open(os.path.join(OUT, f"r2048_s{seed}.log"), "a", buffering=1)
def say(*a): print(time.strftime("%H:%M:%S"), *a, file=LOG, flush=True)
R = dict(d=d, seed=seed, bar="BAR-D8 Amendment 3", grid=dict(N_steps=200, refining_factor=10), stages={})
def save(): json.dump(R, open(JP, "w"), default=float)
lam = np.sort(D.gue_spectrum(d, seed)); say("start", "spread", lam[-1] - lam[0])

# progress inside the exact split: count certified oracle calls
_orig = D.WarmMatcher.solve
def _traced(self, delta, max_rounds=400):
    r = _orig(self, delta, max_rounds)
    if len(D.STATS) % 20 == 0: say(f"  exact oracle calls {len(D.STATS)} (last N={self.N} delta={delta:.4f} {D.STATS[-1]['secs']}s {D.STATS[-1]['route'][:3]})")
    return r
D.WarmMatcher.solve = _traced

# S1 exact level-1 split
t0 = time.time(); n0 = len(D.STATS); ex1, pairs1, rest1, _ = D.exact_cao_split(lam); S = D.STATS[n0:]
R["stages"]["S1"] = dict(E_q=ex1["E_q"], ratio=ex1["ratio"], secs=round(time.time() - t0, 1), oracle_calls=len(S), max_gap=max(s["gap"] for s in S),
                         routes=sorted(set(s["route"][:3] for s in S)))
save(); say("S1 EXACT level 1", json.dumps(R["stages"]["S1"]))

# S2 THEIRS-NB level-1 sift
t0 = time.time(); nb1 = T.best_split_nb(lam, N_steps=200, refining_factor=10, progress=lambda m: say(" ", m))
gap = (nb1["ratio_cao"] - ex1["ratio"]) / ex1["ratio"]; relEq = abs(nb1["E_q"] - ex1["E_q"]) / ex1["E_q"]
R["stages"]["S2"] = dict(nb_E_q=nb1["E_q"], nb_ratio=nb1["ratio_cao"], gap=gap, relEq=relEq, same_spike=bool(relEq < 0.01), ind_min=nb1["ind_min"],
                         wrapped=nb1["wrapped"], upper_clamped=nb1["upper_clamped"], T2_pass=bool(ex1["ratio"] <= nb1["ratio_cao"] + 1e-9), secs=round(time.time() - t0, 1))
save(); say("S2 NB level 1", json.dumps(R["stages"]["S2"]))

# S3 T3s-NB around the exact optimum
t0 = time.time(); qstar = ex1["E_q"] / 2; rstar = ex1["ratio"]; best = np.inf
for k, u in enumerate(np.linspace(-0.02, 0.02, 401)):
    q = qstar * (1 + u); y0 = np.zeros(2 + d // 2); y0[0] = -q; y0[1] = q
    np.random.seed(200000 + k); dummy = lam.copy(); np.random.shuffle(dummy); y0[2:] = 0.5 * (dummy[::2] + dummy[1::2])
    EA, EB, Eint, nit = T._nb_partition(lam, y0); best = min(best, np.sum(Eint ** 2) / (2 * q * q))
    if (k + 1) % 50 == 0: say(f"  T3s fits {k+1}/401 best so far {best:.6g} (exact {rstar:.6g})")
rel = (best - rstar) / rstar
R["stages"]["S3"] = dict(qstar=qstar, exact_ratio=rstar, best_nb_near=float(best), rel=float(rel),
                         reading=("grid effect" if best <= rstar * (1 + 1e-3) else ("optimiser effect" if best > rstar * (1 + 1e-2) else "mixed")),
                         secs=round(time.time() - t0, 1))
save(); say("S3 T3s-NB", json.dumps(R["stages"]["S3"]))

# S4 NB peel (levels >= 2) with exact splits on the NB inputs
R["stages"]["S4"] = [dict(level=0, dim=d, nb_E_q=nb1["E_q"], nb_ratio=nb1["ratio_cao"], exact_E_q=ex1["E_q"], exact_ratio=ex1["ratio"], gap=gap, relEq=relEq)]
spec = nb1["rest"]; lvl = 1
while len(spec) > 2:
    t0 = time.time(); nb = T.best_split_nb(spec, N_steps=200, refining_factor=10); ex, _, _, _ = D.exact_cao_split(spec)
    R["stages"]["S4"].append(dict(level=lvl, dim=len(spec), nb_E_q=nb["E_q"], nb_ratio=nb["ratio_cao"], exact_E_q=ex["E_q"], exact_ratio=ex["ratio"],
                                  gap=(nb["ratio_cao"] - ex["ratio"]) / ex["ratio"], relEq=abs(nb["E_q"] - ex["E_q"]) / ex["E_q"],
                                  upper_clamped=nb["upper_clamped"], wrapped=nb["wrapped"], secs=round(time.time() - t0, 1)))
    save(); say("S4 NB level", json.dumps(R["stages"]["S4"][-1])); spec = nb["rest"]; lvl += 1
Enb = [r["nb_E_q"] for r in R["stages"]["S4"]] + [float(spec[1] - spec[0])]

# S5 exact peel (levels >= 2)
Eex = [ex1["E_q"]]; spec = rest1
while len(spec) > 2:
    t0 = time.time(); ex, _, rest, _ = D.exact_cao_split(spec); Eex.append(ex["E_q"]); say("S5 EXACT level", len(spec), f"E_q {ex['E_q']:.6f} ratio {ex['ratio']:.6g} {time.time()-t0:.1f}s"); spec = rest
Eex.append(float(spec[1] - spec[0]))
def frac(Eqs):
    m = np.array([0.0])
    for E in Eqs: m = np.concatenate([m - E / 2, m + E / 2])
    m = np.sort(m) + lam.mean(); return float(((lam - m) ** 2).sum() / (lam ** 2).sum())
a, b = np.sort(Enb), np.sort(Eex)
R["stages"]["S5"] = dict(E_q_nb=Enb, E_q_exact=Eex, sqrt_f_nb=frac(Enb) ** 0.5, sqrt_f_exact=frac(Eex) ** 0.5, T6=float(np.max(np.abs(a - b) / b)))
save(); say("S5", json.dumps({k: v for k, v in R["stages"]["S5"].items() if not k.startswith("E_q")})); say("DONE")
