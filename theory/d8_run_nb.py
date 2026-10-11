"""BAR-D8 Amendment 2 runner: THEIRS-NB arm (notebook-faithful port), one (d, seed) per process.
Reuses the exact peel of results/d8/run_d{d}_s{seed}.json (same input spectrum). Writes results/d8/run_nb_d{d}_s{seed}.json/.log.
Port deviation (logged): their return_best_split raises IndexError when ind_min + scan_width >= N_steps; the port uses the last q."""
import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d8_exact as D, d8_theirs as T, qubit_exact as QE

d, seed = int(sys.argv[1]), int(sys.argv[2]); OUT = os.path.join(QE.t1.OUT, "d8")
JP = os.path.join(OUT, f"run_nb_d{d}_s{seed}.json"); LOG = open(os.path.join(OUT, f"run_nb_d{d}_s{seed}.log"), "a", buffering=1)
def say(*a): print(time.strftime("%H:%M:%S"), *a, file=LOG, flush=True)
base = json.load(open(os.path.join(OUT, f"run_d{d}_s{seed}.json")))
NS, RF = 200, 10
R = dict(d=d, seed=seed, bar="BAR-D8 Amendment 2", grid=dict(N_steps=NS, refining_factor=RF), G0=[], nb=[], exact_on_nb=[])
def save(): json.dump(R, open(JP, "w"), default=float)
lam = np.sort(D.gue_spectrum(d, seed))
# ---- G0: gradient fidelity of their returned gradient vs finite differences of their cost
rng = np.random.default_rng(seed)
for k in range(5):
    q = 0.3 + 0.5 * rng.random(); x = 0.5 * (lam[::2] + lam[1::2])[1:] + 1e-3 * rng.standard_normal(d // 2 - 1)
    g_theirs = T._nb_grad(x, q, lam); h = 1e-6; gfd = np.zeros_like(x)
    for i in rng.choice(len(x), 40, replace=False):
        e = np.zeros_like(x); e[i] = h; gfd[i] = (T._nb_cost(x + e, q, lam) - T._nb_cost(x - e, q, lam)) / (2 * h)
    idx = np.nonzero(gfd)[0]; rel = np.linalg.norm(g_theirs[idx] - gfd[idx]) / max(1e-300, np.linalg.norm(gfd[idx]))
    true = (T._nb_grad(x, q, lam)); fix = np.linalg.norm((g_theirs[idx] - (np.concatenate([[0], g_theirs])[0])) - gfd[idx])
    R["G0"].append(dict(q=q, rel_mismatch=float(rel)))
save(); say("G0 rel mismatch of their gradient vs finite differences:", [round(r["rel_mismatch"], 4) for r in R["G0"]])
# ---- NB peel with exact split on the same per-level input
spec = lam.copy(); lvl = 0
while len(spec) > 2:
    t0 = time.time(); nb = T.best_split_nb(spec, N_steps=NS, refining_factor=RF); t1 = time.time()
    if lvl == 0:   # identical input to the base run's level 0: reuse its certified exact split (no recomputation)
        b0 = base['exact_on_theirs'][0]; ex = dict(E_q=b0['exact_E_q'], ratio=b0['exact_ratio'])
    else:
        ex, pairs, rest, nc = D.exact_cao_split(spec)
    row = dict(level=lvl, dim=len(spec), nb_E_q=nb["E_q"], nb_ratio=nb["ratio_cao"], exact_E_q=ex["E_q"], exact_ratio=ex["ratio"],
               rel_gap=(nb["ratio_cao"] - ex["ratio"]) / ex["ratio"], rel_Eq=abs(nb["E_q"] - ex["E_q"]) / ex["E_q"], wrapped=nb["wrapped"],
               nb_secs=round(t1 - t0, 1), exact_secs=round(time.time() - t1, 1), T2_pass=bool(ex["ratio"] <= nb["ratio_cao"] + 1e-9))
    R["nb"].append(dict(level=lvl, E_q=nb["E_q"], ratio=nb["ratio_cao"])); R["exact_on_nb"].append(row); save(); say("NB level", json.dumps(row))
    if lvl == 0: grid1 = nb["grid"]
    spec = nb["rest"]; lvl += 1
R["nb"].append(dict(level=lvl, E_q=float(spec[1] - spec[0]), final=True))
Enb = [r["E_q"] for r in R["nb"]]; Eex = [r["E_q"] for r in base["exact"]]; Eth = [r["E_q"] for r in base["theirs"]]
def frac(Eqs):
    model = np.array([0.0])
    for E in Eqs: model = np.concatenate([model - E / 2, model + E / 2])
    model = np.sort(model) + lam.mean(); return float(((lam - model) ** 2).sum() / (lam ** 2).sum())
R["T5"] = dict(sqrt_f_nb=frac(Enb) ** 0.5, sqrt_f_exact=frac(Eex) ** 0.5, sqrt_f_R=frac(Eth) ** 0.5)
a, b = np.sort(Enb), np.sort(Eex); R["T6"] = dict(max_rel_diff=float(np.max(np.abs(a - b) / b)), sorted_nb=a.tolist(), sorted_exact=b.tolist())
r0 = R["exact_on_nb"][0]; th0 = base["exact_on_theirs"][0]
R["NB_vs_R"] = dict(gap_nb=r0["rel_gap"], gap_R=th0["rel_gap"], closer="NB" if r0["rel_gap"] < th0["rel_gap"] else "R")
save(); say("T5", json.dumps(R["T5"])); say("T6", json.dumps(R["T6"])); say("NB_vs_R", json.dumps(R["NB_vs_R"]))
# ---- T3s-NB: their port's fits around the exact optimum q*
qstar = base["exact_on_theirs"][0]["exact_E_q"] / 2; rstar = base["exact_on_theirs"][0]["exact_ratio"]; best = np.inf
for k, u in enumerate(np.linspace(-0.02, 0.02, 401)):
    q = qstar * (1 + u); y0 = np.zeros(2 + d // 2); y0[0] = -q; y0[1] = q
    np.random.seed(200000 + k); dummy = lam.copy(); np.random.shuffle(dummy); y0[2:] = 0.5 * (dummy[::2] + dummy[1::2])
    EA, EB, Eint, nit = T._nb_partition(lam, y0); r = np.sum(Eint ** 2) / (2 * q * q); best = min(best, r)
reading = "grid effect" if best <= rstar * (1 + 1e-3) else ("optimiser effect" if best > rstar * (1 + 1e-2) else "mixed")
R["T3s_nb"] = dict(qstar=qstar, exact_ratio=rstar, best_nb_near=float(best), rel=float((best - rstar) / rstar), reading=reading)
save(); say("T3s-NB", json.dumps(R["T3s_nb"])); say("DONE")
