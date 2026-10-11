"""BAR-D8 Amendment 4, EXACT arm: certified exact peel on spectra saved by the notebook run (d8_nbrun.py FULL).
Usage (from probe/):
  py ../theory/d8_exact_on.py PEEL   exact peel of results/d8/nbrun/level1_input_d2048.npy -> results/d8/nbrun/exact_peel.json
  py ../theory/d8_exact_on.py ONNB   exact split on each saved NB level input (level 1 reused from PEEL) -> exact_onnb.json"""
import sys, os, json, time, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d8_exact as D, qubit_exact as QE

mode = sys.argv[1]; OUT = os.path.join(QE.t1.OUT, "d8", "nbrun")
LOG = open(os.path.join(OUT, f"exact_{mode}.log"), "a", buffering=1)
def say(*a): print(time.strftime("%Y-%m-%d %H:%M:%S"), *a, file=LOG, flush=True)
lam = np.load(os.path.join(OUT, "level1_input_d2048.npy"))

def frac(Eqs):   # Amendment 3 definition (their return_H_prod construction)
    m = np.array([0.0])
    for E in Eqs: m = np.concatenate([m - E / 2, m + E / 2])
    m = np.sort(m) + lam.mean(); return float(((lam - m) ** 2).sum() / (lam ** 2).sum())

if mode == "PEEL":
    JP = os.path.join(OUT, "exact_peel.json"); R = dict(input="level1_input_d2048.npy", levels=[])
    spec = lam.copy(); say("PEEL start d", len(spec))
    while len(spec) > 2:
        t0 = time.time(); n0 = len(D.STATS); ex, pairs, rest, _ = D.exact_cao_split(spec); S = D.STATS[n0:]
        R["levels"].append(dict(d=len(spec), E_q=ex["E_q"], ratio=ex["ratio"], secs=round(time.time() - t0, 1), oracle_calls=len(S),
                                max_gap=max(s["gap"] for s in S), routes=sorted(set(s["route"][:3] for s in S))))
        json.dump(R, open(JP, "w"), indent=1, default=float); say("level", json.dumps(R["levels"][-1])); spec = rest
    Eex = [r["E_q"] for r in R["levels"]] + [float(spec[1] - spec[0])]
    R.update(E_q_exact=Eex, sqrt_f_exact=frac(Eex) ** 0.5); json.dump(R, open(JP, "w"), indent=1, default=float)
    say("PEEL DONE sqrt_f_exact", R["sqrt_f_exact"])
elif mode == "ONNB":
    nb = json.load(open(os.path.join(OUT, "nbrun_FULL.json"))); pe = json.load(open(os.path.join(OUT, "exact_peel.json")))
    rows = []
    for L in nb["levels"]:
        if "crashed" in L: rows.append(dict(level=L["level"], d=L["d"], crashed=L["crashed"])); continue
        if L["level"] == 1: ex = dict(E_q=pe["levels"][0]["E_q"], ratio=pe["levels"][0]["ratio"]); secs = 0.0
        else:
            spec = np.load(os.path.join(OUT, f"level{L['level']}_input_d{L['d']}.npy")); t0 = time.time()
            ex, _, _, _ = D.exact_cao_split(spec); secs = round(time.time() - t0, 1)
        rows.append(dict(level=L["level"], d=L["d"], nb_E_q=L["E_q"], nb_ratio=L["ratio_cao"], exact_E_q=ex["E_q"], exact_ratio=ex["ratio"],
                         gap=(L["ratio_cao"] - ex["ratio"]) / ex["ratio"], relEq=abs(L["E_q"] - ex["E_q"]) / ex["E_q"],
                         T2_pass=bool(ex["ratio"] <= L["ratio_cao"] + 1e-9), exact_secs=secs))
        json.dump(dict(rows=rows), open(os.path.join(OUT, "exact_onnb.json"), "w"), indent=1, default=float); say("ONNB", json.dumps(rows[-1]))
    say("ONNB DONE")
