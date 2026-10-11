"""Recompute the headline numbers of NOTE.md from the shipped result files (no solver runs; seconds).
Each line prints the value found and PASS/FAIL against the value stated in the note.
Usage (from probe/): python ../theory/d8_verify_note.py"""
import json, os, glob, numpy as np
from scipy.special import jn_zeros

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
NB = os.path.join(R, "d8", "nbrun")
fails = 0

def check(label, value, lo, hi, fmt="{:.6g}"):
    global fails
    ok = lo <= value <= hi; fails += (not ok)
    print(f"{'PASS' if ok else 'FAIL'}  {label}: {fmt.format(value)}  (note: [{lo:g}, {hi:g}])")

nb = json.load(open(os.path.join(NB, "nbrun_FULL.json")))
pe = json.load(open(os.path.join(NB, "exact_peel.json")))
on = json.load(open(os.path.join(NB, "exact_onnb.json")))["rows"]
lam = np.load(os.path.join(NB, "level1_input_d2048.npy"))

# Abstract / 3.1: first split of their notebook on their matrix
L1 = nb["levels"][0]; E1 = pe["levels"][0]
check("their first-split ratio", L1["ratio_cao"], 0.0124765, 0.0124775)
check("certified first-split ratio", E1["ratio"], 0.00620255, 0.00620265)
check("first-split gap (%)", 100 * (L1["ratio_cao"] / E1["ratio"] - 1), 100.5, 101.5)
check("their first-split E_q", L1["E_q"], 0.61785, 0.61795)
check("certified first-split E_q", E1["E_q"], 0.31955, 0.31965)
check("fits per split", L1["fits"], 15001, 15001, "{:d}")
cells = json.load(open(os.path.join(R, "d9", "npy_seed3_d2048.json")))["runs"][0]["rows"]
own = [c for c in cells if c["lo"] <= L1["E_q"] <= c["hi"]][0]          # certified best inside the peak cell their split sits in
check(f"their split vs certified best on its own peak (cell {own['cell']}) (%)", 100 * (L1["ratio_cao"] / own["ratio"] - 1), 17.5, 18.5)

# 3.1: all ten splits
gaps = {r["d"]: 100 * r["gap"] for r in on}
for d, (lo, hi) in {2048: (100.5, 101.5), 1024: (236, 237), 512: (147.5, 148.5), 256: (114.5, 115.5)}.items():
    check(f"gap at d = {d} (%)", gaps[d], lo, hi)
check("max |gap| for d <= 128 (%)", max(abs(gaps[d]) for d in gaps if d <= 128), 0, 0.01)
for d in (1024, 512):
    r = [x for x in on if x["d"] == d][0]
    check(f"E_q agreement at d = {d} (relative)", r["relEq"], 0, 0.01)
    check(f"ratio multiple at d = {d}", r["nb_ratio"] / r["exact_ratio"], 2.45, 3.4)

# 3.3: fine-window wrap at the full grid (N_steps 5000, refining_factor 100 -> scan_width 50)
N, sw, idx = 5000, 50, []
for L in range(1, 11):
    f = glob.glob(os.path.join(NB, f"level{L}_fits_d*.jsonl"))[0]
    obj = [json.loads(x)["obj"] for x in open(f)][:N]; idx.append(int(np.argmin(obj)))
print("      coarse winner (0-based) per split:", idx)
check("min coarse index (wrap needs < 50)", min(idx), 50, N)
check("max coarse index + 50 (IndexError needs >= 5000)", max(idx) + sw, 0, N - 1, "{:d}")

# 3.5: leftover interaction of the product construction
def sqrt_f(Eqs):
    m = np.array([0.0])
    for E in Eqs: m = np.concatenate([m - E / 2, m + E / 2])
    m = np.sort(m) + lam.mean(); return float(np.sqrt(((lam - m) ** 2).sum() / (lam ** 2).sum()))
EB = np.sort(np.load(os.path.join(NB, "level10_EB_d4.npy")).real.ravel())
check("their sqrt f (%)", 100 * sqrt_f([L["E_q"] for L in nb["levels"]] + [float(EB[-1] - EB[0])]), 2.87, 2.885)
check("exact per-step sqrt f (%)", 100 * pe["sqrt_f_exact"], 2.89, 2.905)

# 3.2: reduced-grid port on other spectra
s512 = json.load(open(os.path.join(R, "d8", "summary_nb_d512.json")))
g512 = sorted(100 * r["gap"] for r in s512["rows"])
print("      d = 512 port gaps (sorted, %):", [round(g) for g in g512])
check("d = 512 port: smallest gap (%)", g512[0], 34.5, 35.5); check("d = 512 port: largest gap (%)", g512[-1], 110.5, 111.5)
G0 = s512["summary"]["G0_range"]
check("gradient vs finite differences: smallest mismatch (%)", 100 * G0[0], 84.5, 85.5); check("largest mismatch (%)", 100 * G0[1], 107.5, 108.5)
s2k = json.load(open(os.path.join(R, "d8", "summary_2048.json")))
print("      d = 2048 port gaps (%):", [round(100 * r["gap"]) for r in s2k["rows"]], "| best-of-401 on peak (%):", [round(100 * r["T3s"]) for r in s2k["rows"]])

# 3.4: certified optima vs 2 pi / x_i
peaks = 2 * np.pi / jn_zeros(1, 20)
Es = [json.load(open(p))["exact_on_nb"][0]["exact_E_q"] for p in sorted(glob.glob(os.path.join(R, "d8", "run_nb_d512_s*.json")))]
Es += [r["ex_Eq"] for r in s2k["rows"]] + [E1["E_q"]]
offs = [float(np.min(np.abs((E - peaks) / peaks))) for E in Es]
check("max offset of certified first-split optima from 2 pi/x_i (%)", 100 * max(offs), 0, 1)
check("their matrix: offset from 2 pi/x_6 (%)", 100 * abs(E1["E_q"] - peaks[5]) / peaks[5], 0.2, 0.25)

print(f"\n{'ALL PASS' if not fails else str(fails) + ' FAIL'}")
raise SystemExit(1 if fails else 0)
