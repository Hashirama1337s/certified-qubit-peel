"""Door 9: which spike wins? Certified best one-qubit split restricted to each Bessel-zero cell.
Selection ratio over perfect matchings M of the levels (pair gaps g): R(M) = n - B^2/A, A = sum g^2, B = sum g, E_q = A/B
(the door-8 criterion of arXiv:2606.10004). Envelope matchings M(delta) minimise A - 2 delta B (certified oracle d8_exact.warm_matching).
window_best(lam, dlo, dhi): exact best R over envelope matchings optimal at some delta in [dlo, dhi]. 1-D branch-and-bound: the
(B, A) points of envelope matchings between delta = d1 and d2 lie in the triangle (M(d1), M(d2), P), P = intersection of the two
supporting lines; B^2/A is convex for A > 0, so its maximum over the triangle is at a vertex (valid bound). Equal matchings at
both ends of an interval mean h is that single line there (concavity), so the interval is done.
Cells: E_i = 2 pi / x_i (x_i = zeros of J1), boundaries at geometric midpoints; cell 0 = [b_1, spread], tail = [tiny, b_I].
Usage (from probe/):
  py ../theory/d9_spikes.py VALIDATE                 min over cells == exact_cao_split global; window result >= dense-grid samples
  py ../theory/d9_spikes.py QUANTILE <d> [I]         semicircle quantile spectrum (no randomness)
  py ../theory/d9_spikes.py GUE <d> <seed0> <nseeds> [I]
  py ../theory/d9_spikes.py NPY <spectrum.npy> <tag> [I]"""
import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d8_exact as D, qubit_exact as QE
from scipy.special import jn_zeros

OUT = os.path.join(QE.t1.OUT, "d9"); os.makedirs(OUT, exist_ok=True)


def window_best(lam, dlo, dhi, cache, tol=1e-12):
    n = len(lam) // 2
    def q(d):
        if d not in cache: cache[d] = D.warm_matching(lam, d)
        return cache[d]
    def val(M): return M[2] ** 2 / M[1] if M[1] > 0 else -np.inf          # B^2/A (maximise)
    Ma, Mb = q(dlo), q(dhi); best = max((Ma, Mb), key=val); stack = [(dlo, Ma, dhi, Mb)]; nodes = 0
    while stack:
        d1, Ma, d2, Mb = stack.pop(); nodes += 1
        if sorted(Ma[0]) == sorted(Mb[0]): continue
        h1, h2 = Ma[1] - 2 * d1 * Ma[2], Mb[1] - 2 * d2 * Mb[2]
        BP = (h1 - h2) / (2 * (d2 - d1)); AP = h1 + 2 * d1 * BP
        ub = max(val(Ma), val(Mb), BP * BP / AP if AP > 0 else np.inf)
        if ub <= val(best) * (1 + tol): continue
        dt = (Mb[1] - Ma[1]) / (2 * (Mb[2] - Ma[2])) if abs(Mb[2] - Ma[2]) > 1e-15 else 0.5 * (d1 + d2)
        dm = dt if d1 + 1e-12 < dt < d2 - 1e-12 else 0.5 * (d1 + d2)
        Mm = q(dm)
        if val(Mm) > val(best): best = Mm
        if dm - d1 < 1e-13 or d2 - dm < 1e-13: continue
        stack.append((d1, Ma, dm, Mm)); stack.append((dm, Mm, d2, Mb))
    pairs, A, B = best
    return dict(E_q=A / B, ratio=n - B * B / A, Q=(n - B * B / A) / n, nodes=nodes)


def cells(lam, I):
    E = 2 * np.pi / jn_zeros(1, I + 1); b = np.sqrt(E[:-1] * E[1:]); spread = lam[-1] - lam[0]
    out = [(1, b[0], spread)] + [(i + 1, b[i], b[i - 1]) for i in range(1, I)]
    return out, float(E[0]), b


def spike_table(lam, I, tail=False):
    cache = {}; rows = []; cl, _, b = cells(lam, I)
    if tail: cl = cl + [(0, 1e-9, b[I - 1])]
    for i, lo, hi in cl:
        t = time.time(); n0 = len(D.STATS); r = window_best(lam, lo, hi, cache)
        r.update(cell=i, lo=float(lo), hi=float(hi), E_i=float(2 * np.pi / jn_zeros(1, i)[-1]) if i > 0 else None,
                 calls=len(D.STATS) - n0, secs=round(time.time() - t, 1)); rows.append(r)
    return rows


def semicircle_quantiles(d):
    u = (np.arange(d) + 0.5) / d; lo, hi = np.full(d, -2.0), np.full(d, 2.0)
    F = lambda x: 0.5 + x * np.sqrt(np.maximum(4 - x * x, 0)) / (4 * np.pi) + np.arcsin(np.clip(x / 2, -1, 1)) / np.pi
    for _ in range(80):
        m = 0.5 * (lo + hi); s = F(m) < u; lo = np.where(s, m, lo); hi = np.where(s, hi, m)
    return 0.5 * (lo + hi)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "VALIDATE":
        worst = 0.0; worst_grid = 0.0; rng = np.random.default_rng(909)
        for d in (16, 32, 64, 128):
            for seed in range(6 if d <= 64 else 3):
                lam = np.sort(D.gue_spectrum(d, 900 + seed)); rows = spike_table(lam, 6, tail=True)
                g, _, _, _ = D.exact_cao_split(lam); mc = min(r["ratio"] for r in rows)
                worst = max(worst, abs(mc - g["ratio"]) / g["ratio"])
                for r in rows[:3]:                                          # dense grid inside the first cells
                    for dl in np.linspace(r["lo"], r["hi"], 200):
                        p, A, B = D.warm_matching(lam, dl); rg = len(lam) // 2 - B * B / A
                        worst_grid = max(worst_grid, (r["ratio"] - rg) / g["ratio"])
                print(f"d={d} seed={900+seed}: global {g['ratio']:.10g} (E_q {g['E_q']:.6f}); min over cells {mc:.10g}; "
                      f"winner cell {min(rows, key=lambda r: r['ratio'])['cell']}", flush=True)
        ok = worst <= 1e-9 and worst_grid <= 1e-9
        print(f"VALIDATE {'PASS' if ok else 'FAIL'}: |min cells - global|/global worst {worst:.2e}; window minus grid sample worst {worst_grid:.2e}")
    elif mode == "NPY":                        # per-cell table for a saved spectrum: NPY <path> <tag> [I]
        lam = np.sort(np.load(sys.argv[2])); tag = sys.argv[3]; I = int(sys.argv[4]) if len(sys.argv) > 4 else 10; t0 = time.time()
        rows = spike_table(lam, I); w = min(rows, key=lambda r: r["ratio"])
        json.dump(dict(mode="NPY", source=sys.argv[2], d=len(lam), I=I, runs=[dict(seed=tag, rows=rows, winner=w["cell"])]),
                  open(os.path.join(OUT, f"npy_{tag}.json"), "w"), indent=1, default=float)
        print(f"NPY {tag} d={len(lam)}: winner cell {w['cell']} ratio {w['ratio']:.6g} E_q {w['E_q']:.6f} | " +
              " ".join(f"{r['cell']}:{r['ratio']:.4g}" for r in rows) + f" | {time.time()-t0:.1f}s", flush=True)
    elif mode in ("QUANTILE", "GUE"):
        d = int(sys.argv[2]); seeds = [None] if mode == "QUANTILE" else list(range(int(sys.argv[3]), int(sys.argv[3]) + int(sys.argv[4])))
        I = int(sys.argv[3 if mode == "QUANTILE" else 5]) if len(sys.argv) > (3 if mode == "QUANTILE" else 5) else 10
        tag = f"{mode.lower()}_d{d}" + ("" if mode == "QUANTILE" else f"_s{seeds[0]}-{seeds[-1]}")
        res = dict(mode=mode, d=d, I=I, runs=[])
        for s in seeds:
            lam = semicircle_quantiles(d) if s is None else np.sort(D.gue_spectrum(d, s)); t0 = time.time()
            rows = spike_table(lam, I); w = min(rows, key=lambda r: r["ratio"])
            res["runs"].append(dict(seed=s, rows=rows, winner=w["cell"], secs=round(time.time() - t0, 1)))
            json.dump(res, open(os.path.join(OUT, tag + ".json"), "w"), indent=1, default=float)
            print(f"{mode} d={d} seed={s}: winner cell {w['cell']} ratio {w['ratio']:.6g} E_q {w['E_q']:.6f} | " +
                  " ".join(f"{r['cell']}:{r['ratio']:.4g}" for r in rows) + f" | {time.time()-t0:.1f}s", flush=True)
