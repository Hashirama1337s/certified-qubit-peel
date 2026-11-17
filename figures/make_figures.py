"""Figures for the certified qubit-peel repository, drawn only from files the repository ships.

usage:  python figures/make_figures.py [repo_root]      (repo_root defaults to the folder above this script)
writes: figures/gap_per_split.svg, figures/first_split_peaks.svg  (next to this script)

Sources: theory/results/d8/nbrun/exact_onnb.json (their notebook vs the certified optimum, all ten splits),
theory/results/d8/nbrun/level1_fits_d2048.jsonl (the 15,001 fits of their notebook at the first split),
theory/results/d8/nbrun/nbrun_FULL.json (their notebook's pick), theory/results/d9/npy_seed3_d2048.json (certified best
split inside each of the first ten peaks of the same spectrum). Needs numpy, scipy (zeros of J1), matplotlib.
"""
import json, os, sys
import numpy as np
from scipy.special import jn_zeros
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.dirname(HERE)
OUT = HERE if os.path.basename(HERE) == "figures" else os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
RES = os.path.join(ROOT, "theory", "results")

# ---- lab style (STYLE.md) -------------------------------------------------------------------------------------------
BG, BORDER, GRID, AX, TXT, ACC = "#1c2026", "#2f353d", "#22272e", "#9aa4ae", "#e8edf2", "#39ff14"
CAT = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767"]
NEUTRAL = "#5b636c"
plt.rcParams.update({
    "svg.fonttype": "none", "svg.hashsalt": "certified-qubit-peel",
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 14, "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": AX, "axes.labelcolor": TXT, "text.color": TXT, "xtick.color": AX, "ytick.color": AX,
    "axes.titlecolor": TXT, "grid.color": GRID, "grid.linewidth": 0.8, "lines.linewidth": 2,
})
W_IN = 12.5


def frame(fig):
    fig.patches.append(FancyBboxPatch((0.004, 0.004), 0.992, 0.992, boxstyle="round,pad=0,rounding_size=0.012",
                                      transform=fig.transFigure, fill=False, ec=BORDER, lw=1.5, zorder=-1))


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), format="svg", metadata={"Date": None, "Creator": None})
    plt.close(fig)
    print("wrote figures/" + name)


def style_ax(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)


# ---- figure 1: their ratio / certified best, per split -----------------------------------------------------------------
def fig_gap():
    rows = json.load(open(os.path.join(RES, "d8", "nbrun", "exact_onnb.json")))["rows"]
    rows = sorted(rows, key=lambda r: r["level"])
    mult = [r["nb_ratio"] / r["exact_ratio"] for r in rows]
    for r, m in zip(rows, mult):
        assert abs((m - 1) - r["gap"]) < 1e-9 and m >= 1 - 1e-9      # stored gap = their / certified - 1
    fig, ax = plt.subplots(figsize=(W_IN, 6.6))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.76, bottom=0.17)
    frame(fig)
    fig.text(0.03, 0.915, "A published qubit search vs the proven best: 2 to 3.4 times the optimum on the four largest splits",
             fontsize=17.5, weight="bold")
    fig.text(0.03, 0.865, "Their own notebook, unchanged, on the matrix behind their paper. Each split halves the number of "
                          "levels; smaller ratio is better.", fontsize=13, color=AX)
    fig.text(0.03, 0.825, "Bar = their selection ratio divided by the certified optimum for the same input "
                          "(1 = they found the best).", fontsize=13, color=AX)
    x = np.arange(len(rows))
    ax.bar(x, mult, width=0.62, color=ACC, ec=BG, lw=0)
    ax.axhline(1.0, color=TXT, lw=1.4, ls=(0, (5, 4)), label="certified best (= 1)")
    for xi, m, r in zip(x, mult, rows):
        lab = f"{m:.2f}x\n(+{100 * (m - 1):.0f} %)" if m - 1 > 1e-4 else "1.00x"
        ax.text(xi, m + 0.06, lab, ha="center", va="bottom", fontsize=12.5, color=TXT, linespacing=1.15)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['d']:,}" for r in rows])
    ax.set_xlabel("levels before the split (split 1 on the left, split 10 on the right)")
    ax.set_ylabel("their ratio / certified best")
    ax.set_ylim(0, max(mult) * 1.25)
    ax.grid(axis="y"); style_ax(ax)
    ax.legend(loc="upper right", frameon=False, fontsize=12.5, handlelength=2.6)
    save(fig, "gap_per_split.svg")
    return [(r["d"], round(m, 4)) for r, m in zip(rows, mult)]


# ---- figure 2: the first split, their search vs certified peak minima -----------------------------------------------------
def fig_peaks():
    d = 2048
    fits = [json.loads(l) for l in open(os.path.join(RES, "d8", "nbrun", "level1_fits_d2048.jsonl"))]
    E = np.array([2 * f["q"] for f in fits])                    # qubit levels (-q, +q): E_q = 2q
    R = np.array([f["obj"] * d for f in fits])                  # stored score = ratio / d (NOTE.md, Section 1)
    nb = json.load(open(os.path.join(RES, "d8", "nbrun", "nbrun_FULL.json")))["levels"][0]
    assert nb["d"] == d and abs(E[np.argmin(R)] - nb["E_q"]) < 1e-12 and abs(R.min() - nb["ratio_cao"]) < 1e-12
    cells = json.load(open(os.path.join(RES, "d9", "npy_seed3_d2048.json")))
    assert cells["source"].endswith("level1_input_d2048.npy") and cells["d"] == d
    rows = cells["runs"][0]["rows"]
    zeros = jn_zeros(1, len(rows))
    for r in rows:                                              # peak centres E_i = 2 pi / x_i, x_i = zeros of J1
        assert abs(r["E_i"] - 2 * np.pi / zeros[r["cell"] - 1]) < 1e-9
    best = min(rows, key=lambda r: r["ratio"])
    ex = next(r for r in json.load(open(os.path.join(RES, "d8", "nbrun", "exact_onnb.json")))["rows"] if r["level"] == 1)
    assert abs(best["ratio"] - ex["exact_ratio"]) < 1e-12 and abs(best["E_q"] - ex["exact_E_q"]) < 1e-12
    own = min(rows, key=lambda r: abs(r["E_i"] - nb["E_q"]))   # the peak their pick sits on

    xl = (0.17, 2.2)
    shown = (E >= xl[0]) & (E <= xl[1])
    fig, ax = plt.subplots(figsize=(W_IN, 7.4))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.78, bottom=0.14)
    frame(fig)
    fig.text(0.03, 0.925, "First split: the search settles in one dip; the proven best is in another",
             fontsize=19.5, weight="bold")
    fig.text(0.03, 0.875, "Good qubit energies come in separate dips (the README's 'peaks'), centred near 2π divided by a zero of "
                          "the Bessel function J1 (dotted lines).", fontsize=13, color=AX)
    fig.text(0.03, 0.835, "Grey: the selection ratio of each of their 15,001 fits. Blue: the certified best inside each dip. "
                          "Lower is better.", fontsize=13, color=AX)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(*xl)
    ax.set_ylim(0.0025, 1.15 * float(R[shown].max()))
    for r in rows:
        if xl[0] <= r["E_i"] <= xl[1]:
            ax.axvline(r["E_i"], color=AX, lw=0.9, ls=(0, (1, 3)), zorder=0)
    # 15,001 fits: drop only exact repeats on a 0.25-point grid of the drawing (far below the 2-point dot size) to keep the SVG small
    P = ax.transData.transform(np.column_stack([E[shown], R[shown]]))
    _, keep = np.unique(np.round(P * 4).astype(np.int64), axis=0, return_index=True)
    keep = np.sort(keep)
    ax.scatter(E[shown][keep], R[shown][keep], s=4, color=NEUTRAL, lw=0, zorder=1)
    ax.plot([r["E_q"] for r in rows], [r["ratio"] for r in rows], ls="none", marker="o", ms=9, color=CAT[0], zorder=3)
    ax.plot([nb["E_q"]], [nb["ratio_cao"]], ls="none", marker="D", ms=11, color=CAT[1], mec=BG, mew=1.2, zorder=4)
    ax.annotate(f"their pick: {nb['ratio_cao']:.6f} (E_q {nb['E_q']:.4f}),\n{100 * (nb['ratio_cao'] / own['ratio'] - 1):.0f} % "
                f"above the best in its own dip", xy=(nb["E_q"], nb["ratio_cao"]), xytext=(0.80, 0.0050),
                fontsize=12.5, color=CAT[1], ha="left", va="center", bbox=dict(fc=BG, ec="none", pad=2),
                arrowprops=dict(arrowstyle="-", color=CAT[1], lw=1.2, shrinkB=7))
    ax.annotate(f"certified best: {best['ratio']:.7f} (E_q {best['E_q']:.4f})", xy=(best["E_q"], best["ratio"]),
                xytext=(0.20, 0.0034), fontsize=12.5, color=CAT[0], ha="left", va="center", bbox=dict(fc=BG, ec="none", pad=2),
                arrowprops=dict(arrowstyle="-", color=CAT[0], lw=1.2, shrinkB=7))
    ticks = [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    ax.set_xticks(ticks); ax.set_xticklabels([f"{t:g}" for t in ticks]); ax.minorticks_off()
    yt = [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    yt = [t for t in yt if ax.get_ylim()[0] <= t <= ax.get_ylim()[1]]
    ax.set_yticks(yt); ax.set_yticklabels([f"{t:g}" for t in yt])
    ax.set_xlabel("qubit energy E_q (log scale)")
    ax.set_ylabel("selection ratio (log scale)")
    ax.grid(True); style_ax(ax)
    hs = [Line2D([], [], ls="none", marker="o", ms=5, color=NEUTRAL), Line2D([], [], ls="none", marker="o", ms=9, color=CAT[0]),
          Line2D([], [], ls="none", marker="D", ms=10, color=CAT[1])]
    ax.legend(hs, ["their fits (one per trial energy)", "certified best in each dip", "their final pick"],
              loc="upper left", frameon=True, facecolor=BG, edgecolor=BG, framealpha=1, fontsize=12.5)
    save(fig, "first_split_peaks.svg")
    return dict(their=(nb["E_q"], nb["ratio_cao"]), best=(best["E_q"], best["ratio"], best["cell"]),
                own_peak=(own["cell"], own["ratio"]), n_fits=len(fits), shown_max=float(R[shown].max()))


if __name__ == "__main__":
    print("  gap per split:", fig_gap())
    print("  first split:", fig_peaks())
