# Captions - certified-qubit-peel

Run: `python figures/make_figures.py .` from the repository root (Python 3.11, numpy, scipy, matplotlib; a few
seconds). The script checks that the stored gaps equal their ratio / certified ratio - 1, that the notebook's best fit is
the recorded pick (E_q 0.6179, ratio 0.012477), that the stored peak centres equal 2 pi / (zeros of J1) computed with
scipy, and that the best of the ten per-peak optima equals the certified first-split optimum. It stops if any check fails.

Fairness line for both captions (from README): their search is a local method and they never claimed a global optimum;
the figures measure distance from the optimum of their own selection rule, not their physical conclusions.

## 1. `figures/gap_per_split.svg` (hero)

**Point.** On their own matrix, with their own notebook, the published search lands 2.0 to 3.4 times above the proven
optimum on the four largest splits and finds the optimum on the six smaller ones.

**Caption.** The published qubit-building search (the authors' notebook, unchanged, on the matrix behind their paper)
against the certified optimum of its own selection rule (for the kind of qubit their code builds), for each of its ten splits. Each bar is their selection ratio
divided by the certified best for the same input, so 1 means they found the best. On the four largest splits (2,048 down
to 256 levels) their split is 2.01, 3.37, 2.48 and 2.15 times the optimum; from 128 levels down it matches the optimum to
within 0.01 %. Their search is a local method (a grid of trial energies, then local optimisation), and they never claimed a global optimum; their headline result, a leftover interaction of about 2.9 %, holds either way (2.88 % with their notebook, 2.90 % with the exact optimum at each step).

**Alt text.** Bar chart with ten bars labelled 2,048 to 4 levels: 2.01x, 3.37x, 2.48x and 2.15x for the first four,
then six bars at 1.00x on a dashed line marking the certified best.

**Sources.**
| item | file | field |
|---|---|---|
| their ratio, certified ratio, d, per split | `theory/results/d8/nbrun/exact_onnb.json` | `rows[*].nb_ratio`, `.exact_ratio`, `.d`, `.level` |
| +101 / +237 / +148 / +115 % | same | `rows[*].gap` (1.0116, 2.3658, 1.4775, 1.1521) |
| within 0.01 % for d = 128..4 | same | `gap` <= 5.2e-5 (largest at d = 16) |
| "unchanged notebook, matrix behind their paper" | `README.md`, `NOTE.md` Section 1 and 3.1 | |

## 2. `figures/first_split_peaks.svg`

**Point.** At the first split their search ends in the dip near E_q = 0.62, while the proven best sits in a different dip
near 0.32; the good dips sit at 2 pi / (zeros of J1).

**Caption.** The first split (2,048 levels), seen as a landscape. Grey: the selection ratio of each of the 15,001 fits
their notebook made, one per trial qubit energy E_q. Blue: the certified best split inside each of the first ten dips,
whose centres lie at 2 pi divided by the zeros of the Bessel function J1 (dotted lines; the README calls the dips
"peaks"). Their final pick (orange, ratio 0.012477 at E_q 0.6179) is in the third dip and 18 % above the best in that
dip; the certified best of the whole first split, 0.0062026 at E_q 0.3196, is in the sixth dip, where their fits stay far higher. Both axes are logarithmic; lower is better. Their search is a local method and they never claimed a global optimum; this measures distance from the optimum of their own selection rule, not their physical conclusions.

**Alt text.** Log-log scatter of selection ratio against qubit energy: a grey curve of fits with deep dips at dotted
vertical lines, ten blue dots below the dips, an orange diamond at the bottom of the dip near 0.62, and a label for the
lowest blue dot near 0.32.

**Sources.**
| item | file | field |
|---|---|---|
| grey fits (E_q = 2q, ratio = obj x 2048) | `theory/results/d8/nbrun/level1_fits_d2048.jsonl` | `q`, `obj` (stored score = ratio / d, `NOTE.md` Section 1) |
| their pick 0.012477, E_q 0.6179 | `theory/results/d8/nbrun/nbrun_FULL.json` | `levels[0].ratio_cao`, `.E_q` |
| blue dots, dip centres | `theory/results/d9/npy_seed3_d2048.json` | `runs[0].rows[*].E_q`, `.ratio`, `.E_i`, `.cell` (input `level1_input_d2048.npy`) |
| certified best 0.0062026, E_q 0.3196 | same, cell 6; equals `exact_onnb.json` `rows[0].exact_ratio`, `.exact_E_q` | |
| 18 % above the best in its own dip | 0.012477 / 0.010566 (cell 3 `ratio`) - 1 | `README.md`: "18 % above the certified best on its own peak" |
| 2 pi / zeros of J1 | `theory/d9_spikes.py` (cell definition); recomputed with scipy `jn_zeros` | |

Display note: of the 15,001 fits, only exact repeats on a 0.25-point grid of the drawing are dropped before plotting (to
keep the SVG small); every visible dot is a real fit, and every fit in the window is drawn at its own position or
under an identical one. The plotted window is E_q 0.17 to 2.2; fits outside it (their grid runs from E_q 0.0005 to 2.55)
are not drawn, and the y range is set from the fits in the window so none is clipped.

Referee note (2026-11-17): the draft caption said "The authors describe their search as a local method"; the README
says only that it is a local method which they describe as preferring "certain minima". Reworded so no words are put
in the authors' mouths; the scope "for the kind of qubit their code builds" and the headline-holds sentence are taken
from the README ("In plain words", "What we do not claim").
