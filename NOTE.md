# Certified optimisation of the minimum-interaction qubit peel of "Wave packets from the spectrum"

**Moki & Julio (AI-assisted)** · October 2026 · version 1.0 · repository https://github.com/Hashirama1337s/certified-qubit-peel

## For the non-specialist
A recent paper builds a lattice of qubits out of nothing but a random list of energies. At each step it picks the qubit that
interacts least with everything else, using a grid search followed by local optimisation. We wrote a solver that finds the best
split under their own selection rule, for the kind of qubit their code builds, and returns a mathematical certificate that nothing
better exists in that search (so nobody has to trust our code), and a second, independent solver that agrees with it. Then we ran the authors' own published code, unchanged, on the exact random matrix behind their paper. Their
first qubit scores about twice as badly as the proven best, sitting on a different candidate, and the next three are two to
three and a half times worse; on the smaller splits their search finds the proven best. Their headline that the leftover
interaction is "of the order of a percent" holds up. We did not test their wave-packet and geometry results.

## Abstract
Cao, Friedrich, Girard, Loizeau and Singh (arXiv:2606.10004) factorise a Hamiltonian into qubits by iterated minimum-interaction
splits, selecting at each step the split with the smallest Tr(H_int^2)/Tr(H_q^2) from a grid of qubit energies optimised by BFGS.
With the qubit frozen, the optimum of their sorted-level loss is a minimum-weight perfect matching of the levels, and their
selection ratio over matchings is n - B^2/A (A = sum of squared pair gaps, B = sum of gaps, n = d/2). We solve this exactly with an
LP-dual certificate (Edmonds' odd-set inequalities, full pricing, certified bound), validated against an exact blossom solver and
against an independent integer-arithmetic implementation. Running the authors' published notebook unmodified on the matrix of their
figures (randH_complex(2^11, seed = 3), their full grid, 15,001 fits), the first split has ratio 0.012477 against the certified
optimum 0.0062026 (+101%), on a different Bessel peak (E_q 0.6179 versus 0.3196), and 18% above the certified best on its own peak.
Over all ten splits of their run the certified optimum beats their split by 101% to 237% at the four largest sizes (d = 2048 to
256) and coincides with it (within 0.01%) from d = 128 down.
A line-for-line port at a reduced grid misses the optimum at every first split tested (d = 512: 5/5, +35% to +111%; d = 2048: 3/3,
+63% to +148%). The certified optima lie within 1% of E_q = 2 pi/x_i (x_i zeros of J1), the first-pole condition of their
Eq. (43); their printed Eq. (47) sets E_q/2 = 2 pi/x_i, a gap twice as large. Their leftover-interaction figure survives (about 2-3%
at d = 2048: 1.6-2.9% over the spectra tested; their matrix 2.88% with their notebook, 2.90% exact per step).

## 1. Their selection rule and code
At each step their method splits one qubit off the spectrum. The paper defines the loss L = Tr(H_int^2) (their Eq. (12)) and,
among its local minima, keeps the one with the smallest ratio Tr(H_int^2)/Tr(H_q^2) (their Sec. 2.2), with H_q = diag(E_q/2, -E_q/2)
(their Eq. (10)). The search itself is in the published notebook: return_best_split freezes the qubit levels at (-q, +q) on a
coarse grid q = EA_max s/N_steps, s = 1..5000, then on a fine grid of 10,001 points around the coarse winner, fits the remaining
spectrum by BFGS at every point, and returns the best point of the fine grid. We report the ratio as Tr(H_int^2) over the full
space divided by the 2 x 2 trace of their H_q (= E_q^2/2); the notebook's stored score is this ratio divided by d, which does
not change which split wins. We ran the notebook at commit 4e354a71 (notebook sha256 e37b25bf...) unmodified, adding only
wrappers that record inputs and fits. On identical inputs their code and our line-for-line port agree (E_q identical, ratio to
5e-15).

## 2. Exact formulation and certification
With the qubit frozen, a split is a perfect matching of the d levels into n = d/2 pairs, one level of each pair going to each
qubit state. For a fixed matching with pair gaps g_k (upper minus lower level), the residual interaction at the best remaining
energies is Tr(H_int^2) = (1/2) sum_k (g_k - E_q)^2, the best qubit energy is E_q = A/B, and the ratio in the convention above is
n - B^2/A (A = sum of g_k^2, B = sum of g_k). Minimising it means minimising A/B^2 over matchings. The minimiser lies at a vertex
of the lower convex hull of the points (B, A) over all matchings, and each hull vertex is the minimum-weight perfect matching for
pair weights (g - delta)^2 at some delta. We compute those matchings exactly, as linear programmes over Edmonds' matching
polytope (odd-set inequalities added until the solution is integral, all pairs priced, the dual certifying optimality for that
delta); a branch-and-bound over delta in [0, lambda_max - lambda_min] finds all hull vertices, and a Dinkelbach iteration on
A - c B^2 selects the optimum among them. Validation: 258/258 instances agree with an exact blossom solver (worst difference
5.5e-16). An independent implementation, written blind from the specification, agrees with brute force over all 2,027,025
perfect matchings at d = 16 and over the 35,357,670 non-nesting matchings at d = 32 (a restricted check, not all matchings),
agrees on 24 levels (E_q to 5e-15, ratio to 6.5e-12), and at d = 2048 agrees with our solver on E_q to 15 digits and on the
ratio to between 1.3e-12 and 3.5e-11 on three spectra including their matrix (agreement between solvers, not enumeration).

## 3. Results
3.1 Their notebook on their matrix (prediction recorded before the fine scan): see abstract. Coarse winner index 1211 of 5000 (0-based, i.e. s = 1212);
    neither the low index wrap nor the upper-edge IndexError occurs at the first split. Deeper levels (their notebook, all ten
    splits, 10.71 h; certified optimum computed on each level's own input, i.e. their rest spectrum): d = 1024 +237%,
    d = 512 +148%, d = 256 +115%, and d = 128 down to 4 within 0.01% (their search finds the certified optimum). At d = 1024
    and 512 their split sits on the certified peak (E_q within 1%) but its ratio is 2.5 to 3.4 times the certified optimum; their d = 1024
    split lands on the peak that is optimal for the first split (E_q 0.3177 vs 0.3196).
3.2 Reduced-grid port, other spectra of their ensemble: d = 512 seeds 21-25 gaps (sorted) +35, +37, +54, +55, +111%; d = 2048 seeds 11-13
    +145, +148, +63% (seeds 11 and 12 pick the certified peak, yet the best of 401 of their fits placed within 2% of the certified qubit energy
    is still +110% and +133% above the certified optimum; seed 13 picks another peak because of its fits there).
3.3 Code observations, measured: the returned gradient omits a chain-rule term (mismatch with finite differences 85-108% at 25/25
    points); a fine-window index wraps when the coarse winner sits in the lowest scan_width of the grid (fired at the 512-level split
    in 3/3 reduced-grid d = 2048 runs; at their full grid it never fires: coarse winners at 0-based indices 186 to 3436 of 5000 against a
    window of 50, so 0/10 splits wrap and 0/10 reach the upper-edge IndexError on their matrix).
3.4 Peak locations: all certified first-split optima within 1% of 2 pi/x_i (8 spectra plus their matrix: 0.23%).
3.5 What survives and what was not tested: the leftover interaction of their own product construction is about 2-3% (their pipeline
    as ported, d = 2048 seeds 11-13: 2.05-2.62%; exact per-step 1.60-2.49%; their matrix with their notebook: theirs 2.88%, exact
    2.90%); the exact per-step optimum does not consistently lower the final leftover (their Sec. 4 already separates the per-step
    ratio from a global loss on the full split). Wave packets, MDS geometry and the
    Pauli-field case were not tested.

## 4. Fairness
Their search is a local method that they describe as preferring "certain minima"; they never claim the global optimum. Our findings
concern how far that search is from the optimum of their own criterion, not their physical conclusions.

## 5. Reproducibility
Code, logs and data: this repository (commands in README.md; `theory/d8_verify_note.py` recomputes every headline number
from the shipped results). Seeds are listed per result. Predictions were recorded in our lab notebook before each run; those
internal notes are not part of this release.

## References (all OPENED by us, with locators in the lab notes)
- Cao, Friedrich, Girard, Loizeau, Singh, "Wave packets from the spectrum", arXiv:2606.10004v1 (Secs. 2-4, Eqs. (12), (43), (47)).
- Notebook: github.com/nicolasloizeau/quantum_mereology, examples/code_for_Wave-packets-from-the-spectrum.ipynb, commit 4e354a71 (MIT).
- J. Edmonds, "Maximum matching and a polyhedron with 0,1-vertices", J. Res. Natl. Bur. Stand. 69B (1965) 125,
  doi:10.6028/jres.069b.013 (odd-set inequalities used in the LP certificate).
- Q. Huangfu and J. A. J. Hall, "Parallelizing the dual revised simplex method", Math. Program. Comput. 10 (2018) 119-142,
  doi:10.1007/s12532-017-0130-5 (HiGHS, the LP solver).
- Exact blossom reference: NetworkX max_weight_matching (Edmonds' blossom and primal-dual methods; documentation cites
  Z. Galil, "Efficient Algorithms for Finding Maximum Matching in Graphs", ACM Computing Surveys, 1986).
