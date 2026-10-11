# Certified optimisation of the minimum-interaction qubit peel of "Wave packets from the spectrum"

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23293218.svg)](https://doi.org/10.5281/zenodo.23293218)

**Moki & Julio (AI-assisted) · 2026** · DOI [10.5281/zenodo.23293218](https://doi.org/10.5281/zenodo.23293218) (all versions; v1.0: 10.5281/zenodo.23293219) · note: [NOTE.md](NOTE.md)

## In plain words

A recent paper (Cao, Friedrich, Girard, Loizeau and Singh, arXiv:2606.10004) builds a lattice of qubits out of nothing but a
random list of energies. At each step it picks the qubit that interacts least with everything else, using a grid search
followed by local optimisation. We wrote a solver that finds the best split under their own selection rule, for the kind of
qubit their code builds, and returns a mathematical certificate that nothing better exists in that search, plus a second,
independently written solver that agrees with it. Then we ran the authors' own published notebook, unchanged, on the exact
random matrix behind their paper (10.7 hours, all ten splits).

- Their first qubit scores about twice as badly as the certified best (+101%), on a different candidate peak; the next three
  are 2 to 3.4 times worse; on the six smaller splits their search finds the certified best.
- Their headline that the leftover interaction is "of the order of a percent" holds up (2.88% with their notebook, 2.90% with
  the exact per-step optimum).
- The certified optima sit at E_q = 2 pi / x_i (x_i the zeros of the Bessel function J1), the first-pole condition of their
  Eq. (43); their printed Eq. (47) states a gap twice as large.
- They never claimed a global optimum; we did not test their wave-packet, geometry or Pauli-field results.

## What is in the repository

| Path | Contents |
|---|---|
| [NOTE.md](NOTE.md) | The note: method, certification, results, fairness, references. |
| `theory/d8_exact.py` | Certified exact split: minimum-weight perfect matchings as linear programmes over Edmonds' matching polytope (HiGHS), with a validation mode against an exact blossom solver. Uses `qubit_exact.py` (parametric hull search) and shared modules (`t1.py`, `probe/impl_a.py`, `probe/r5.py`). |
| `theory/d8_nbrun.py` | Runs their published notebook unmodified (wrappers only record inputs and fits). Their notebook is not redistributed; see below. |
| `theory/d8_exact_on.py` | Exact peel of their matrix (`PEEL`) and the certified optimum on each level their notebook produced (`ONNB`). |
| `theory/d8_theirs.py`, `d8_run.py`, `d8_run_nb.py`, `d8_run2048.py`, `d8_spike.py` | Line-for-line port of their search and the reduced-grid comparisons on other spectra. |
| `theory/d9_spikes.py` | Per-peak certified optima (used for the "on its own peak" comparison). |
| `theory/d8_verify_note.py` | Recomputes every headline number of the note from the shipped result files in seconds. |
| `theory/results/d8/` | All result files (JSON) of the comparisons; `nbrun/` holds the notebook run: per-level inputs (`.npy`), every fit (`.jsonl`), summaries and logs, and the exact results. |
| `theory/results/d9/npy_seed3_d2048.json` | Per-peak certified table for their matrix. |
| `probe/independent_b/` | The independent implementation, written from the specification only, with its sanity checks (brute force) and results. |

## Reproduce

Python 3.11 with numpy, scipy (HiGHS), networkx and torch (CPU is enough; torch is imported by the shared modules). Versions
used: [ENV.txt](ENV.txt). Run every command from `probe/`:

| What | Command | Time |
|---|---|---|
| Every headline number, from the shipped results | `python ../theory/d8_verify_note.py` | seconds |
| Certified oracle against an exact blossom solver | `python ../theory/d8_exact.py VALIDATE` | minutes |
| Exact peel of their matrix | `python ../theory/d8_exact_on.py PEEL` | about 1.5 h |
| Certified optimum on each level of their run | `python ../theory/d8_exact_on.py ONNB` | about 20 min |
| Independent implementation, brute-force checks | `python independent_b/sanity.py` | minutes |
| Their notebook as published | `python -I -u ../theory/d8_nbrun.py FULL` | about 11 h |

For the last command, place their notebook at `third_party/quantum_mereology_4e354a71/code_for_Wave-packets-from-the-spectrum.ipynb`
from github.com/nicolasloizeau/quantum_mereology at commit 4e354a71346692c2a2bb2e383f0410ecd18b259c
(`examples/`, MIT License, (c) 2025 Nicolas Loizeau); the script checks its SHA-256
(e37b25bf1e3ce775da81fe05e5c06b9fc8c2bf219f40c6f206554e8d68f1e46c) before running.

## Licence and citation

Text and data: CC BY 4.0 ([LICENSE](LICENSE)). Code: MIT ([LICENSE-CODE](LICENSE-CODE)). See [CITATION.cff](CITATION.cff).
