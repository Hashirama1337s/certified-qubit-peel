"""BAR-D8 Amendment 4: execute the published notebook unmodified.
Source: github.com/nicolasloizeau/quantum_mereology @ 4e354a71346692c2a2bb2e383f0410ecd18b259c,
examples/code_for_Wave-packets-from-the-spectrum.ipynb (MIT License, (c) 2025 Nicolas Loizeau), pinned copy in
third_party/quantum_mereology_4e354a71/ (sha256 checked before use).
Code cells 1, 3, 5, 6, 8 are exec'd verbatim; FULL then execs the run cell (cell 9) verbatim. Instrumentation only replaces
two names in the notebook namespace (partition, return_best_split) with wrappers that call the originals unchanged and record
inputs and outputs. No numerical code is altered; the notebook's own prints go to stdout.
Usage (from probe/, interpreter run with -I -u):
  py ../theory/d8_nbrun.py PROBE <seed>   d = 512, N_steps 200, refining 10: their code vs the port (d8_theirs) on one input
  py ../theory/d8_nbrun.py TIME <k>       k of their fits on the seed-3 matrix (d = 2048) near the expected optimum
  py ../theory/d8_nbrun.py FULL           cell 9 as published (randH_complex(2**11, seed=3), N_steps 5000) -> results/d8/nbrun/"""
import sys, os, json, time, hashlib
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
NB = os.path.join(HERE, "..", "third_party", "quantum_mereology_4e354a71", "code_for_Wave-packets-from-the-spectrum.ipynb")
NB_SHA = "e37b25bf1e3ce775da81fe05e5c06b9fc8c2bf219f40c6f206554e8d68f1e46c"
OUT = os.path.join(HERE, "results", "d8", "nbrun"); os.makedirs(OUT, exist_ok=True)
mode = sys.argv[1]
LOG = open(os.path.join(OUT, f"nbrun_{mode}.log"), "a", buffering=1)
def say(*a): print(time.strftime("%Y-%m-%d %H:%M:%S"), *a, file=LOG, flush=True)

raw = open(NB, "rb").read(); sha = hashlib.sha256(raw).hexdigest()
assert sha == NB_SHA, f"notebook sha256 mismatch: {sha}"
cells = json.loads(raw.decode("utf-8"))["cells"]
ns = {"__name__": "notebook"}
for i in (1, 3, 5, 6, 8):
    assert cells[i]["cell_type"] == "code"
    exec(compile("".join(cells[i]["source"]), f"<notebook cell {i}>", "exec"), ns)
RUN_CELL = "".join(cells[9]["source"])
say("loaded notebook", sha[:16], "| python", sys.version.split()[0], "| numpy", np.__version__,
    "| OPENBLAS_NUM_THREADS", os.environ.get("OPENBLAS_NUM_THREADS"), "| OMP_NUM_THREADS", os.environ.get("OMP_NUM_THREADS"))

# ---- instrumentation (wrappers only) ----
orig_partition, orig_split = ns["partition"], ns["return_best_split"]
FIT = dict(n=0, level=None, n_level=0, t_level=None, file=None, expect=None)
def partition_traced(*args, **kwargs):
    E, y0 = args[0], kwargs.get("y0"); t = time.time()
    res = orig_partition(*args, **kwargs)
    d = len(E); loss = float(np.sum(res.Eint ** 2) / d); sA = float(np.sum(res.EA ** 2))
    FIT["n"] += 1; FIT["n_level"] += 1
    if FIT["file"] is not None:
        FIT["file"].write(json.dumps(dict(d=d, k=FIT["n_level"], q=float(y0[1]), loss=loss, sumEA2=sA, obj=loss / sA,
                                          secs=round(time.time() - t, 3))) + "\n")
    if FIT["n_level"] % 100 == 0 and FIT["expect"]:
        el = time.time() - FIT["t_level"]; rate = el / FIT["n_level"]
        say(f"  d={d} fit {FIT['n_level']}/{FIT['expect']}  {rate:.2f} s/fit  level ETA {(FIT['expect'] - FIT['n_level']) * rate / 3600:.2f} h")
    return res
LEVELS = []
def split_traced(evals, N_steps, refining_factor=100):
    d = len(evals); lvl = len(LEVELS) + 1
    np.save(os.path.join(OUT, f"level{lvl}_input_d{d}.npy"), evals)
    FIT.update(level=lvl, n_level=0, t_level=time.time(), expect=N_steps + 2 * (N_steps // refining_factor) * refining_factor + 1,
               file=open(os.path.join(OUT, f"level{lvl}_fits_d{d}.jsonl"), "w", buffering=1))
    say(f"level {lvl} start: d={d}, N_steps={N_steps}, refining_factor={refining_factor}, expected fits {FIT['expect']}")
    t0 = time.time()
    try:
        EA, EB, Eint = orig_split(evals, N_steps, refining_factor)
    except Exception as e:
        LEVELS.append(dict(level=lvl, d=d, crashed=f"{type(e).__name__}: {e}", fits=FIT["n_level"], secs=round(time.time() - t0, 1)))
        save_levels(); say(f"level {lvl} CRASHED in their code: {type(e).__name__}: {e}"); raise
    finally:
        FIT["file"].close(); FIT["file"] = None
    row = dict(level=lvl, d=d, EA=[float(x) for x in EA], E_q=float(EA[1] - EA[0]), sumEA2=float(np.sum(EA ** 2)),
               loss=float(np.sum(Eint ** 2) / d), ratio_cao=float(np.sum(Eint ** 2) / np.sum(EA ** 2)), fits=FIT["n_level"],
               secs=round(time.time() - t0, 1))
    np.save(os.path.join(OUT, f"level{lvl}_EB_d{d}.npy"), EB); np.save(os.path.join(OUT, f"level{lvl}_Eint_d{d}.npy"), Eint)
    LEVELS.append(row); save_levels(); say(f"level {lvl} done: " + json.dumps(row))
    return EA, EB, Eint
def save_levels(extra=None):
    json.dump(dict(notebook_sha256=sha, commit="4e354a71346692c2a2bb2e383f0410ecd18b259c", python=sys.version.split()[0],
                   numpy=np.__version__, threads=os.environ.get("OPENBLAS_NUM_THREADS"), levels=LEVELS, **(extra or {})),
              open(os.path.join(OUT, f"nbrun_{mode}.json"), "w"), indent=1)
ns["partition"], ns["return_best_split"] = partition_traced, split_traced

if mode == "PROBE":
    import d8_exact as D, d8_theirs as T
    seed = int(sys.argv[2]); lam = np.sort(D.gue_spectrum(512, seed))
    t0 = time.time(); EA, EB, Eint = ns["return_best_split"](lam, 200, refining_factor=10); t1 = time.time()
    port = T.best_split_nb(lam, N_steps=200, refining_factor=10); t2 = time.time()
    theirs = dict(E_q=float(EA[1] - EA[0]), ratio=float(np.sum(Eint ** 2) / np.sum(EA ** 2)))
    fits = [json.loads(l) for l in open(os.path.join(OUT, "level1_fits_d512.jsonl"))]
    obj_c = np.array([f["obj"] for f in fits[:200]]); ind_min = int(np.argmin(obj_c))
    stored = json.load(open(os.path.join(HERE, "results", "d8", f"run_nb_d512_s{seed}.json")))["nb"][0]
    cmp = dict(seed=seed, theirs=theirs, port=dict(E_q=port["E_q"], ratio=port["ratio_cao"], ind_min=port["ind_min"]),
               ind_min_theirs=ind_min, rel_dEq=abs(theirs["E_q"] - port["E_q"]) / port["E_q"],
               rel_dratio=abs(theirs["ratio"] - port["ratio_cao"]) / port["ratio_cao"],
               rest_maxdiff=float(np.max(np.abs(np.sort(EB) - port["rest"]))),
               stored_run=dict(E_q=stored["E_q"], ratio=stored["ratio"]), secs_theirs=round(t1 - t0, 1), secs_port=round(t2 - t1, 1))
    cmp["PASS"] = bool(cmp["rel_dEq"] <= 1e-9 and cmp["rel_dratio"] <= 1e-9 and ind_min == port["ind_min"])
    say("PROBE " + json.dumps(cmp)); save_levels(dict(probe=cmp))
    os.replace(os.path.join(OUT, "nbrun_PROBE.json"), os.path.join(OUT, f"nbrun_PROBE_s{seed}.json"))
elif mode == "TIME":
    k = int(sys.argv[2]); H = ns["randH_complex"](2 ** 11, seed=3); H -= np.trace(H) * np.identity(2 ** 11) / 2 ** 11
    evals = np.sort(np.linalg.eigh(H)[0]); d = len(evals); EA_max = 1.5 * np.mean(evals[d // 2:]); secs = []
    for s in range(900, 900 + k):          # their coarse-loop construction of y0 at s (q near the expected spike)
        y0 = np.zeros(2 + d // 2); q = EA_max * s / 5000; y0[0] = -q; y0[1] = q
        np.random.seed(s); dummy = evals.copy(); np.random.shuffle(dummy); y0[2:] = 0.5 * (dummy[::2] + dummy[1::2])
        t = time.time(); res = orig_partition(evals, 2, y0=y0, optimizeA=False); secs.append(time.time() - t)
        say(f"TIME s={s} q={q:.5f} {secs[-1]:.2f}s obj={np.sum(res.Eint ** 2) / d / np.sum(res.EA ** 2):.6g}")
    say(f"TIME threads={os.environ.get('OPENBLAS_NUM_THREADS')} mean {np.mean(secs):.2f} s/fit (k={k}); "
        f"one split = 15001 fits -> {15001 * np.mean(secs) / 3600:.2f} h")
elif mode == "FULL":
    say("FULL: executing notebook cell 9 verbatim:\n" + RUN_CELL)
    t0 = time.time()
    exec(compile(RUN_CELL, "<notebook cell 9>", "exec"), ns)
    Hp = ns["H_prod"]; sZ = ns["sigmas_Z"]; n = len(sZ)
    zE = [float(np.real(np.trace(Z.dot(Hp))) / 2 ** n) for Z in sZ]
    np.save(os.path.join(OUT, "H_prod_diag.npy"), np.real(np.diag(Hp)))
    save_levels(dict(cell9_completed=True, secs_total=round(time.time() - t0, 1), sigmaZ_energies_final_order=zE,
                     n_c=len(ns["c_list"]), n_cReal=len(ns["cReal_list"])))
    say(f"FULL DONE in {(time.time() - t0) / 3600:.2f} h; sigma_Z energies (final order): {zE}")
