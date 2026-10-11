"""Detached driver (driver added later, solver untouched): implementation B's exact split (exact_split4.BigLevel5) on a saved spectrum.
Usage: py run_npy.py <spectrum.npy> <tag> [cap_minutes]  -> RESULTS_<tag>.json, match_<tag>.json, log_<tag>.txt"""
import os
os.environ['OMP_NUM_THREADS'] = '2'
import sys, json, math, time, hashlib, threading
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from exact_split4 import BigLevel5 as BigLevel
src, tag = sys.argv[1], sys.argv[2]; cap = float(sys.argv[3]) if len(sys.argv) > 3 else 180.0
T0 = time.time(); threading.Timer(cap * 60 + 300, lambda: os._exit(3)).start()
OUT = os.path.join(HERE, 'RESULTS_%s.json' % tag); logf = open(os.path.join(HERE, 'log_%s.txt' % tag), 'a')
def log(msg):
    line = '[%7.0fs] %s' % (time.time() - T0, msg); print(line, flush=True); logf.write(line + '\n'); logf.flush()
lam = np.sort(np.load(src)); rec = dict(implementation='B (independent solver), driver run_npy.py', source=src, d=len(lam),
    lam_sha256=hashlib.sha256(np.ascontiguousarray(lam).tobytes()).hexdigest(), started=time.strftime('%Y-%m-%d %H:%M:%S'))
log('start %s d=%d sha %s' % (tag, len(lam), rec['lam_sha256'][:16]))
lv = BigLevel(lam, log=log); out = lv.split(deadline=T0 + cap * 60, progress_every=25)
rec.update(Eq=out['Eq_str'], R=out['R_str'], timed_out=out['timed_out'], R_upper_bound_certified=out['R_upper_bound'],
           R_lower_bound_certified=out['R_lower_bound'], n_oracle_calls=out['n_oracle'], split_time_s=round(out['time'], 1))
g = np.array([lam[j] - lam[i] for i, j in out['pairs']]); rec['R_plain_float64'] = repr(float(lv.n - g.sum() ** 2 / (g ** 2).sum()))
json.dump(out['pairs'], open(os.path.join(HERE, 'match_%s.json' % tag), 'w'))
rec['status'] = 'EXACT (all hull triangles closed)' if not out['timed_out'] else 'TIMED OUT: certified bounds only'
rec['total_time_s'] = round(time.time() - T0, 1); json.dump(rec, open(OUT, 'w'), indent=1)
log('done Eq=%s R=%s status=%s %.0fs' % (rec['Eq'], rec['R'], rec['status'], rec['total_time_s'])); os._exit(0)
