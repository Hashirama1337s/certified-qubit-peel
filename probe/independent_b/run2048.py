"""Detached driver: d=2048, level 1 only, seeds 11 then 12, hard cap 3 h total.
Writes RESULTS_2048.json after each seed (and match_d2048_s{seed}.json)."""
import os
os.environ['OMP_NUM_THREADS'] = '2'
import sys
import json
import math
import time
import hashlib
import threading
from fractions import Fraction
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exact_split4 import spectrum, BigLevel5 as BigLevel, fmt

T0 = time.time()
HARD_CAP = 150 * 60               # hard kill: whole d=2048 effort < 3 h from 09:58
threading.Timer(HARD_CAP, lambda: os._exit(3)).start()
DEADLINES = {11: T0 + 70 * 60, 12: T0 + 145 * 60}
OUT = os.path.join(HERE, 'RESULTS_2048.json')
logf = open(os.path.join(HERE, 'log_2048.txt'), 'a')


def log(msg):
    line = '[%7.0fs] %s' % (time.time() - T0, msg)
    print(line, flush=True)
    logf.write(line + '\n')
    logf.flush()


def write(res):
    tmp = OUT + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(res, fh, indent=1)
    os.replace(tmp, OUT)


res = dict(implementation='B (independent), d=2048 level 1',
           method='same exact hull + exact-MWPM method as RESULTS.json; oracle '
                  '(exact_split4.py BigLevel5): LP duals from the bipartite-double-cover '
                  'assignment (linear_sum_assignment + SPFA potentials); certified LP '
                  'bounds prune hull triangles; warm HiGHS LP with '
                  'odd-set cuts -> integer duals -> float64 reduced-cost screening with a '
                  'rigorous margin (>100x rounding bound), exact integers for every '
                  'uncleared edge -> exact min-cost perfect matching on surviving edges '
                  '(forced leaves, even cycles, else networkx integer blossom). '
                  'Regression: reproduces N=256/512 matchings of RESULTS.json exactly.',
           started=time.strftime('%Y-%m-%d %H:%M:%S'), cases={})
write(res)
for seed in (11, 12):
    key = 'd2048_seed%d' % seed
    res['cases'][key] = dict(status='RUNNING')
    write(res)
    t = time.time()
    lam = spectrum(2048, seed)
    rec = dict(d=2048, seed=seed, N=2048,
               lam_sha256=hashlib.sha256(np.ascontiguousarray(lam).tobytes()).hexdigest(),
               lam_first=repr(float(lam[0])), lam_last=repr(float(lam[-1])))
    log('seed %d start, sha %s' % (seed, rec['lam_sha256'][:16]))
    lv = BigLevel(lam, log=log)
    out = lv.split(deadline=DEADLINES[seed], progress_every=25)
    oi = out['oracle_info']
    rec.update(Eq=out['Eq_str'], R=out['R_str'], timed_out=out['timed_out'],
               R_upper_bound_certified=out['R_upper_bound'],
               R_lower_bound_certified=out['R_lower_bound'],
               n_open_triangles=out['n_open_triangles'],
               n_oracle_calls=out['n_oracle'], n_chain_points=out['n_chain_points'],
               split_time_s=round(out['time'], 1),
               oracle_max_need_edges=max(c['need'] for c in oi),
               oracle_max_exact_screened_edges=max(c['ncand'] for c in oi),
               oracle_lp_integral_frac=round(float(np.mean([c['lp_integral'] for c in oi])), 3),
               oracle_mean_time_s=round(float(np.mean([c['t'] for c in oi])), 2),
               n_certified_bound_queries=out.get('n_bound_queries'),
               dual_sources={k: sum(1 for c in oi if c.get('dual_source', 'analytic') == k)
                             for k in set(c.get('dual_source', 'analytic') for c in oi)})
    g = np.array([lam[j] - lam[i] for i, j in out['pairs']])
    rec['R_plain_float64'] = repr(float(lv.n - g.sum() ** 2 / (g ** 2).sum()))
    with open(os.path.join(HERE, 'match_d2048_s%d.json' % seed), 'w') as fh:
        json.dump(out['pairs'], fh)
    # post-check: M* is the exact argmin of sum(g^2 - s g) at s = 2 Eq*
    A, B = lv.AB(out['pairs'])
    P, Q = B, 2 * A
    gg = math.gcd(P, Q)
    P //= gg
    Q //= gg
    _, _, _, val = lv.oracle(P, Q)
    rec['post_check_Mstar_exact_argmin_at_s_eq_2Eq'] = (val == P * A - Q * B)
    rec['status'] = ('EXACT (all hull triangles closed)' if not out['timed_out']
                     else 'TIMED OUT: certified bounds only')
    rec['total_time_s'] = round(time.time() - t, 1)
    res['cases'][key] = rec
    write(res)
    log('seed %d done: Eq=%s R=%s status=%s calls=%d %.0fs' % (
        seed, rec['Eq'], rec['R'], rec['status'], rec['n_oracle_calls'], rec['total_time_s']))
res['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
res['wall_s'] = round(time.time() - T0, 1)
write(res)
log('ALL DONE')
os._exit(0)
