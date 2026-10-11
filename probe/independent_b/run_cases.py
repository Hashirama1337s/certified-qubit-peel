"""Run one case: python run_cases.py d seed [levels]
Writes case_d{d}_s{seed}.json (numbers) and match_d{d}_s{seed}.json (matchings)."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import sys
import json
import time
import hashlib
from fractions import Fraction
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exact_split import spectrum, Level, fmt
from sanity import brute_all, dyck_all, exact_R


def main(d, seed, max_levels=None):
    t_all = time.time()
    lam = spectrum(d, seed)
    rec = dict(d=d, seed=seed,
               lam_sha256=hashlib.sha256(np.ascontiguousarray(lam).tobytes()).hexdigest(),
               lam_first=repr(float(lam[0])), lam_last=repr(float(lam[-1])),
               levels=[])
    matches = {}
    cur = lam
    logf = open(os.path.join(HERE, 'log_d%d_s%d.txt' % (d, seed)), 'w')

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + '\n')
        logf.flush()

    while len(cur) > 2:
        lv = Level(cur, log=log)
        out = lv.split()
        L = dict(N=out['N'], n=out['n'], Eq=out['Eq_str'], R=out['R_str'],
                 Eq_float=float(out['Eq']), R_float_recomputed=out['R_float'],
                 n_oracle_calls=out['n_oracle'], n_chain_points=out['n_chain_points'],
                 time_s=round(out['time'], 2),
                 max_need_edges=max(c['need'] for c in out['oracle_info']),
                 lp_integral_frac=round(np.mean([c['lp_integral'] for c in out['oracle_info']]), 3))
        # independent checks on small levels
        if out['N'] <= 16:
            p, cnt, nc = brute_all(cur)
            R2, E2 = exact_R(cur, p)
            L['check'] = dict(method='all %d matchings' % cnt, R=fmt(R2), Eq=fmt(E2),
                              same_matching=(sorted(p) == out['pairs']))
        elif out['N'] == 32:
            p, cnt, nc = dyck_all(cur)
            R2, E2 = exact_R(cur, p)
            L['check'] = dict(method='all %d non-nesting matchings' % cnt, R=fmt(R2),
                              Eq=fmt(E2), same_matching=(sorted(p) == out['pairs']))
        log('N=%4d  Eq=%s  R*=%s  oracle=%d  t=%.1fs %s' % (
            out['N'], out['Eq_str'], out['R_str'], out['n_oracle'], out['time'],
            ('check:' + str(L['check']['same_matching'])) if 'check' in L else ''))
        rec['levels'].append(L)
        matches['N%d' % out['N']] = out['pairs']
        cur = out['rest']
        if max_levels is not None and len(rec['levels']) >= max_levels:
            break
    if len(cur) == 2:
        rec['final_2level_Eq'] = repr(float(cur[1] - cur[0]))
        rec['final_2level_Eq_exact'] = fmt(Fraction(float(cur[1])) - Fraction(float(cur[0])))
    rec['total_time_s'] = round(time.time() - t_all, 1)
    with open(os.path.join(HERE, 'case_d%d_s%d.json' % (d, seed)), 'w') as fh:
        json.dump(rec, fh, indent=1)
    with open(os.path.join(HERE, 'match_d%d_s%d.json' % (d, seed)), 'w') as fh:
        json.dump(matches, fh)
    log('total %.1fs' % rec['total_time_s'])


if __name__ == '__main__':
    d = int(sys.argv[1])
    seed = int(sys.argv[2])
    ml = int(sys.argv[3]) if len(sys.argv) > 3 else None
    main(d, seed, ml)
