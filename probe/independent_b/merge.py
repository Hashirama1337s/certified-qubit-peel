"""Post-checks and merge into RESULTS.json.
For every level: (i) recompute R, Eq from the stored matching in exact rationals,
(ii) re-derive the level spectrum from the previous level's matching (float64
pair means, as specified), (iii) necessary condition: the stored M* is an exact
minimiser of sum(g^2 - s* g) with s* = 2 Eq* (one more exact oracle call)."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import sys
import json
import math
from fractions import Fraction
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exact_split import spectrum, Level, fmt
from sanity import exact_R

CASES = [(256, 31), (256, 32), (256, 33), (512, 21)]


def check_case(d, seed):
    rec = json.load(open(os.path.join(HERE, 'case_d%d_s%d.json' % (d, seed))))
    mt = json.load(open(os.path.join(HERE, 'match_d%d_s%d.json' % (d, seed))))
    cur = spectrum(d, seed)
    for L in rec['levels']:
        pairs = [tuple(p) for p in mt['N%d' % L['N']]]
        assert sorted(v for p in pairs for v in p) == list(range(len(cur)))
        R, Eq = exact_R(cur, pairs)
        ok_val = (fmt(R) == L['R'] and fmt(Eq) == L['Eq'])
        lv = Level(cur, log=lambda *a: None)
        A, B = lv.AB(pairs)
        P, Q = B, 2 * A
        g = math.gcd(P, Q)
        P //= g
        Q //= g
        _, A2, B2, val = lv.oracle(P, Q)
        ok_fix = (val == P * A - Q * B)
        L['post_check'] = dict(exact_recompute_matches=ok_val,
                               Mstar_is_exact_argmin_at_s_eq_2Eq=ok_fix)
        cur = np.array(sorted((cur[i] + cur[j]) / 2 for i, j in pairs))
    return rec


if __name__ == '__main__':
    out = dict(
        implementation='B (independent)',
        method=('exact: lower convex hull of (B,A) over perfect matchings explored with '
                'exact MWPM oracle (HiGHS LP duals -> integer reduced-cost edge '
                'elimination -> networkx blossom in exact integer arithmetic); '
                'branch-and-bound on hull triangles with exact Fractions'),
        notes=['rest spectrum = sorted float64 (lam_i + lam_j)/2 as in the spec',
               'R and Eq are exact rational values for the float64 input levels, '
               'printed to 15 significant digits',
               'final_2level_Eq = float64 difference of the last 2 levels'],
        cases={})
    for d, seed in CASES:
        fn = os.path.join(HERE, 'case_d%d_s%d.json' % (d, seed))
        if not os.path.exists(fn):
            out['cases']['d%d_seed%d' % (d, seed)] = 'NOT FINISHED'
            continue
        out['cases']['d%d_seed%d' % (d, seed)] = check_case(d, seed)
        print(d, seed, [(L['N'], L['post_check']) for L in out['cases']['d%d_seed%d' % (d, seed)]['levels']])
    sfn = os.path.join(HERE, 'sanity_out.json')
    if os.path.exists(sfn):
        s = json.load(open(sfn))
        for k, v in s.items():
            for kk in list(v):
                if isinstance(v[kk], dict):
                    v[kk].pop('pairs', None)
        out['sanity'] = s
    sc = os.path.join(HERE, 'oracle_spotcheck.json')
    if os.path.exists(sc):
        out['oracle_vs_full_blossom_spotcheck'] = json.load(open(sc))
    import glob
    fc = sorted(glob.glob(os.path.join(HERE, 'fullchain_d*_N*.json')))
    out['full_lower_chain_no_pruning_check'] = [json.load(open(f)) for f in fc]
    out['full_lower_chain_note'] = ('complete lower hull chain enumerated without '
                                    'f-pruning; its max-f vertex must equal the pruned '
                                    'result. d256 s31 N=256 full chain did not finish '
                                    'in 20 min (thousands of vertices).')
    with open(os.path.join(HERE, 'RESULTS.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print('wrote RESULTS.json')
