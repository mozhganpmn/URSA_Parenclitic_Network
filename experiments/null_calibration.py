import sys, json, numpy as np
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from parenclitic_core import *
from synthetic_regimes import gen, kinds, M, T
D, N = int(sys.argv[1]), int(sys.argv[2]); pr = Pairs(M); out = {}
facs = {'circular shift, fixed references': None,
        'detrended circular shift, refitted': lambda x: make_shift_factory(x, 1),
        'increment shift, refitted': make_increment_factory}
for kind in kinds:
    rng = np.random.default_rng(20261014); data = [gen(kind, rng) for _ in range(D)]
    for vn, fac in facs.items():
        r2 = np.random.default_rng(7); rates = []
        for x in data:
            res = mc_test_fixed(x, pr, range(5, T), N, r2)[2] if fac is None else mc_test_refit(x, pr, range(5, T), N, r2, fac(x))[2]
            rates.append(np.mean([v['P'] <= 0.05 for v in res.values()]))
        rates = np.array(rates); bs = [rates[r2.integers(0, D, D)].mean() for _ in range(2000)]
        out.setdefault(kind, {})[vn] = dict(rate=round(float(rates.mean()), 3), ci95=[round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)])
        print(f'{kind:>18} | {vn:>44}: {out[kind][vn]}', flush=True)
json.dump(dict(datasets=D, surrogates=N, series=M, years=T, missing=0.10, tested_year_indices=list(range(5, T)), results=out), open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'null_calibration.json'), 'w'), indent=1)
