# Type-I error of the refitted increment-shift test at alpha = 0.05, 0.01 and 0.005 (N = 399 surrogates).
import json, numpy as np
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from parenclitic_core import *
from synthetic_regimes import gen, kinds, M, T
D, N = 200, 399; pr = Pairs(M); out = {}
for kind in kinds:
    rng = np.random.default_rng(20261014); r2 = np.random.default_rng(7); P = []
    for _ in range(D):
        x = gen(kind, rng); res = mc_test_refit(x, pr, range(5, T), N, r2, make_increment_factory(x))[2]
        P.append([v['P'] for v in res.values()])
    P = np.array(P); out[kind] = {str(a): round(float((P <= a).mean()), 4) for a in [0.05, 0.01, 0.005]}
    print(kind, out[kind], flush=True)
json.dump(dict(datasets=D, surrogates=N, results=out), open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'null_calibration_levels.json'), 'w'), indent=1)
