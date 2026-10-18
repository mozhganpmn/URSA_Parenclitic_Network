import json, numpy as np
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from parenclitic_core import *
from synthetic_regimes import gen, M, T
pr = Pairs(M); out = {}
for kind in ['common factor', 'random walk', 'AR(1)', 'independent trends']:
    rng = np.random.default_rng(20261014); r2 = np.random.default_rng(7); zs = []
    for _ in range(150):
        x = gen(kind, rng); res = mc_test_refit(x, pr, range(5, T), 199, r2, make_increment_factory(x))[2]
        zs += [(v['NAD'] - v['mean'])/v['sd'] for v in res.values() if v['sd'] > 0]
    zs = np.array(zs); out[kind] = dict(q95=round(float(np.quantile(zs, 0.95)), 2), q995=round(float(np.quantile(zs, 0.995)), 2), max=round(float(zs.max()), 2), n=int(zs.size))
    print(kind, out[kind], flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'null_z_quantiles.json'), 'w'), indent=1)
