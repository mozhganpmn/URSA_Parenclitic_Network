# Network-level detection power of the Monte Carlo NAD test (refitted increment-shift null) for a one-year shock
# applied to a fraction of the series, against the fixed-reference circular-shift design.
import sys, json, numpy as np
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from parenclitic_core import *
M, T, J = 30, 14, 10
def gen(rng, frac, delta):
    tt = np.arange(T); e = rng.normal(size=(M, T)); ar = e.copy()
    for t in range(1, T): ar[:, t] = 0.5*ar[:, t-1] + np.sqrt(0.75)*e[:, t]
    x = ar + rng.normal(0, 0.3, M)[:, None]*tt                  # AR(1) noise around independent linear trends
    k = int(round(frac*M)); idx = rng.choice(M, k, replace=False)
    x[idx, J] += delta*rng.choice([-1, 1], k)                    # one-year shock, random signs, innovation-SD units
    x = x + 10.0; x[rng.random((M, T)) < 0.10] = np.nan; return x
D, N = int(sys.argv[1]), int(sys.argv[2]); pr = Pairs(M); out = []
for frac in [0.1, 0.2, 0.3, 0.5]:
    for delta in [0, 1, 2, 3, 4]:
        if delta == 0 and frac > 0.1: continue
        rng = np.random.default_rng(int(1000*frac) + delta); data = [gen(rng, frac, delta) for _ in range(D)]
        rec = dict(frac=frac, delta=delta)
        r2 = np.random.default_rng(9); rec['fixed_reference'] = round(float(np.mean([mc_test_fixed(x, pr, [J], N, r2)[2][J]['P'] <= 0.05 for x in data])), 3)
        r2 = np.random.default_rng(9); rec['increment_refit'] = round(float(np.mean([mc_test_refit(x, pr, [J], N, r2, make_increment_factory(x))[2][J]['P'] <= 0.05 for x in data])), 3)
        out.append(rec); print(rec, flush=True)
json.dump(dict(datasets=D, surrogates=N, series=M, years=T, target_index=J, base='AR(1) phi=0.5 around independent linear trends, 10% missing', results=out), open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'power_network.json'), 'w'), indent=1)
