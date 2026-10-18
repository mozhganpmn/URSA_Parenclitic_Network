"""Checks of variants.py. Development seeds only (below 1000); nothing here feeds a reported number except the
exactness checks that are quoted as a property of the activation thresholds.

1. M-reference statistics equal the existing engines (NadEngine, benchmark_monitors) to numerical precision.
2. Exactness of the pair thresholds: under independent bivariate-normal sampling with series-specific linear trends and
   no missing values, the share of pairs above the threshold is 5% at every training window n for the T reference
   (and, without trends, for the M reference). The same for the univariate limits.
3. Mean of the continuous score is close to 1 under the null (p uniform, truncation at 1e-6 costs about 1e-5).
"""
import sys, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..')); sys.path.insert(0, HERE)
from parenclitic_core import Pairs, NadEngine
import benchmark_monitors as bm
from synthetic_regimes import gen, M, T
import variants as V

def same(a, b):
    both = np.isfinite(a) & np.isfinite(b); return bool((np.isfinite(a) == np.isfinite(b)).all() and np.allclose(a[both], b[both], rtol=1e-9, atol=1e-12))

def test_equivalence():
    pr = Pairs(M); ok = True
    for kind in ['iid', 'AR(1)', 'independent trends', 'saturating']:
        X = gen(kind, np.random.default_rng(11)); X[2, :5] = np.nan
        s = V.AllStats(X, pr)(X); nad = NadEngine(X, pr).nad(X)[0]; ued, dt2 = bm.ued_dt2(X); spe = bm.spe_all(X)
        r = [same(s['NAD-M'], nad), same(s['UED-M'], ued), same(s['dT2-M'], dt2), same(s['SPE-M'], spe)]
        print('equivalence', kind, r); ok &= all(r)
    return ok

def test_exactness(D=300):
    rng = np.random.default_rng(31); pr = Pairs(M); res = {k: [[] for _ in range(T)] for k in ['NAD-M_trend', 'NAD-T_trend', 'NAD-M_flat', 'NAD-T_flat', 'CON-T_trend', 'CON-M_flat', 'UED-T_trend', 'UED-M_flat']}
    for _ in range(D):
        e = rng.normal(size=(M, T)); sl = rng.normal(0, 0.6, M)[:, None]*np.arange(T)
        for name, X in [('trend', e + sl + 10.0), ('flat', e + 10.0)]:
            s = V.AllStats(X, pr)(X)
            for k in res:
                if k.endswith(name): res[k][0:0]; [res[k][j].append(s[k.split('_')[0]][j]) for j in range(T)]
    print('mean over datasets by training window n = year index j (expected 0.05 for NAD and UED, 1.0 for CON)')
    for k, v in res.items(): print(f'{k:12s}', ' '.join(f'{np.nanmean(v[j]):.3f}' for j in range(3, T)))
    return res

if __name__ == '__main__':
    print('equivalence all ok:', test_equivalence()); test_exactness()
