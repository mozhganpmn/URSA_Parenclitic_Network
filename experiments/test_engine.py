# Check that the vectorised engine reproduces a direct per-pair implementation exactly (observed data and surrogates).
import numpy as np, scipy.stats as st, sys
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from parenclitic_core import *
_D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
x, V, yc = load_input(os.path.join(_D, 'Input.xlsx')); pr = Pairs(V.shape[0])
def thr(n): return np.sqrt(2*(n+1)*(n-1)/(n*(n-2))*st.f.ppf(0.95, 2, n-2))
def slow(V, cv):
    T = V.shape[1]; act = np.zeros(T); el = np.zeros(T)
    for a, b in zip(pr.ia, pr.ib):
        xa, xb = V[a], V[b]; both = np.isfinite(xa) & np.isfinite(xb)
        for t in range(T):
            if not both[t]: continue
            tr = np.where(both[:t])[0]
            if len(tr) < 3: continue
            ta, tb = xa[tr], xb[tr]; mu = np.array([ta.mean(), tb.mean()]); si = ta.std(ddof=1); sj = tb.std(ddof=1)
            if si < 1e-10 or sj < 1e-10: continue
            if (abs(mu[0]) > 1e-8 and si/abs(mu[0]) < cv) or (abs(mu[1]) > 1e-8 and sj/abs(mu[1]) < cv): continue
            rho = float(np.clip(np.corrcoef(ta, tb)[0, 1], -0.9999, 0.9999)); C = np.array([[si*si, rho*si*sj], [rho*si*sj, sj*sj]])
            d = np.array([xa[t]-mu[0], xb[t]-mu[1]]); w = np.sqrt(max(float(d @ np.linalg.inv(C) @ d), 0))
            el[t] += 1; act[t] += w >= thr(len(tr))
    return np.where(el > 0, act/np.maximum(el, 1), np.nan), el
for cv in [0.0, 0.01]:
    s, e = slow(V, cv); f, ef = NadEngine(V, pr, cv=cv).nad(V); g, eg = nad_all(V, pr, cv=cv)
    k = e > 0
    print(cv, 'eligible equal', np.array_equal(e, ef), np.array_equal(e, eg), 'max|NAD diff| engine', np.nanmax(np.abs(s[k]-f[k])), 'nad_all', np.nanmax(np.abs(s[k]-g[k])))
rng = np.random.default_rng(3); d = make_increment_factory(V)
for k in range(2):
    Z = d(rng); s, e = slow(Z, 0.01); f, ef = NadEngine(V, pr).nad(Z)
    print('surrogate', k, np.array_equal(e, ef), np.nanmax(np.abs(s[e > 0]-f[e > 0])))
x2, V2, _ = load_input(os.path.join(_D, 'Input_Calgary.xlsx')); pr = Pairs(V2.shape[0])
for cv in [0.0, 0.01]:
    s, e = slow(V2, cv); f, ef = NadEngine(V2, pr, cv=cv).nad(V2); print('Calgary', cv, np.array_equal(e, ef), np.nanmax(np.abs(s[e > 0]-f[e > 0])))
