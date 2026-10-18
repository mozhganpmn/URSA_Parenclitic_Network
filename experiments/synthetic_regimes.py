"""Synthetic null regimes for the network-level calibration (30 series, 14 years, 10% missing)."""
import numpy as np
M, T = 30, 14
kinds = ['iid', 'AR(1)', 'independent trends', 'common trend', 'saturating', 'common factor', 'random walk']
def gen(kind, rng, miss=0.10):
    tt = np.arange(T); e = rng.normal(size=(M, T)); ar = e.copy()
    for t in range(1, T): ar[:, t] = 0.5*ar[:, t-1] + np.sqrt(0.75)*e[:, t]
    if kind == 'iid': x = e
    elif kind == 'AR(1)': x = ar
    elif kind == 'independent trends': x = ar + rng.normal(0, 0.6, M)[:, None]*tt
    elif kind == 'common trend': x = ar + 0.5*tt[None, :]*rng.choice([-1, 1], M)[:, None]
    elif kind == 'saturating': x = ar + rng.uniform(1, 4, M)[:, None]*(1 - np.exp(-tt/rng.uniform(2, 6, M)[:, None]))
    elif kind == 'common factor':
        f = np.zeros(T); u = rng.normal(size=T)
        for t in range(T): f[t] = (0.5*f[t-1] if t else 0) + u[t]
        x = rng.uniform(0.5, 1.5, M)[:, None]*f[None, :]*rng.choice([-1, 1], M)[:, None] + ar
    elif kind == 'random walk': x = np.cumsum(e, axis=1)
    x = x + 10.0; x[rng.random((M, T)) < miss] = np.nan
    return x
