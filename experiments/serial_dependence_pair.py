"""Run from the repository root: python experiments/serial_dependence_pair.py [OUT.json]
Held-out per-pair false-positive rate of the F threshold when the two series are AR(1).
Design follows Supplementary Table S19: bivariate normal innovations with rho = 0.6, n training years, one held-out
year that directly follows the training window, F-corrected threshold of main-text eq. (4), alpha = 0.05."""
import numpy as np, scipy.stats as st, json, sys, os
P, RHO = 2, 0.6
NS = [3, 4, 5, 7, 10, 14]; PHIS = [0.0, 0.3, 0.5, 0.7]; TRIALS = 200_000
rng = np.random.default_rng(20261026)
L = np.linalg.cholesky(np.array([[1, RHO], [RHO, 1.0]]))
def fsq(n): return P*(n+1)*(n-1)/(n*(n-P))*st.f.ppf(0.95, P, n-P)
def maha2(tr, x):
    # same unclipped bivariate Mahalanobis distance as power_pair.maha_batch
    mu = tr.mean(1); d = tr - mu[:, None, :]; S = np.einsum('mij,mik->mjk', d, d)/(tr.shape[1]-1)
    det = S[:, 0, 0]*S[:, 1, 1] - S[:, 0, 1]**2
    dx = x - mu
    return (S[:, 1, 1]*dx[:, 0]**2 - 2*S[:, 0, 1]*dx[:, 0]*dx[:, 1] + S[:, 0, 0]*dx[:, 1]**2)/det
def gen(n, phi):
    T = n + 1; e = rng.standard_normal((TRIALS, T, P)) @ L.T; x = np.empty_like(e)
    x[:, 0] = e[:, 0]
    for t in range(1, T): x[:, t] = phi*x[:, t-1] + np.sqrt(1-phi**2)*e[:, t]
    return x
res = []
for phi in PHIS:
    for n in NS:
        x = gen(n, phi); w2 = maha2(x[:, :n], x[:, n])
        rate = float(np.mean(w2 >= fsq(n)**1)); se = float(np.sqrt(rate*(1-rate)/TRIALS))
        res.append(dict(phi=phi, n=n, rate=round(rate, 4), se=round(se, 4)))
        print(res[-1], flush=True)
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join('experiments', 'serial_dependence_pair.json')
json.dump(dict(design=dict(rho=RHO, trials=TRIALS, alpha=0.05, threshold='F(2,n-2) scaled, eq. 4', seed=20261026), results=res),
          open(OUT, 'w'), indent=1)
