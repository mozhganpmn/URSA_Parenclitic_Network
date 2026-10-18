# Pair-level detection power for the first year of a relational break that persists for K years.
# LFO rules see only the n pre-break years. The whole-record reference (classic parenclitic) also contains the
# target year and the K-1 later break years, as when a reference is estimated from the full series.
import numpy as np, scipy.stats as st, json, warnings
warnings.filterwarnings('ignore')
from sklearn.covariance import MinCovDet
P = 2; RHO = 0.6; NS = [3, 4, 5, 7, 10, 14]; DELTAS = [0, 1, 2, 3, 4, 6]; K = 3
CHEAP, ROBUST = 60_000, 600; rng = np.random.default_rng(20261015)
L = np.linalg.cholesky(np.array([[1, RHO], [RHO, 1.0]])); u = np.array([1.0, -1.0])/np.sqrt(2)
CHI2 = st.chi2.ppf(0.95, P)
def Fsq(n): return P*(n+1)*(n-1)/(n*(n-P))*st.f.ppf(0.95, P, n-P)
def Bsq(m): return (m-1)**2/m*st.beta.ppf(0.95, P/2, (m-P-1)/2)   # in-sample law for m points including the target
def maha_batch(tr, x):
    mu = tr.mean(1); d = tr - mu[:, None, :]; S = np.einsum('mij,mik->mjk', d, d)/(tr.shape[1]-1)
    det = S[:, 0, 0]*S[:, 1, 1] - S[:, 0, 1]**2
    Si = np.stack([np.stack([S[:, 1, 1], -S[:, 0, 1]], -1), np.stack([-S[:, 1, 0], S[:, 0, 0]], -1)], 1)/det[:, None, None]
    dx = x - mu; return np.einsum('mi,mij,mj->m', dx, Si, dx)
def arw_cn(Xtr, m0, c0inv, alpha=0.025):
    n, p = Xtr.shape; diff = Xtr - m0; d2 = np.einsum('ij,jk,ik->i', diff, c0inv, diff)
    d2o = np.sort(d2); delta = st.chi2.ppf(1-alpha, p); pcrit = (0.24 - 0.003*p)/np.sqrt(n)
    dif = st.chi2.cdf(d2o, p) - (np.arange(n)+0.5)/n; mask = (d2o >= delta) & (dif > 0)
    alfan = float(np.max(dif[mask])) if mask.any() else 0.0
    if alfan < pcrit: alfan = 0.0
    return (max(d2o[n - int(np.ceil(n*alfan)) - 1], delta) if alfan > 0 else np.inf)
out = {'design': dict(p=P, rho=RHO, persist_years=K, shift='delta*(1,-1)/sqrt(2), marginal SD units',
                      closed_form_trials=CHEAP, robust_trials=ROBUST), 'results': []}
for n in NS:
    stat = {}
    for dl in DELTAS:
        Z = rng.standard_normal((CHEAP, n+K, P)) @ L.T; Z[:, n:, :] += dl*u; x = Z[:, n, :]
        stat[dl] = dict(lfo=maha_batch(Z[:, :n, :], x), whole=maha_batch(Z, x))
        dm = []; fz = []
        for k in range(ROBUST):
            tr = Z[k, :n, :]
            try:
                mc = MinCovDet(random_state=0).fit(tr); ci = np.linalg.inv(mc.covariance_); dx = x[k] - mc.location_
                d2 = float(dx @ ci @ dx); dm.append(d2); fz.append(d2 >= arw_cn(tr, mc.location_, ci))
            except Exception: dm.append(np.nan); fz.append(False)
        stat[dl]['mcd'] = np.array(dm); stat[dl]['arw'] = np.array(fz)
    q = {k: np.nanquantile(stat[0][k], 0.95) for k in ['lfo', 'whole', 'mcd']}
    for dl in DELTAS:
        s = stat[dl]
        rec = dict(n=n, delta=dl, F=float(np.mean(s['lfo'] >= Fsq(n))), chi2=float(np.mean(s['lfo'] >= CHI2)),
                   whole_chi2=float(np.mean(s['whole'] >= CHI2)), whole_beta=float(np.mean(s['whole'] >= Bsq(n+K))),
                   mcd_chi2=float(np.nanmean(s['mcd'] >= CHI2)), arw=float(np.mean(s['arw'])),
                   sa_lfo=float(np.mean(s['lfo'] >= q['lfo'])), sa_whole=float(np.mean(s['whole'] >= q['whole'])),
                   sa_mcd=float(np.nanmean(s['mcd'] >= q['mcd'])))
        out['results'].append(rec); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()}, flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'power_pair.json'), 'w'), indent=1)
