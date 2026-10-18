"""Vectorised leave-future-out parenclitic NAD, refitted surrogate nulls and hierarchical scores.

Used by the notebook (network-level test, robustness checks) and by the scripts in experiments/.
The rules match the notebook: LFO references from co-observed earlier years, n >= 3, CV < 1% exclusion,
zero-variance guard, correlation clip 0.9999, F(2, n-2) activation thresholds, 20-sigma cap in aggregation only."""
import numpy as np, pandas as pd
from scipy import stats
P_DIM = 2
def fthr(n, alpha=0.05):
    n = np.asarray(n, float); out = np.full(n.shape, np.nan); ok = n >= 3
    k = n[ok]; out[ok] = np.sqrt(P_DIM*(k+1)*(k-1)/(k*(k-P_DIM))*stats.f.ppf(1-alpha, P_DIM, k-P_DIM)); return out
THR = fthr(np.arange(0, 60))
class Pairs:
    def __init__(self, M):
        self.M = M; self.ia, self.ib = np.triu_indices(M, 1)
def lfo_stats(X, pr, clip=0.9999, drop_above=None, cv=0.01):
    """For all pairs and target years t, statistics of co-observed years < t. Returns dict of arrays (pairs x T)."""
    cen = np.array([np.nanmean(r) if np.isfinite(r).any() else 0.0 for r in X])      # centring keeps sums precise
    ca = cen[pr.ia][:, None]; cb = cen[pr.ib][:, None]
    A = X[pr.ia]; B = X[pr.ib]; m = np.isfinite(A) & np.isfinite(B)
    a = np.where(m, A - ca, 0.0); b = np.where(m, B - cb, 0.0); mf = m.astype(float)
    def cs(z):  # exclusive cumulative sum along years: sum over tau < t
        c = np.cumsum(z, axis=1); return np.concatenate([np.zeros((z.shape[0], 1)), c[:, :-1]], axis=1)
    n = cs(mf); Sa = cs(a); Sb = cs(b); Saa = cs(a*a); Sbb = cs(b*b); Sab = cs(a*b)
    with np.errstate(invalid='ignore', divide='ignore'):
        mac = Sa/n; mbc = Sb/n
        va = (Saa - n*mac*mac)/(n-1); vb = (Sbb - n*mbc*mbc)/(n-1); cab = (Sab - n*mac*mbc)/(n-1)
        sa = np.sqrt(np.clip(va, 0, None)); sb = np.sqrt(np.clip(vb, 0, None))
        ma = mac + ca; mb = mbc + cb
        r_raw = cab/(sa*sb); r = np.clip(r_raw, -clip, clip)
        ok = (n >= 3) & (sa > 1e-10) & (sb > 1e-10) & (sa > 1e-7*np.abs(mac)) & (sb > 1e-7*np.abs(mbc))   # second pair: constant windows under rounding
        if drop_above is not None: ok &= ~(np.abs(r_raw) > drop_above)
        ok &= ~((np.abs(ma) > 1e-8) & (sa/np.abs(ma) < cv)) & ~((np.abs(mb) > 1e-8) & (sb/np.abs(mb) < cv))
    th = THR[np.clip(n.astype(int), 0, len(THR)-1)]
    return dict(n=n, ma=ma, mb=mb, sa=sa, sb=sb, r=r, ok=ok, th=th, m=m, A=A, B=B)
def weights(S, A, B):
    with np.errstate(invalid='ignore', divide='ignore'):
        za = (A - S['ma'])/S['sa']; zb = (B - S['mb'])/S['sb']; r = S['r']
        return np.sqrt(np.clip((za*za - 2*r*za*zb + zb*zb)/(1 - r*r), 0, None))
def nad_all(X, pr, S=None, clip=0.9999, drop_above=None, cv=0.01):
    """NAD for every year. If S is given, X is scored against the fixed references S."""
    if S is None: S = lfo_stats(X, pr, clip, drop_above, cv)
    A = X[pr.ia]; B = X[pr.ib]; obs = np.isfinite(A) & np.isfinite(B)
    w = weights(S, A, B); elig = S['ok'] & obs
    act = elig & (w >= S['th'])
    ne = elig.sum(0); return np.where(ne > 0, act.sum(0)/np.maximum(ne, 1), np.nan), ne
# ---------------- surrogates ----------------
def _trend(t, y, deg):
    if deg is None or len(t) < deg + 3: return np.full(len(t), y.mean()) if deg is not None else np.zeros(len(t))
    c = np.polyfit(t, y, deg); return np.polyval(c, t)
def make_shift_factory(X, deg):
    """Circular-shift surrogates over the observed years of each series. deg=None shifts the values; deg=1 shifts the
    residuals of a linear trend and adds the trend back (detrended circular shift)."""
    info = []
    for i in range(X.shape[0]):
        o = np.where(np.isfinite(X[i]))[0]; y = X[i, o]
        if deg is None: tr = np.zeros(len(o)); res = y.copy()
        else: tr = _trend(o.astype(float), y, deg); res = y - tr
        info.append((o, tr, res))
    def draw(rng):
        Z = np.full(X.shape, np.nan)
        for i, (o, tr, res) in enumerate(info):
            if len(o) == 0: continue
            s = int(rng.integers(0, len(o))); Z[i, o] = tr + np.roll(res, s)
        return Z
    return draw
def load_input(path, min_obs=3):
    x = pd.read_excel(path, sheet_name=0)
    yc = sorted([c for c in x.columns if isinstance(c, (int, np.integer))])
    V = x[yc].apply(pd.to_numeric, errors='coerce').values.astype(float)
    keep = np.isfinite(V).sum(1) >= min_obs
    return x[keep].reset_index(drop=True), V[keep], yc
def mc_test_fixed(X, pr, years_idx, N, rng):
    """Fixed-reference circular shift: the values of each series are shifted circularly and scored against the
    LFO references of the observed data, which are not refitted. Used only as a comparison design."""
    obs, ne = nad_all(X, pr)
    draw = make_shift_factory(X, None); Sfix = lfo_stats(X, pr)
    null = np.array([nad_all(draw(rng), pr, S=Sfix)[0] for _ in range(N)])
    res = {}
    for j in years_idx:
        nj = null[:, j]; gt = int((nj > obs[j] + 1e-12).sum()); eq = int((np.abs(nj - obs[j]) <= 1e-12).sum())
        u = rng.random()
        res[j] = dict(NAD=float(obs[j]), mean=float(np.nanmean(nj)), sd=float(np.nanstd(nj, ddof=1)),
                      P=(gt + u*eq + 1)/(N + 1), P_conservative=(gt + eq + 1)/(N + 1),
                      pct=float((nj < obs[j]).mean()*100), eligible=int(ne[j]))
    return obs, null, res
def mc_test_refit(X, pr, years_idx, N, rng, draw):
    """Monte Carlo test with surrogates from draw(rng); every LFO reference and threshold is refitted on each
    surrogate. Ties are broken at random (mid-P), which matters for small synthetic networks."""
    obs, ne = nad_all(X, pr); null = np.array([nad_all(draw(rng), pr)[0] for _ in range(N)]); res = {}
    for j in years_idx:
        nj = null[:, j]; gt = int((nj > obs[j] + 1e-12).sum()); eq = int((np.abs(nj - obs[j]) <= 1e-12).sum())
        res[j] = dict(NAD=float(obs[j]), mean=float(nj.mean()), sd=float(nj.std(ddof=1)), P=(gt + rng.random()*eq + 1)/(N + 1),
                      pct=float((nj < obs[j]).mean()*100), eligible=int(ne[j]))
    return obs, null, res
def make_increment_factory(X):
    """Increment-shift surrogate: circularly shift the year-to-year increments of each series over its observed
    years and cumulate them from the first observed value. Each surrogate keeps the start value, the end value
    (hence the mean trend), the marginal distribution of increments and most of their serial structure."""
    info = []
    for i in range(X.shape[0]):
        o = np.where(np.isfinite(X[i]))[0]; y = X[i, o]; info.append((o, y))
    def draw(rng):
        Z = np.full(X.shape, np.nan)
        for i, (o, y) in enumerate(info):
            if len(o) < 3: Z[i, o] = y; continue
            d = np.diff(y); d = np.roll(d, int(rng.integers(0, len(d))))
            Z[i, o] = y[0] + np.concatenate([[0.0], np.cumsum(d)])
        return Z
    return draw
# ---------------- hierarchical scores (vectorised; same rules as the notebook) ----------------
CAP, LAM = 20.0, 5.0
def scores(X, pr, ind_of, cov_w, dom_of_ind, std=False, cv=0.01):
    """Node, indicator and domain anomaly scores for every year. ind_of: indicator name per row of X;
    cov_w: dict indicator -> coverage weight; dom_of_ind: dict indicator -> domain.
    std=False: edge weights in standard-deviation units (capped at 20). std=True: the capped weight is divided by the
    pair-year activation threshold w*(n, 0.05), so that a score of 1 is the activation threshold at any training window."""
    S = lfo_stats(X, pr, cv=cv); A = X[pr.ia]; B = X[pr.ib]; obs = np.isfinite(A) & np.isfinite(B)
    elig = S['ok'] & obs; w = np.minimum(weights(S, A, B), CAP)
    if std: w = w/S['th']
    C = np.where(elig, 1 - np.exp(-(S['n'] - 3)/LAM), 0.0)
    cw = np.where(elig, C*np.nan_to_num(w), 0.0); M, T = X.shape
    num = np.zeros((M, T)); den = np.zeros((M, T))
    np.add.at(num, pr.ia, cw); np.add.at(num, pr.ib, cw); np.add.at(den, pr.ia, C); np.add.at(den, pr.ib, C)
    node = np.where(den > 0, num/np.maximum(den, 1e-300), np.nan)
    inds = sorted(set(ind_of)); ii = {k: n for n, k in enumerate(inds)}; ia = np.array([ii[ind_of[i]] for i in pr.ia]); ib = np.array([ii[ind_of[i]] for i in pr.ib])
    inum = np.zeros((len(inds), T)); iden = np.zeros((len(inds), T))
    np.add.at(inum, ia, cw); np.add.at(iden, ia, C); same = ia == ib
    np.add.at(inum, ib[~same], cw[~same]); np.add.at(iden, ib[~same], C[~same])
    ind = np.where(iden > 0, inum/np.maximum(iden, 1e-300), np.nan)
    doms = sorted(set(dom_of_ind[k] for k in inds)); dom = np.full((len(doms), T), np.nan)
    for d_i, d in enumerate(doms):
        ks = [ii[k] for k in inds if dom_of_ind[k] == d]; wv = np.array([cov_w[inds[k]] for k in ks])[:, None]
        v = ind[ks]; ok = np.isfinite(v); tw = (wv*ok).sum(0)
        dom[d_i] = np.where(tw > 0, (np.where(ok, v, 0)*wv).sum(0)/np.maximum(tw, 1e-300), np.nan)
    return node, (inds, ind), (doms, dom)
class NadEngine:
    """Fast NAD for surrogates that keep the observed missingness pattern of X (true for every surrogate here):
    co-observation masks, training counts and thresholds are computed once."""
    def __init__(self, X, pr, clip=0.9999, drop_above=None, cv=0.01, dtype=np.float64):
        self.pr = pr; self.clip = clip; self.drop = drop_above; self.cv = cv
        A = X[pr.ia]; B = X[pr.ib]; m = np.isfinite(A) & np.isfinite(B); self.m = m
        mf = m.astype(dtype); c = np.cumsum(mf, 1); self.n = np.concatenate([np.zeros((m.shape[0], 1), dtype), c[:, :-1]], 1)
        self.th = THR[np.clip(self.n.astype(int), 0, len(THR)-1)]; self.base_ok = (self.n >= 3) & m; self.dtype = dtype
    def nad(self, X):
        pr = self.pr; m = self.m; n = self.n
        cen = np.array([np.nanmean(r) if np.isfinite(r).any() else 0.0 for r in X]); ca = cen[pr.ia][:, None]; cb = cen[pr.ib][:, None]
        A = np.where(m, X[pr.ia] - ca, 0.0).astype(self.dtype); B = np.where(m, X[pr.ib] - cb, 0.0).astype(self.dtype)
        def cs(z): c = np.cumsum(z, 1); return np.concatenate([np.zeros((z.shape[0], 1), z.dtype), c[:, :-1]], 1)
        Sa = cs(A); Sb = cs(B); Saa = cs(A*A); Sbb = cs(B*B); Sab = cs(A*B)
        with np.errstate(invalid='ignore', divide='ignore'):
            mac = Sa/n; mbc = Sb/n; va = (Saa - n*mac*mac)/(n-1); vb = (Sbb - n*mbc*mbc)/(n-1); cab = (Sab - n*mac*mbc)/(n-1)
            sa = np.sqrt(np.clip(va, 0, None)); sb = np.sqrt(np.clip(vb, 0, None)); r_raw = cab/(sa*sb); r = np.clip(r_raw, -self.clip, self.clip)
            ma = mac + ca; mb = mbc + cb
            ok = self.base_ok & (sa > 1e-10) & (sb > 1e-10) & (sa > 1e-7*np.abs(mac)) & (sb > 1e-7*np.abs(mbc))
            ok &= ~((np.abs(ma) > 1e-8) & (sa/np.abs(ma) < self.cv)) & ~((np.abs(mb) > 1e-8) & (sb/np.abs(mb) < self.cv))
            if self.drop is not None: ok &= ~(np.abs(r_raw) > self.drop)
            za = (A - mac)/sa; zb = (B - mbc)/sb
            w2 = (za*za - 2*r*za*zb + zb*zb)/(1 - r*r)
        act = ok & (w2 >= self.th*self.th); ne = ok.sum(0)
        return np.where(ne > 0, act.sum(0)/np.maximum(ne, 1), np.nan), ne
