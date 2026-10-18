"""Engine options studied in experiments/PLAN_improvements.md.

Two references for the held-out year (all prospective, leave-future-out, same eligibility rules as parenclitic_core):
  M  mean reference (the current method)
  T  linear-trend reference: each series is regressed on calendar year over the earlier co-observed years; the
     prediction error is scaled with the prediction leverage h = 1/n + (t0 - tbar)^2 / Sxx, and the pair thresholds use
     the exact law of the Hotelling T^2 of a prediction error, 2(n-2)/(n-3) F(2, n-3) (n >= 4).
Three pair-level aggregates per reference:
  NAD   share of eligible pairs above the 95% threshold (the current network anomaly density)
  CON   mean of -ln(max(p, 1e-6)) over eligible pairs, p the upper tail of the pair F law
  REL   relationship channel: mean of -ln(max(p, 1e-6)) of the minor-axis part of the pair Mahalanobis distance over
        pairs with training |r| >= 0.7 (undefined with fewer than 5 such pairs)
Univariate and PCA baselines get the same two references (UED, dT2, SPE).
"""
import numpy as np
from scipy import stats
from scipy.special import fdtrc
from parenclitic_core import THR

CAPV = 20.0; CV = 0.01; PCLIP = 1e-6; RMIN = 0.7; MIN_REL = 5
NT = 80
FQ_T = np.full(NT, np.nan); FQ_T[4:] = stats.f.ppf(0.95, 2, np.arange(4, NT) - 3)            # F(2, n-3) 95% point
TQ_M = np.full(NT, np.nan); TQ_M[3:] = stats.t.ppf(0.975, np.arange(3, NT) - 1)             # t(n-1) 97.5% point
TQ_T = np.full(NT, np.nan); TQ_T[4:] = stats.t.ppf(0.975, np.arange(4, NT) - 2)             # t(n-2) 97.5% point
NAMES_NAD = ['NAD-M', 'CON-M', 'REL-M', 'NAD-T', 'CON-T', 'REL-T']
NAMES_UNI = ['UED-M', 'dT2-M', 'SPE-M', 'UED-T', 'dT2-T', 'SPE-T']

def _cs(z):
    c = np.cumsum(z, axis=1); return np.concatenate([np.zeros((z.shape[0], 1), z.dtype), c[:, :-1]], axis=1)
def _cen(X): return np.array([np.nanmean(r) if np.isfinite(r).any() else 0.0 for r in X])
def _idx(n): return np.clip(np.nan_to_num(n, nan=0).astype(int), 0, NT - 1)
def _share(act, ok): ne = ok.sum(0); return np.where(ne > 0, act.sum(0)/np.maximum(ne, 1), np.nan)
def _mean(u, ok): ne = ok.sum(0); return np.where(ne > 0, np.where(ok, u, 0.0).sum(0)/np.maximum(ne, 1), np.nan)
def _nlp(p): return -np.log(np.maximum(p, PCLIP))

class VariantEngine:
    """Pair statistics for all six NAD-family variants. The co-observation pattern of X is cached, so surrogates that
    keep the missingness pattern (all of ours) are cheap."""
    def __init__(self, X, pr, times=None, cv=CV, clip=0.9999, rmin=RMIN, min_rel=MIN_REL):
        self.pr = pr; self.cv = cv; self.clip = clip; self.rmin = rmin; self.min_rel = min_rel
        T = X.shape[1]; t = np.arange(T, dtype=float) if times is None else np.asarray(times, float); self.t = (t - t.mean())[None, :]
        m = np.isfinite(X[pr.ia]) & np.isfinite(X[pr.ib]); self.m = m; mf = m.astype(float)
        self.n = _cs(mf); self.St = _cs(mf*self.t); self.Stt = _cs(mf*self.t*self.t)
    def stats(self, X):
        pr = self.pr; m = self.m; n = self.n; t = self.t; cv = self.cv
        cen = _cen(X); ca = cen[pr.ia][:, None]; cb = cen[pr.ib][:, None]
        A = np.where(m, X[pr.ia] - ca, 0.0); B = np.where(m, X[pr.ib] - cb, 0.0)
        Sa = _cs(A); Sb = _cs(B); Saa = _cs(A*A); Sbb = _cs(B*B); Sab = _cs(A*B); Sta = _cs(A*t); Stb = _cs(B*t)
        out = {}
        with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
            # ---------------- mean reference
            mac = Sa/n; mbc = Sb/n; va = (Saa - n*mac*mac)/(n-1); vb = (Sbb - n*mbc*mbc)/(n-1); cab = (Sab - n*mac*mbc)/(n-1)
            sa = np.sqrt(np.clip(va, 0, None)); sb = np.sqrt(np.clip(vb, 0, None)); rraw = cab/(sa*sb); r = np.clip(rraw, -self.clip, self.clip)
            ok = m & (n >= 3) & (sa > 1e-10) & (sb > 1e-10) & (sa > 1e-7*np.abs(mac)) & (sb > 1e-7*np.abs(mbc))
            ma = mac + ca; mb = mbc + cb
            ok &= ~((np.abs(ma) > 1e-8) & (sa/np.abs(ma) < cv)) & ~((np.abs(mb) > 1e-8) & (sb/np.abs(mb) < cv))
            za = (A - mac)/sa; zb = (B - mbc)/sb
            w2 = (za*za - 2*r*za*zb + zb*zb)/(1 - r*r)
            out['NAD-M'] = _share(ok & (w2 >= THR[_idx(n)]**2), ok)
            p = fdtrc(2, n-2, (n-2)/(2*(n-1))*w2/(1 + 1/n)); out['CON-M'] = _mean(_nlp(p), ok)
            g = ok & (np.abs(rraw) >= self.rmin); s = np.sign(r)
            minor = (zb - s*za)**2/(2*(1 - np.abs(r))); pm = fdtrc(1, n-1, minor/(1 + 1/n)); cnt = g.sum(0)
            out['REL-M'] = np.where(cnt >= self.min_rel, np.where(g, _nlp(pm), 0.0).sum(0)/np.maximum(cnt, 1), np.nan)
            # ---------------- trend reference
            tbar = self.St/n; Sxx = self.Stt - n*tbar*tbar
            ba = (Sta - n*tbar*mac)/Sxx; bb = (Stb - n*tbar*mbc)/Sxx
            rssa = Saa - n*mac*mac - ba*ba*Sxx; rssb = Sbb - n*mbc*mbc - bb*bb*Sxx; rsc = Sab - n*mac*mbc - ba*bb*Sxx
            d = n - 2; sa = np.sqrt(np.clip(rssa/d, 0, None)); sb = np.sqrt(np.clip(rssb/d, 0, None)); rraw = (rsc/d)/(sa*sb); r = np.clip(rraw, -self.clip, self.clip)
            dt = t - tbar; h = 1/n + dt*dt/Sxx
            ea = A - (mac + ba*dt); eb = B - (mbc + bb*dt)
            ok = m & (n >= 4) & (Sxx > 1e-9) & (sa > 1e-10) & (sb > 1e-10) & (sa > 1e-7*np.abs(mac)) & (sb > 1e-7*np.abs(mbc))
            ok &= ~((np.abs(ma) > 1e-8) & (sa/np.abs(ma) < cv)) & ~((np.abs(mb) > 1e-8) & (sb/np.abs(mb) < cv))
            za = ea/sa; zb = eb/sb
            w2 = (za*za - 2*r*za*zb + zb*zb)/(1 - r*r); f = (n-3)/(2*(n-2))*w2/(1 + h)
            out['NAD-T'] = _share(ok & (f >= FQ_T[_idx(n)]), ok)
            out['CON-T'] = _mean(_nlp(fdtrc(2, n-3, f)), ok)
            g = ok & (np.abs(rraw) >= self.rmin); s = np.sign(r)
            minor = (zb - s*za)**2/(2*(1 - np.abs(r))); pm = fdtrc(1, n-2, minor/(1 + h)); cnt = g.sum(0)
            out['REL-T'] = np.where(cnt >= self.min_rel, np.where(g, _nlp(pm), 0.0).sum(0)/np.maximum(cnt, 1), np.nan)
        return out

# ---------------------------------------------------------------- univariate and PCA baselines
def uni_stats(X, times=None, cv=CV):
    """UED and dT2 with both references. Returns dict of arrays over years."""
    Mx, T = X.shape; t = np.arange(T, dtype=float) if times is None else np.asarray(times, float); t = (t - t.mean())[None, :]
    cen = _cen(X)[:, None]; m = np.isfinite(X); a = np.where(m, X - cen, 0.0); mf = m.astype(float); tt = mf*t
    n = _cs(mf); St = _cs(tt); Stt = _cs(tt*t); Sa = _cs(a); Sta = _cs(a*t); Saa = _cs(a*a); out = {}
    with np.errstate(invalid='ignore', divide='ignore'):
        mc = Sa/n; mu = mc + cen
        sd = np.sqrt(np.clip((Saa - n*mc*mc)/(n-1), 0, None))
        ok = m & (n >= 3) & (sd > 1e-10) & (sd > 1e-7*np.abs(mc)) & ~((np.abs(mu) > 1e-8) & (sd/np.abs(mu) < cv))
        tm = (a - mc)/(sd*np.sqrt(1 + 1/n)); thr = TQ_M[_idx(n)]
        out['UED-M'] = _share(ok & (np.abs(tm) >= thr), ok); out['dT2-M'] = _mean(np.minimum(np.abs(tm), CAPV)**2, ok)
        tbar = St/n; Sxx = Stt - n*tbar*tbar; b = (Sta - n*tbar*mc)/Sxx
        rss = Saa - n*mc*mc - b*b*Sxx; sd = np.sqrt(np.clip(rss/(n-2), 0, None)); dt = t - tbar; h = 1/n + dt*dt/Sxx
        ok = m & (n >= 4) & (Sxx > 1e-9) & (sd > 1e-10) & (sd > 1e-7*np.abs(mc)) & ~((np.abs(mu) > 1e-8) & (sd/np.abs(mu) < cv))
        tt_ = (a - (mc + b*dt))/(sd*np.sqrt(1 + h)); thr = TQ_T[_idx(n)]
        out['UED-T'] = _share(ok & (np.abs(tt_) >= thr), ok); out['dT2-T'] = _mean(np.minimum(np.abs(tt_), CAPV)**2, ok)
    return out

def spe_stats(X, times=None, share=0.8, min_series=5, cv=CV, years=None):
    """PCA-residual monitor with the mean reference (SPE-M, as in the first benchmark) and the trend reference (SPE-T)."""
    Mx, T = X.shape; t = np.arange(T, dtype=float) if times is None else np.asarray(times, float); t = t - t.mean()
    outM = np.full(T, np.nan); outT = np.full(T, np.nan)
    for j in (range(3, T) if years is None else [y for y in years if y >= 3]):
        tr = X[:, :j]; mt = np.isfinite(tr); cnt = mt.sum(1); x = X[:, j]; tj = t[:j]
        with np.errstate(invalid='ignore', divide='ignore'):
            mu = np.where(cnt > 0, np.where(mt, tr, 0).sum(1)/np.maximum(cnt, 1), np.nan)
            sd = np.sqrt(np.where(mt, (tr - mu[:, None])**2, 0).sum(1)/np.maximum(cnt - 1, 1))
            el = (cnt >= 3) & (sd > 1e-10) & (sd > 1e-7*np.abs(mu)) & ~((np.abs(mu) > 1e-8) & (sd/np.abs(mu) < cv))
            if (el & np.isfinite(x)).sum() >= min_series:
                Z = np.where(mt, (tr - mu[:, None])/sd[:, None], 0.0)[el]
                _, S, Vt = np.linalg.svd(Z.T, full_matrices=False); var = S**2
                if var.sum() > 0:
                    k = max(1, min(int(np.searchsorted(np.cumsum(var)/var.sum(), share) + 1), j - 2))
                    zh = np.clip((x[el] - mu[el])/sd[el], -CAPV, CAPV); ob = np.isfinite(zh); z0 = np.where(ob, zh, 0.0)
                    Vk = Vt[:k]; outM[j] = float(((z0 - Vk.T @ (Vk @ z0))[ob]**2).mean())
            if j >= 4:
                tb = np.where(mt, tj[None, :], 0).sum(1)/np.maximum(cnt, 1); dtr = np.where(mt, tj[None, :] - tb[:, None], 0.0)
                Sxx = (dtr**2).sum(1); b = np.where(mt, (tr - mu[:, None])*dtr, 0).sum(1)/Sxx
                res = np.where(mt, tr - mu[:, None] - b[:, None]*dtr, 0.0); s = np.sqrt((res**2).sum(1)/(cnt - 2))
                dt0 = t[j] - tb; h = 1/cnt + dt0**2/Sxx
                el = (cnt >= 4) & (Sxx > 1e-9) & (s > 1e-10) & (s > 1e-7*np.abs(mu)) & ~((np.abs(mu) > 1e-8) & (s/np.abs(mu) < cv))
                if (el & np.isfinite(x)).sum() >= min_series:
                    Z = (res/s[:, None])[el]
                    _, S, Vt = np.linalg.svd(Z.T, full_matrices=False); var = S**2
                    if var.sum() > 0:
                        k = max(1, min(int(np.searchsorted(np.cumsum(var)/var.sum(), share) + 1), j - 3))
                        zh = np.clip((x[el] - mu[el] - b[el]*dt0[el])/(s[el]*np.sqrt(1 + h[el])), -CAPV, CAPV); ob = np.isfinite(zh); z0 = np.where(ob, zh, 0.0)
                        Vk = Vt[:k]; outT[j] = float(((z0 - Vk.T @ (Vk @ z0))[ob]**2).mean())
    return {'SPE-M': outM, 'SPE-T': outT}

class AllStats:
    """All twelve statistics of a data matrix; reuse one instance for the surrogates of that matrix."""
    names = NAMES_NAD + NAMES_UNI
    def __init__(self, X, pr, times=None): self.E = VariantEngine(X, pr, times); self.times = times
    def __call__(self, X, years=None):
        o = self.E.stats(X); o.update(uni_stats(X, self.times)); o.update(spe_stats(X, self.times, years=years)); return o
