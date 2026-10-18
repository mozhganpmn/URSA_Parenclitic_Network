"""Domain-to-domain activated-edge density, observed and under refitted increment-shift surrogates (diagnostic).
Run from the repository root: python experiments/domain_matrix.py [N]. Writes results/domain_matrix.json."""
import sys, os, json, time, numpy as np, pandas as pd
sys.path.insert(0, '.')
import parenclitic_core as pc
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
DATA = 'data'; RES = 'results'
TARGETS = {'Vancouver': [2020, 2023], 'Calgary': [2015, 2020, 2023]}
MIN_ELIG = 30   # cells with fewer eligible pairs are not shown

def prep(city):
    full = pd.read_excel(os.path.join(DATA, {'Vancouver': 'Input.xlsx', 'Calgary': 'Input_Calgary.xlsx'}[city]), sheet_name='Sheet1')
    yc = sorted([c for c in full.columns if isinstance(c, (int, np.integer))])
    V = full[yc].apply(pd.to_numeric, errors='coerce').values.astype(float)
    keep = np.isfinite(V).sum(1) >= 3
    return full[keep].reset_index(drop=True), V[keep], yc

class Eng(pc.NadEngine):
    def act(self, X):
        pr = self.pr; m = self.m; n = self.n
        cen = np.array([np.nanmean(r) if np.isfinite(r).any() else 0.0 for r in X]); ca = cen[pr.ia][:, None]; cb = cen[pr.ib][:, None]
        A = np.where(m, X[pr.ia] - ca, 0.0); B = np.where(m, X[pr.ib] - cb, 0.0)
        def cs(z): c = np.cumsum(z, 1); return np.concatenate([np.zeros((z.shape[0], 1)), c[:, :-1]], 1)
        Sa = cs(A); Sb = cs(B); Saa = cs(A*A); Sbb = cs(B*B); Sab = cs(A*B)
        with np.errstate(invalid='ignore', divide='ignore'):
            mac = Sa/n; mbc = Sb/n; va = (Saa - n*mac*mac)/(n-1); vb = (Sbb - n*mbc*mbc)/(n-1); cab = (Sab - n*mac*mbc)/(n-1)
            sa = np.sqrt(np.clip(va, 0, None)); sb = np.sqrt(np.clip(vb, 0, None)); r = np.clip(cab/(sa*sb), -self.clip, self.clip)
            ma = mac + ca; mb = mbc + cb
            ok = self.base_ok & (sa > 1e-10) & (sb > 1e-10) & (sa > 1e-7*np.abs(mac)) & (sb > 1e-7*np.abs(mbc))
            ok &= ~((np.abs(ma) > 1e-8) & (sa/np.abs(ma) < self.cv)) & ~((np.abs(mb) > 1e-8) & (sb/np.abs(mb) < self.cv))
            za = (A - mac)/sa; zb = (B - mbc)/sb; w2 = (za*za - 2*r*za*zb + zb*zb)/(1 - r*r)
        return ok & (w2 >= self.th*self.th), ok

out = {}
for city, years in TARGETS.items():
    x, V, yc = prep(city); yi = {y: k for k, y in enumerate(yc)}; jj = [yi[y] for y in years]
    doms = sorted(set(x['Domain'])); di = {d: k for k, d in enumerate(doms)}; D = len(doms)
    pr = pc.Pairs(V.shape[0]); da = np.array([di[d] for d in x['Domain']])
    a_ = np.minimum(da[pr.ia], da[pr.ib]); b_ = np.maximum(da[pr.ia], da[pr.ib]); cell = a_*D + b_
    def agg(act, ok):
        Ea = np.zeros((D*D, len(jj))); El = np.zeros((D*D, len(jj)))
        for q, j in enumerate(jj):
            Ea[:, q] = np.bincount(cell, weights=act[:, j].astype(float), minlength=D*D)
            El[:, q] = np.bincount(cell, weights=ok[:, j].astype(float), minlength=D*D)
        return Ea, El
    eng = Eng(V, pr); act, ok = eng.act(V); Eo, Elo = agg(act, ok)
    draw = pc.make_increment_factory(V); rng = np.random.default_rng(20261017)
    dens = np.zeros((N, D*D, len(jj))); t0 = time.time()
    for s in range(N):
        a2, o2 = eng.act(draw(rng)); Es, Els = agg(a2, o2)
        with np.errstate(invalid='ignore', divide='ignore'): dens[s] = np.where(Els > 0, Es/Els, np.nan)
        if s == 20: print(city, 'est. total s', round((time.time()-t0)/21*N), flush=True)
    with np.errstate(invalid='ignore', divide='ignore'): dobs = np.where(Elo > 0, Eo/Elo, np.nan)
    mu = np.nanmean(dens, 0); sd = np.nanstd(dens, 0, ddof=1)
    res = dict(domains=doms, N=N, min_elig=MIN_ELIG)
    for q, y in enumerate(years):
        res[str(y)] = dict(obs=dobs[:, q].reshape(D, D).tolist(), null_mean=mu[:, q].reshape(D, D).tolist(), null_sd=sd[:, q].reshape(D, D).tolist(),
                           eligible=Elo[:, q].reshape(D, D).tolist(), activated=Eo[:, q].reshape(D, D).tolist())
    out[city] = res; print(city, 'done', flush=True)
json.dump(out, open(os.path.join(RES, 'domain_matrix.json'), 'w'))
