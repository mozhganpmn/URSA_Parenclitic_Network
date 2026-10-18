"""Benchmark of the parenclitic NAD monitor against three standard multivariate monitors.

All monitors are prospective: the statistic for year t uses only the years before t (leave-future-out), the same
eligibility rules (at least three training values, zero-variance guard, CV < 1% exclusion), and the same 20-unit cap.
  NAD   share of pairs outside the F(2, n-2) boundary (the framework of the paper)
  UED   univariate exceedance density: share of series whose held-out value lies outside the two-sided
        t(n-1) 95% limits (exact for independent normal sampling; no pairwise information)
  dT2   diagonal Hotelling T2 (Dempster-type statistic for p > n): mean squared held-out t score
  SPE   PCA residual (squared prediction error): LFO principal components that explain 80% of the training variance
        (at most n-2 components), mean squared residual of the held-out standardised vector

Modes (each writes a JSON file next to this script):
  check                  verify the vectorised univariate statistics against a direct per-series implementation
  null  D N              type-I error of the surrogate test in the seven null regimes (D datasets, N surrogates)
  power D Dnull          power at a matched 5% false-positive rate (critical values from null datasets of the same process)
  real  N                Vancouver and Calgary, refitted increment-shift null shared by all monitors
"""
import sys, os, json, time, numpy as np, pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..')); sys.path.insert(0, HERE)
from parenclitic_core import Pairs, NadEngine, make_increment_factory
from synthetic_regimes import gen, kinds, M, T
CAPV = 20.0; CV = 0.01
TCRIT = np.concatenate([[np.nan]*3, stats.t.ppf(0.975, np.arange(2, 60))])   # index n = number of training values
METHODS = ['NAD', 'UED', 'dT2', 'SPE']

def eligible(mu, sd, cnt, cv=CV):
    with np.errstate(invalid='ignore'):
        return (cnt >= 3) & (sd > 1e-10) & (sd > 1e-7*np.abs(mu)) & ~((np.abs(mu) > 1e-8) & (sd/np.abs(mu) < cv))

def uni_lfo(X):
    """Held-out t scores of every series and year against its own earlier values (exclusive cumulative sums)."""
    cen = np.array([np.nanmean(r) if np.isfinite(r).any() else 0.0 for r in X])[:, None]
    m = np.isfinite(X); a = np.where(m, X - cen, 0.0)
    def cs(z): c = np.cumsum(z, 1); return np.concatenate([np.zeros((z.shape[0], 1)), c[:, :-1]], 1)
    n = cs(m.astype(float)); S = cs(a); SS = cs(a*a)
    with np.errstate(invalid='ignore', divide='ignore'):
        mc = S/n; sd = np.sqrt(np.clip((SS - n*mc*mc)/(n-1), 0, None)); mu = mc + cen
        ok = m & eligible(mu, sd, n)
        t = (a - mc)/(sd*np.sqrt(1 + 1/n))
    return ok, t, n

def ued_dt2(X):
    ok, t, n = uni_lfo(X); ne = ok.sum(0)
    thr = TCRIT[np.clip(n.astype(int), 0, len(TCRIT)-1)]
    act = ok & (np.abs(t) >= thr); tc = np.where(ok, np.minimum(np.abs(t), CAPV), 0.0)
    ued = np.where(ne > 0, act.sum(0)/np.maximum(ne, 1), np.nan); dt2 = np.where(ne > 0, (tc**2).sum(0)/np.maximum(ne, 1), np.nan)
    return ued, dt2

def spe_all(X, share=0.8, min_series=5):
    Mx, Tx = X.shape; out = np.full(Tx, np.nan)
    for j in range(3, Tx):
        tr = X[:, :j]; mt = np.isfinite(tr); cnt = mt.sum(1)
        with np.errstate(invalid='ignore', divide='ignore'):
            mu = np.where(cnt > 0, np.where(mt, tr, 0).sum(1)/np.maximum(cnt, 1), np.nan)
            sd = np.sqrt(np.where(mt, (tr - mu[:, None])**2, 0).sum(1)/np.maximum(cnt - 1, 1))
        x = X[:, j]; el = eligible(mu, sd, cnt) & np.isfinite(x)
        if el.sum() < min_series: continue
        e2 = eligible(mu, sd, cnt)
        with np.errstate(invalid='ignore', divide='ignore'): Z = np.where(mt, (tr - mu[:, None])/sd[:, None], 0.0)[e2]
        _, S, Vt = np.linalg.svd(Z.T, full_matrices=False); var = S**2
        if var.sum() <= 0: continue
        k = int(np.searchsorted(np.cumsum(var)/var.sum(), share) + 1); k = max(1, min(k, j - 2))
        zh = np.clip((x[e2] - mu[e2])/sd[e2], -CAPV, CAPV); ob = np.isfinite(zh); z0 = np.where(ob, zh, 0.0)
        Vk = Vt[:k]; r = (z0 - Vk.T @ (Vk @ z0))[ob]; out[j] = float((r**2).mean())
    return out

def all_stats(X, E):
    ued, dt2 = ued_dt2(X)
    return dict(NAD=E.nad(X)[0], UED=ued, dT2=dt2, SPE=spe_all(X))

def mc(X, E, years, N, rng, draw, conservative=False):
    obs = all_stats(X, E); null = {k: np.full((N, X.shape[1]), np.nan) for k in METHODS}
    for i in range(N):
        s = all_stats(draw(rng), E)
        for k in METHODS: null[k][i] = s[k]
    res = {k: {} for k in METHODS}
    for k in METHODS:
        for j in years:
            nj = null[k][:, j]; nj = nj[np.isfinite(nj)]; o = obs[k][j]
            if not np.isfinite(o) or len(nj) < 10: continue
            gt = int((nj > o + 1e-12).sum()); eq = int((np.abs(nj - o) <= 1e-12).sum())
            p = (gt + eq + 1)/(len(nj) + 1) if conservative else (gt + rng.random()*eq + 1)/(len(nj) + 1)
            res[k][j] = dict(obs=float(o), null_mean=float(nj.mean()), null_sd=float(nj.std(ddof=1)), P=float(p),
                             z=float((o - nj.mean())/nj.std(ddof=1)) if nj.std(ddof=1) > 0 else float('nan'))
    return res

def holm(ps):
    keys = sorted(ps, key=ps.get); m = len(keys); out = {}; prev = 0.0
    for r, k in enumerate(keys): prev = min(1.0, max(prev, (m - r)*ps[k])); out[k] = prev
    return out

# ------------------------------------------------------------------ check
def check():
    rng = np.random.default_rng(5); X = gen('AR(1)', rng); X[3, :4] = np.nan
    ok, t, n = uni_lfo(X); ued, dt2 = ued_dt2(X); worst = 0.0; el_ok = True
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            tr = X[i, :j][np.isfinite(X[i, :j])]
            if not np.isfinite(X[i, j]) or len(tr) < 3: el_ok &= not ok[i, j]; continue
            mu, sd = tr.mean(), tr.std(ddof=1)
            if not eligible(np.array([mu]), np.array([sd]), np.array([len(tr)]))[0]: el_ok &= not ok[i, j]; continue
            tt = (X[i, j] - mu)/(sd*np.sqrt(1 + 1/len(tr))); worst = max(worst, abs(tt - t[i, j])); el_ok &= bool(ok[i, j])
    print('max |t| difference', worst, '| eligibility identical', el_ok)
    # type-I rate of the closed-form univariate threshold under iid normal sampling
    rng = np.random.default_rng(1); Z = rng.normal(size=(20000, 11)); u = []
    for n_ in [3, 5, 10]:
        tr = Z[:, :n_]; x = Z[:, n_]; tt = (x - tr.mean(1))/(tr.std(1, ddof=1)*np.sqrt(1 + 1/n_)); u.append(float((np.abs(tt) >= TCRIT[n_]).mean()))
    print('univariate threshold false-positive rate at n = 3, 5, 10:', u)
    X = gen('iid', np.random.default_rng(2)); print('SPE by year', np.round(spe_all(X), 3))

# ------------------------------------------------------------------ null calibration of the surrogate tests
def run_null(D, N):
    pr = Pairs(M); out = {}
    for kind in kinds:
        rng = np.random.default_rng(20261014); data = [gen(kind, rng) for _ in range(D)]
        r2 = np.random.default_rng(7); rates = {k: [] for k in METHODS}; t0 = time.time()
        for x in data:
            E = NadEngine(x, pr); res = mc(x, E, range(5, T), N, r2, make_increment_factory(x))
            for k in METHODS: rates[k].append(np.mean([v['P'] <= 0.05 for v in res[k].values()]))
        out[kind] = {}
        for k in METHODS:
            a = np.array(rates[k]); bs = [a[r2.integers(0, D, D)].mean() for _ in range(2000)]
            out[kind][k] = dict(rate=round(float(a.mean()), 3), ci95=[round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)])
        print(kind, {k: out[kind][k]['rate'] for k in METHODS}, f'{time.time()-t0:.0f}s', flush=True)
    json.dump(dict(datasets=D, surrogates=N, series=M, years=T, tested_year_indices=list(range(5, T)), results=out), open(os.path.join(HERE, 'benchmark_null.json'), 'w'), indent=1)

# ------------------------------------------------------------------ power at a matched false-positive rate
J = 10
def ar1(rng, shape):
    e = rng.normal(size=shape); a = e.copy()
    for t in range(1, shape[-1]): a[..., t] = 0.5*a[..., t-1] + np.sqrt(0.75)*e[..., t]
    return a
def finish(x, rng, miss=0.10):
    x = x + 10.0; x[rng.random(x.shape) < miss] = np.nan; return x
def gen_marginal(rng, frac, delta, slope, miss=0.10):
    x = ar1(rng, (M, T)) + rng.normal(0, slope, M)[:, None]*np.arange(T)
    k = int(round(frac*M)); idx = rng.choice(M, k, replace=False); x[idx, J] += delta*rng.choice([-1, 1], k)
    return finish(x, rng, miss)
def gen_break(rng, frac, lam, slope, miss=0.10):
    """Three blocks of ten series that share an AR(1) factor. In year J a fraction of the series in each block takes
    an independent draw in place of the shared factor, with the same variance, so no marginal distribution changes."""
    blk = np.repeat(np.arange(3), 10); f = ar1(rng, (3, T)); eta = ar1(rng, (M, T)); sg = rng.choice([-1, 1], M)
    x = lam*sg[:, None]*f[blk] + np.sqrt(1 - lam**2)*eta
    for b in range(3):
        k = int(round(frac*10))
        if k: idx = rng.choice(np.where(blk == b)[0], k, replace=False); x[idx, J] = lam*rng.normal(size=k) + np.sqrt(1 - lam**2)*eta[idx, J]
    x = x + rng.normal(0, slope, M)[:, None]*np.arange(T)
    return finish(x, rng, miss)
def gen_opposing(rng, delta, slope, miss=0.10):
    """Same block factor model (loading 0.9). In year J half of the series in each block move by +delta and half by
    -delta marginal standard deviations in the direction of the shared factor, so every pair that normally moves
    together is pushed in opposite directions while each marginal departure stays moderate."""
    blk = np.repeat(np.arange(3), 10); f = ar1(rng, (3, T)); eta = ar1(rng, (M, T)); sg = rng.choice([-1, 1], M); lam = 0.9
    x = lam*sg[:, None]*f[blk] + np.sqrt(1 - lam**2)*eta
    for b in range(3):
        idx = rng.permutation(np.where(blk == b)[0]); h = len(idx)//2
        x[idx[:h], J] += delta*sg[idx[:h]]; x[idx[h:], J] -= delta*sg[idx[h:]]
    x = x + rng.normal(0, slope, M)[:, None]*np.arange(T)
    return finish(x, rng, miss)
def stat_at_J(x, pr):
    E = NadEngine(x, pr); s = all_stats(x, E); return [float(s[k][J]) for k in METHODS]
def matched_power(nul, alt, alpha=0.05, rng=None):
    nul = nul[np.isfinite(nul)]; alt = alt[np.isfinite(alt)]
    for u in np.unique(nul):
        pg = float((nul > u).mean())
        if pg <= alpha:
            peq = float((nul == u).mean()); g = (alpha - pg)/peq if peq > 0 else 0.0
            return float((alt > u).mean() + min(max(g, 0.0), 1.0)*(alt == u).mean())
    return 0.0
def run_power(D, Dn):
    pr = Pairs(M); out = []
    scen = []
    for slope, bname in [(0.0, 'stationary'), (0.3, 'trending')]:
        for frac in [0.1, 0.3]:
            for delta in [2, 3, 4]: scen.append(dict(kind='marginal shift', base=bname, frac=frac, delta=delta, slope=slope))
        for lam in [0.9, 0.7]:
            for frac in [0.3, 0.6, 1.0]: scen.append(dict(kind='relationship break', base=bname, frac=frac, lam=lam, slope=slope))
        for delta in [1.0, 1.5, 2.0]: scen.append(dict(kind='opposing shift', base=bname, frac=1.0, delta=delta, lam=0.9, slope=slope))
    cache = {}
    for sc in scen:
        gk = ('marginal shift' if sc['kind'] == 'marginal shift' else 'factor model', sc['base'], sc.get('lam'))
        if gk not in cache:
            rng = np.random.default_rng(4242 + len(cache))
            g0 = (lambda r: gen_marginal(r, 0.0, 0, sc['slope'])) if sc['kind'] == 'marginal shift' else (lambda r: gen_break(r, 0.0, sc['lam'], sc['slope']))
            cache[gk] = np.array([stat_at_J(g0(rng), pr) for _ in range(Dn)])
        nul = cache[gk]
        rng = np.random.default_rng(777 + len(out))
        if sc['kind'] == 'marginal shift': g1 = lambda r: gen_marginal(r, sc['frac'], sc['delta'], sc['slope'])
        elif sc['kind'] == 'opposing shift': g1 = lambda r: gen_opposing(r, sc['delta'], sc['slope'])
        else: g1 = lambda r: gen_break(r, sc['frac'], sc['lam'], sc['slope'])
        alt = np.array([stat_at_J(g1(rng), pr) for _ in range(D)])
        rec = dict(sc); rec.update({k: round(matched_power(nul[:, i], alt[:, i]), 3) for i, k in enumerate(METHODS)}); out.append(rec); print(rec, flush=True)
    json.dump(dict(datasets=D, null_datasets=Dn, series=M, years=T, target_index=J, alpha=0.05, results=out), open(os.path.join(HERE, 'benchmark_power.json'), 'w'), indent=1)

# ------------------------------------------------------------------ Vancouver and Calgary
_D = os.path.join(HERE, '..', 'data'); FILES = {'Vancouver': os.path.join(_D, 'Input.xlsx'), 'Calgary': os.path.join(_D, 'Input_Calgary.xlsx')}
def run_real(N):
    out = {}
    for city in ['Vancouver', 'Calgary']:
        full = pd.read_excel(FILES[city], sheet_name='Sheet1'); yc = sorted([c for c in full.columns if isinstance(c, (int, np.integer))])
        V = full[yc].apply(pd.to_numeric, errors='coerce').values.astype(float); V = V[np.isfinite(V).sum(1) >= 3]
        pr = Pairs(V.shape[0]); E = NadEngine(V, pr); yi = {y: k for k, y in enumerate(yc)}; rep = list(range(2015, 2025))
        rng = np.random.default_rng(20231); t0 = time.time()
        res = mc(V, E, [yi[y] for y in rep], N, rng, make_increment_factory(V), conservative=True)
        out[city] = dict(working=int(V.shape[0]), surrogates=N)
        for k in METHODS:
            yr = {y: res[k][yi[y]] for y in rep if yi[y] in res[k]}; ph = holm({y: yr[y]['P'] for y in yr})
            for y in yr: yr[y]['P_holm'] = ph[y]
            out[city][k] = {str(y): {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()} for y, v in yr.items()}
        print(city, f'{time.time()-t0:.0f}s', {k: [y for y in rep if str(y) in out[city][k] and out[city][k][str(y)]['P_holm'] < 0.05] for k in METHODS}, flush=True)
    json.dump(out, open(os.path.join(HERE, 'benchmark_real.json'), 'w'), indent=1)

if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'check': check()
    elif mode == 'null': run_null(int(sys.argv[2]), int(sys.argv[3]))
    elif mode == 'power': run_power(int(sys.argv[2]), int(sys.argv[3]))
    elif mode == 'real': run_real(int(sys.argv[2]))
