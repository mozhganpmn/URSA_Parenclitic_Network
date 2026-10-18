# Robustness checks: minimum training window, geography/source sensitivity, threshold-standardised scores,
# and the top-10 node-score tables for the significant city-years.
# Usage: python experiments/robustness_checks.py REPO_DIR N_SURROGATES OUT.json
import sys, os, json, time, numpy as np, pandas as pd
REPO = sys.argv[1]; N = int(sys.argv[2]); OUT = sys.argv[3]
sys.path.insert(0, REPO)
from parenclitic_core import *
D = os.path.join(REPO, 'data')
FILES = {'Vancouver': os.path.join(D, 'Input.xlsx'), 'Calgary': os.path.join(D, 'Input_Calgary.xlsx')}
H = pd.read_excel(FILES['Vancouver'], sheet_name='Harmonization')
nat = set((H[H['Status'].str.startswith('National')]['Row'] - 1).tolist())
prov = set((H[H['Geography'] == 'Province']['Row'] - 1).tolist())
REP = list(range(2015, 2025))
def holm(ps):
    keys = sorted(ps, key=ps.get); m = len(keys); out = {}; prev = 0.0
    for r, k in enumerate(keys): prev = min(1.0, max(prev, (m - r)*ps[k])); out[k] = prev
    return out
def load(city, drop_rows=()):
    full = pd.read_excel(FILES[city], sheet_name='Sheet1'); yc = sorted([c for c in full.columns if isinstance(c, (int, np.integer))])
    V = full[yc].apply(pd.to_numeric, errors='coerce').values.astype(float).copy()
    if drop_rows: V[list(drop_rows)] = np.nan
    keep = np.isfinite(V).sum(1) >= 3
    return full, full[keep].reset_index(drop=True), V[keep], yc, keep
def test(V, yc, nmin, NN, seed=20231):
    pr = Pairs(V.shape[0]); E = NadEngine(V, pr); E.base_ok = (E.n >= nmin) & E.m
    obs, ne = E.nad(V); draw = make_increment_factory(V); rng = np.random.default_rng(seed)
    null = np.array([E.nad(draw(rng))[0] for _ in range(NN)]); yi = {y: k for k, y in enumerate(yc)}
    yrs = {}; 
    for y in REP:
        j = yi[y]
        if ne[j] == 0 or not np.isfinite(obs[j]): continue
        nj = null[:, j]; k = int((nj >= obs[j] - 1e-12).sum())
        yrs[y] = dict(NAD=float(obs[j]), z=float((obs[j]-nj.mean())/nj.std(ddof=1)), NADstar=float(obs[j]-nj.mean()), P=(k+1)/(NN+1), eligible=int(ne[j]))
    ph = holm({y: v['P'] for y, v in yrs.items()})
    for y in yrs: yrs[y]['P_holm'] = ph[y]
    win = {}
    for wn, ys in {'2015-2016': [2015, 2016], '2020': [2020], '2022-2024': [2022, 2023, 2024]}.items():
        if any(y not in yrs for y in ys): continue
        js = [yi[y] for y in ys]; o = obs[js].mean(); nn = null[:, js].mean(1)
        win[wn] = dict(z=float((o-nn.mean())/nn.std(ddof=1)), P=(int((nn >= o - 1e-12).sum())+1)/(NN+1))
    wh = holm({k: v['P'] for k, v in win.items()})
    for k in win: win[k]['P_holm'] = wh[k]
    return dict(working=int(V.shape[0]), years=yrs, windows=win)
out = {'N': N}
t0 = time.time()
# sanity: engine with nmin=3 equals the reference implementation
full, x, V, yc, keep = load('Vancouver'); pr = Pairs(V.shape[0]); E = NadEngine(V, pr)
a = E.nad(V)[0]; b = nad_all(V, pr)[0]; out['engine_check_maxdiff'] = float(np.nanmax(np.abs(a-b))); print('engine check', out['engine_check_maxdiff'])
# A. minimum training window
out['nmin'] = {}
for city in ['Vancouver', 'Calgary']:
    _, x, V, yc, _ = load(city)
    for nmin in [3, 4, 5, 6]:
        out['nmin'][f'{city}_{nmin}'] = test(V, yc, nmin, N); print(city, 'nmin', nmin, round(time.time()-t0), 's', flush=True)
        json.dump(out, open(OUT, 'w'), indent=1)
# B. geography / source sensitivity
out['geo'] = {}
for city in ['Vancouver', 'Calgary']:
    for name, rows in {'no_national': nat, 'local_only': nat | prov}.items():
        _, x, V, yc, _ = load(city, rows)
        out['geo'][f'{city}_{name}'] = test(V, yc, 3, N); print(city, name, V.shape[0], round(time.time()-t0), 's', flush=True)
        json.dump(out, open(OUT, 'w'), indent=1)
# C. threshold-standardised scores and D. top-10 tables
def node_scores(V, pr, standardise):
    S = lfo_stats(V, pr); A = V[pr.ia]; B = V[pr.ib]; obs = np.isfinite(A) & np.isfinite(B)
    elig = S['ok'] & obs; w = np.minimum(weights(S, A, B), CAP)
    if standardise: w = w/S['th']
    C = np.where(elig, 1 - np.exp(-(S['n'] - 3)/LAM), 0.0); cw = np.where(elig, C*np.nan_to_num(w), 0.0)
    M, T = V.shape; num = np.zeros((M, T)); den = np.zeros((M, T))
    np.add.at(num, pr.ia, cw); np.add.at(num, pr.ib, cw); np.add.at(den, pr.ia, C); np.add.at(den, pr.ib, C)
    return np.where(den > 0, num/np.maximum(den, 1e-300), np.nan)
def domain_scores(V, pr, x, full, keep, standardise):
    cov_w = full.assign(k=keep).groupby('Indicator')['k'].mean().to_dict(); dom_of_ind = full.groupby('Indicator')['Domain'].first().to_dict()
    if not standardise: _, (inds, ind), (doms, dom) = scores(V, pr, list(x['Indicator']), cov_w, dom_of_ind); return doms, dom
    # same aggregation with standardised edge weights
    S = lfo_stats(V, pr); A = V[pr.ia]; B = V[pr.ib]; obs = np.isfinite(A) & np.isfinite(B); elig = S['ok'] & obs
    w = np.minimum(weights(S, A, B), CAP)/S['th']; C = np.where(elig, 1 - np.exp(-(S['n'] - 3)/LAM), 0.0); cw = np.where(elig, C*np.nan_to_num(w), 0.0)
    ind_of = list(x['Indicator']); inds = sorted(set(ind_of)); ii = {k: n for n, k in enumerate(inds)}
    ia = np.array([ii[ind_of[i]] for i in pr.ia]); ib = np.array([ii[ind_of[i]] for i in pr.ib]); T = V.shape[1]
    inum = np.zeros((len(inds), T)); iden = np.zeros((len(inds), T)); np.add.at(inum, ia, cw); np.add.at(iden, ia, C); same = ia == ib
    np.add.at(inum, ib[~same], cw[~same]); np.add.at(iden, ib[~same], C[~same]); ind = np.where(iden > 0, inum/np.maximum(iden, 1e-300), np.nan)
    doms = sorted(set(dom_of_ind[k] for k in inds)); dom = np.full((len(doms), T), np.nan)
    for d_i, d in enumerate(doms):
        ks = [ii[k] for k in inds if dom_of_ind[k] == d]; wv = np.array([cov_w[inds[k]] for k in ks])[:, None]; v = ind[ks]; ok = np.isfinite(v); tw = (wv*ok).sum(0)
        dom[d_i] = np.where(tw > 0, (np.where(ok, v, 0)*wv).sum(0)/np.maximum(tw, 1e-300), np.nan)
    return doms, dom
out['std'] = {}; out['top10'] = {}
for city, yrs in [('Vancouver', [2020, 2023]), ('Calgary', [2015, 2020, 2023])]:
    full, x, V, yc, keep = load(city); pr = Pairs(V.shape[0]); yi = {y: k for k, y in enumerate(yc)}
    raw = node_scores(V, pr, False); std = node_scores(V, pr, True)
    dr, Dr = domain_scores(V, pr, x, full, keep, False); ds, Ds = domain_scores(V, pr, x, full, keep, True)
    peaks = {}
    for k, d in enumerate(dr):
        yr_raw = int(REP[int(np.nanargmax(Dr[k, [yi[y] for y in REP]]))]); yr_std = int(REP[int(np.nanargmax(Ds[k, [yi[y] for y in REP]]))]); peaks[d] = [yr_raw, yr_std]
    out['std'][city] = dict(domain_peak_raw_vs_std=peaks, same=sum(1 for v in peaks.values() if v[0] == v[1]), n=len(peaks), top={})
    H_ = H.set_index('Row')
    for y in yrs:
        j = yi[y]; other = [yi[t] for t in REP if t != y]
        def excess(sc): return sc[:, j] - np.nanmedian(sc[:, other], axis=1)
        er = excess(raw); es = excess(std); 
        top_r = [int(i) for i in np.argsort(-np.where(np.isfinite(er), er, -1e9))[:10]]; top_s = [int(i) for i in np.argsort(-np.where(np.isfinite(es), es, -1e9))[:10]]
        top20_r = set(int(i) for i in np.argsort(-np.where(np.isfinite(er), er, -1e9))[:20]); 
        out['std'][city]['top'][y] = dict(top10_overlap=len(set(top_r) & set(top_s)), top10_raw_in_std_top20=len(set(top_r) & set(int(i) for i in np.argsort(-np.where(np.isfinite(es), es, -1e9))[:20])),
            spearman_excess=float(pd.Series(er).corr(pd.Series(es), method='spearman')))
        rows = []
        for i in top_r:
            rid = int(full.index[keep][i]) + 1
            vals = V[i]; rows.append(dict(sub=str(x.loc[i, 'Subindicator']), indicator=str(x.loc[i, 'Indicator']), domain=str(x.loc[i, 'Domain']), node=round(float(raw[i, j]), 2),
                median_other=round(float(np.nanmedian(raw[i, other])), 2), prev=None if not np.isfinite(vals[j-1]) else float(vals[j-1]), val=None if not np.isfinite(vals[j]) else float(vals[j]),
                status=str(H_.loc[rid, 'Status']) if rid in H_.index else '', geography=str(H_.loc[rid, 'Geography']), rank_in_std=int(np.where(np.argsort(-np.where(np.isfinite(es), es, -1e9)) == i)[0][0]) + 1))
        out['top10'][f'{city}_{y}'] = rows
    print(city, 'scores done', flush=True)
json.dump(out, open(OUT, 'w'), indent=1); print('done', round(time.time()-t0), 's')
