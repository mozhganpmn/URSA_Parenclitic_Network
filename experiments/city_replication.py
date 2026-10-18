# Vancouver and Calgary on the harmonised inputs: Monte Carlo NAD test with the refitted increment-shift null
# (primary) and the detrended circular-shift null (sensitivity), scenario-window tests, domain maxima and drill-down,
# for the full sub-indicator sets and the matched core.
import sys, json, numpy as np, pandas as pd
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from parenclitic_core import *
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
_D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
FILES = {'Vancouver': os.path.join(_D, 'Input.xlsx'), 'Calgary': os.path.join(_D, 'Input_Calgary.xlsx')}
_H = pd.read_excel(FILES['Vancouver'], sheet_name='Harmonization')
_nv = _H['Vancouver years']; _nc = _H['Calgary years']
core_rows = set((_H[(~_H['Status'].str.startswith(('Close', 'Vancouver'))) & (_nv >= 4) & (_nc >= 4)]['Row'] - 1).tolist())
WINDOWS = {'2015-2016': [2015, 2016], '2020': [2020], '2022-2024': [2022, 2023, 2024]}
def holm(ps):
    keys = sorted(ps, key=ps.get); m = len(keys); out = {}; prev = 0.0
    for r, k in enumerate(keys): prev = min(1.0, max(prev, (m - r)*ps[k])); out[k] = prev
    return out
def run(city, subset):
    full = pd.read_excel(FILES[city], sheet_name='Sheet1'); yc = sorted([c for c in full.columns if isinstance(c, (int, np.integer))])
    Vall = full[yc].apply(pd.to_numeric, errors='coerce').values.astype(float).copy()
    if subset == 'core': Vall[[i for i in range(len(full)) if i not in core_rows]] = np.nan
    keep = np.isfinite(Vall).sum(1) >= 3; x = full[keep].reset_index(drop=True); V = Vall[keep]; pr = Pairs(V.shape[0])
    cov_w = full.assign(k=keep).groupby('Indicator')['k'].mean().to_dict(); dom_of_ind = full.groupby('Indicator')['Domain'].first().to_dict()
    yi = {y: k for k, y in enumerate(yc)}; rep = list(range(2015, 2025)); res = {'city': city, 'subset': subset, 'working': int(V.shape[0])}
    E = NadEngine(V, pr)
    for nulln, fac, NN in [('increment', make_increment_factory, N), ('detrended', lambda X: make_shift_factory(X, 1), max(N//5, 1000))]:
        rng = np.random.default_rng(20231); obs, ne = nad_all(V, pr); draw = fac(V)
        null = np.array([E.nad(draw(rng))[0] for _ in range(NN)])
        yr = {}
        for y in rep:
            j = yi[y]; nj = null[:, j]; gt = int((nj > obs[j] + 1e-12).sum()); eq = int((np.abs(nj - obs[j]) <= 1e-12).sum())
            yr[y] = dict(NAD=round(float(obs[j]), 4), null_mean=round(float(nj.mean()), 4), null_sd=round(float(nj.std(ddof=1)), 4),
                         NADstar=round(float(obs[j] - nj.mean()), 4), z=round(float((obs[j] - nj.mean())/nj.std(ddof=1)), 2),
                         k=gt + eq, N=NN, P=(gt + eq + 1)/(NN + 1), pct=round(float((nj < obs[j]).mean()*100), 2), eligible=int(ne[j]),
                         q=[round(float(v), 4) for v in np.percentile(nj, [2.5, 25, 50, 75, 97.5])])
        ph = holm({y: yr[y]['P'] for y in rep})
        for y in rep: yr[y]['P_holm'] = ph[y]
        win = {}
        for wn, ys in WINDOWS.items():
            js = [yi[y] for y in ys]; o = obs[js].mean(); nn = null[:, js].mean(1)
            win[wn] = dict(NADstar=round(float(o - nn.mean()), 4), z=round(float((o - nn.mean())/nn.std(ddof=1)), 2), P=(int((nn >= o - 1e-12).sum()) + 1)/(NN + 1))
        wh = holm({k: v['P'] for k, v in win.items()})
        for k in win: win[k]['P_holm'] = wh[k]
        res[nulln] = dict(years=yr, windows=win)
        if nulln == 'increment':
            pass
    node, (inds, ind), (doms, dom) = scores(V, pr, list(x['Indicator']), cov_w, dom_of_ind)
    D = pd.DataFrame(dom, index=doms, columns=yc); Dr = D.loc[:, 2015:2024]
    drill = {}
    for d in doms:
        y = int(Dr.loc[d].idxmax()); j = yi[y]; other = [yi[t] for t in rep if t != y]
        ind_rows = [k for k, nm in enumerate(inds) if dom_of_ind[nm] == d]
        top_ind = sorted([(inds[k], round(float(ind[k, j]), 2), round(float(np.nanmedian(ind[k, other])), 2)) for k in ind_rows if np.isfinite(ind[k, j])], key=lambda t: -(t[1] - t[2]))[:3]
        rows = [k for k in range(len(x)) if dom_of_ind[x.loc[k, 'Indicator']] == d]
        top_node = sorted([(x.loc[k, 'Subindicator'], round(float(node[k, j]), 2), round(float(np.nanmedian(node[k, other])), 2)) for k in rows if np.isfinite(node[k, j])], key=lambda t: -(t[1] - t[2]))[:4]
        drill[d] = dict(year=y, D=round(float(Dr.loc[d, y]), 2), top_indicators=top_ind, top_nodes=top_node)
    res['domain_scores'] = {d: {int(y): (None if not np.isfinite(v) else round(float(v), 3)) for y, v in D.loc[d].items()} for d in doms}
    res['drill'] = drill
    return res
import os
out = {}
for city in ['Vancouver', 'Calgary']:
    for subset in ['full', 'core']:
        if f'{city}_{subset}' in out: continue
        r = run(city, subset); out[f'{city}_{subset}'] = r
        print(city, subset, 'working', r['working'])
        for y, v in r['increment']['years'].items(): print('  ', y, v['NADstar'], 'z', v['z'], 'P', round(v['P'], 5), 'Holm', round(v['P_holm'], 4), '| detr z', r['detrended']['years'][y]['z'])
        print('  windows', r['increment']['windows'])
        for d, v in r['drill'].items(): print('  ', d, v['year'], v['D'], v['top_nodes'][:2])
        json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'city_replication.json'), 'w'), indent=1)
