# Domain peak years for Vancouver under CV cutoffs 0, 0.5, 1, 2% and support scales lambda = 3, 5, 10,
# using threshold-relative scores (std=True), the convention of the manuscript.
import sys, os, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.join(HERE, '..'); sys.path.insert(0, REPO); sys.path.insert(0, HERE)
import parenclitic_core as pc
import importlib.util
spec = importlib.util.spec_from_file_location('cf_prep', os.path.join(HERE, 'city_figures.py'))
src = open(os.path.join(HERE, 'city_figures.py')).read().split('def prep(city):')[0]
ns = {'__file__': os.path.join(HERE, 'city_figures.py')}; exec(compile(src, 'cf', 'exec'), ns)
full = pd.read_excel(os.path.join(REPO, 'data', 'Input.xlsx'), sheet_name='Sheet1'); yc = sorted([c for c in full.columns if isinstance(c, (int, np.integer))])
V = full[yc].apply(pd.to_numeric, errors='coerce').values.astype(float); keep = np.isfinite(V).sum(1) >= 3; x = full[keep].reset_index(drop=True); V = V[keep]
pr = pc.Pairs(V.shape[0]); cov_w = full.assign(k=keep).groupby('Indicator')['k'].mean().to_dict(); dom_of = full.groupby('Indicator')['Domain'].first().to_dict()
yi = {y: k for k, y in enumerate(yc)}; RY = list(range(2015, 2025)); out = {}
def run(cv, lam):
    pc.LAM = lam
    node, (inds, ind), (doms, dom) = pc.scores(V, pr, list(x['Indicator']), cov_w, dom_of, std=True, cv=cv)
    dd = pd.DataFrame(dom, index=doms, columns=yc)[RY]
    peaks = {d: int(dd.loc[d].idxmax()) for d in doms}; top = dd.stack().idxmax()
    return dict(domain_peak_years=peaks, n_2020=sum(1 for v in peaks.values() if v == 2020), largest=[top[0], int(top[1]), round(float(dd.stack().max()), 3)])
for cv in [0.0, 0.005, 0.01, 0.02]: out[f'cv_{cv}'] = run(cv, 5.0)
for lam in [3.0, 5.0, 10.0]: out[f'lambda_{lam}'] = run(0.01, lam)
json.dump(out, open(os.path.join(REPO, 'results', 'cv_lambda_domains.json'), 'w'), indent=1)
for k, v in out.items(): print(k, v['n_2020'], v['largest'], {d[:6]: y for d, y in v['domain_peak_years'].items() if y != 2020})
