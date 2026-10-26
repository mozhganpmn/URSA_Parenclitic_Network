"""Domain peak years (2015-2024) under three cap conventions for the threshold-relative scores.
A = manuscript (cap 20 on w, then divide by w*), B = cap applied after normalisation, min(w/w*, 20), C = no cap.
Run from the repository root: python experiments/cap_variants_domains.py [OUT.json]"""
import sys, json, inspect, numpy as np, pandas as pd
root = '.'; sys.path.insert(0, root)
import parenclitic_core as pc
src = inspect.getsource(pc.scores)
old = "w = np.minimum(weights(S, A, B), CAP)\n    if std: w = w/S['th']"
assert old in src
variants = {
 'A_cap_before_division': old,
 'B_cap_after_division': "w = weights(S, A, B)\n    if std: w = np.minimum(w/S['th'], CAP)",
 'C_no_cap': "w = weights(S, A, B)\n    if std: w = w/S['th']"}
RY = list(range(2015, 2025)); out = {}
for city, f in [('Vancouver', 'Input.xlsx'), ('Calgary', 'Input_Calgary.xlsx')]:
    full = pd.read_excel(f'{root}/data/{f}', sheet_name='Sheet1'); yc = sorted([c for c in full.columns if isinstance(c, (int, np.integer))])
    V = full[yc].apply(pd.to_numeric, errors='coerce').values.astype(float); keep = np.isfinite(V).sum(1) >= 3
    x = full[keep].reset_index(drop=True); V = V[keep]; pr = pc.Pairs(V.shape[0])
    cov_w = full.assign(k=keep).groupby('Indicator')['k'].mean().to_dict(); dom_of = full.groupby('Indicator')['Domain'].first().to_dict()
    out[city] = {'working_set': int(V.shape[0])}
    for name, rep in variants.items():
        ns = dict(pc.__dict__); exec(src.replace(old, rep), ns)
        node, (inds, ind), (doms, dom) = ns['scores'](V, pr, list(x['Indicator']), cov_w, dom_of, std=True)
        dd = pd.DataFrame(dom, index=doms, columns=yc)[RY]
        peaks = {d: int(dd.loc[d].idxmax()) for d in doms}; top = dd.stack().idxmax()
        nd = pd.DataFrame(node, columns=yc)[[2020]]
        out[city][name] = dict(domain_peak_years=peaks, n_peak_2020=sum(v == 2020 for v in peaks.values()), largest=[top[0], int(top[1]), round(float(dd.stack().max()), 3)],
                               domain_max={d: round(float(dd.loc[d].max()), 3) for d in doms})
json.dump(out, open((sys.argv[1] if len(sys.argv) > 1 else 'results/cap_variants_domains.json'), 'w'), indent=1)
for c in out:
    for k, v in out[c].items():
        if isinstance(v, dict): print(c, k, v['n_peak_2020'], v['largest'], {d[:8]: y for d, y in v['domain_peak_years'].items() if y != 2020})
