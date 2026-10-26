# Effect of the three data corrections of 22 Oct 2026 on the NAD test (2000 refitted increment-shift surrogates, one seed).
# Usage: python experiments/data_fix_sensitivity.py OUT.json
import sys, os, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from parenclitic_core import *
OUT = sys.argv[1]; N = 2000; REP = list(range(2015, 2025)); D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
def holm(ps):
    keys = sorted(ps, key=ps.get); m = len(keys); out = {}; prev = 0.0
    for r, k in enumerate(keys): prev = min(1.0, max(prev, (m - r)*ps[k])); out[k] = prev
    return out
def load(fn):
    full = pd.read_excel(os.path.join(D, fn), sheet_name='Sheet1'); yc = sorted([c for c in full.columns if isinstance(c, (int, np.integer))])
    return full, yc, full[yc].apply(pd.to_numeric, errors='coerce').values.astype(float).copy()
def test(V, yc):
    V = V[np.isfinite(V).sum(1) >= 3]; pr = Pairs(V.shape[0]); E = NadEngine(V, pr); obs, ne = E.nad(V); draw = make_increment_factory(V); rng = np.random.default_rng(20231)
    null = np.array([E.nad(draw(rng))[0] for _ in range(N)]); yi = {y: k for k, y in enumerate(yc)}; r = {}
    for y in REP:
        j = yi[y]; nj = null[:, j]; r[y] = dict(NAD=float(obs[j]), z=float((obs[j]-nj.mean())/nj.std(ddof=1)), P=(int((nj >= obs[j]-1e-12).sum())+1)/(N+1))
    ph = holm({y: v['P'] for y, v in r.items()})
    for y in r: r[y]['P_holm'] = ph[y]
    return dict(working=int(V.shape[0]), years=r)
out = {}
fV, yc, V = load('Input_orig217.xlsx'); out['Vancouver_original'] = test(V, yc)
fV, yc, V = load('Input.xlsx'); out['Vancouver_UGS2021_removed'] = test(V, yc)
iu = list(fV['Subindicator']).index('Amount of Urban Green Space (UGS)'); V2 = V.copy(); V2[iu, yc.index(2021)] = 11.43; out['Vancouver_UGS2021_11.43'] = test(V2, yc)
fC, yc, C = load('Input_Calgary_orig217.xlsx'); out['Calgary_original'] = test(C, yc)
fC, yc, C = load('Input_Calgary.xlsx'); out['Calgary_duplicates_removed'] = test(C, yc)
json.dump(out, open(OUT, 'w'), indent=1)
for k, v in out.items():
    print(k, v['working'], {y: (round(d['z'], 2), round(d['P_holm'], 3)) for y, d in v['years'].items() if y in (2015, 2020, 2023, 2024)})
