# Checks added in response to review: internal observation gaps, NAD test restricted to series without internal gaps,
# and the composition of the 2023 anomaly network restricted to pairs eligible in both 2022 and 2023.
# Usage: python experiments/review_checks.py OUT.json
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
    V = full[yc].apply(pd.to_numeric, errors='coerce').values.astype(float).copy(); keep = np.isfinite(V).sum(1) >= 3
    return full[keep].reset_index(drop=True), yc, V[keep]
def gaps(V):
    ng = 0; glen = []; inc_all = 0; inc_gap = 0
    for row in V:
        idx = np.where(np.isfinite(row))[0]; d = np.diff(idx); inc_all += len(d); inc_gap += int((d > 1).sum())
        if (d > 1).any(): ng += 1; glen += [int(k - 1) for k in d if k > 1]
    return dict(series=int(V.shape[0]), series_with_internal_gaps=ng, share=round(ng/V.shape[0], 3), n_gaps=len(glen),
                gap_length_counts={int(k): int(glen.count(k)) for k in sorted(set(glen))}, increments=inc_all, increments_spanning_gap=inc_gap, share_increments=round(inc_gap/inc_all, 3))
def test(V, yc):
    pr = Pairs(V.shape[0]); E = NadEngine(V, pr); obs, ne = E.nad(V); draw = make_increment_factory(V); rng = np.random.default_rng(20231)
    null = np.array([E.nad(draw(rng))[0] for _ in range(N)]); yi = {y: k for k, y in enumerate(yc)}; r = {}
    for y in REP:
        j = yi[y]; nj = null[:, j]; r[y] = dict(NAD=float(obs[j]), z=float((obs[j]-nj.mean())/nj.std(ddof=1)), P=(int((nj >= obs[j]-1e-12).sum())+1)/(N+1))
    ph = holm({y: v['P'] for y, v in r.items()})
    for y in r: r[y]['P_holm'] = ph[y]
    return dict(working=int(V.shape[0]), years=r)
out = {}
for city, fn in [('Vancouver', 'Input.xlsx'), ('Calgary', 'Input_Calgary.xlsx')]:
    x, yc, V = load(fn); g = gaps(V); out[city] = dict(gaps=g); print(city, g, flush=True)
    nogap = np.array([not (np.diff(np.where(np.isfinite(r))[0]) > 1).any() for r in V]); print(city, 'series without internal gaps', int(nogap.sum()), flush=True)
    out[city]['nogap'] = test(V[nogap], yc); print(city, {y: (round(d['z'], 2), round(d['P_holm'], 3)) for y, d in out[city]['nogap']['years'].items()}, flush=True)
    json.dump(out, open(OUT, 'w'), indent=1)
# 2023 composition (Vancouver), pairs eligible in both years
x, yc, V = load('Input.xlsx'); pr = Pairs(V.shape[0]); S = lfo_stats(V, pr); A = V[pr.ia]; B = V[pr.ib]; w = weights(S, A, B)
el = S['ok'] & np.isfinite(A) & np.isfinite(B); act = el & (w >= S['th']); j22, j23 = yc.index(2022), yc.index(2023)
both = el[:, j22] & el[:, j23]; a23 = act[:, j23]
res = dict(active_2023=int(a23.sum()), active_2023_eligible_both=int((a23 & both).sum()), newly_eligible_active_2023=int((a23 & ~el[:, j22]).sum()),
           inactive_in_2022_among_eligible_both=int((a23 & both & ~act[:, j22]).sum()))
res['share_inactive_2022_of_eligible_both'] = round(res['inactive_in_2022_among_eligible_both']/max(res['active_2023_eligible_both'], 1), 3)
res['share_old_definition'] = round(int((a23 & ~act[:, j22]).sum())/int(a23.sum()), 3)
dom = x['Domain'].values; new = a23 & both & ~act[:, j22]; hits = {}
for d in np.unique(dom): hits[d] = int(((dom[pr.ia] == d) & new).sum() + ((dom[pr.ib] == d) & new).sum())
res['endpoint_counts_new_edges'] = hits; res['per_node'] = {d: round(hits[d]/(dom == d).sum(), 1) for d in hits}
out['new_edges_2023'] = res; print(res, flush=True); json.dump(out, open(OUT, 'w'), indent=1)
