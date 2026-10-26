# Second review round: effective series participation, sensitivity to near-duplicate representations of the same activity
# and to explicitly event-coded inputs. Usage: python experiments/review_checks2.py OUT.json
import sys, os, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import parenclitic_core as pc
OUT = sys.argv[1]; N = 2000; REP = list(range(2015, 2025)); D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
def holm(ps):
    keys = sorted(ps, key=ps.get); m = len(keys); out = {}; prev = 0.0
    for r, k in enumerate(keys): prev = min(1.0, max(prev, (m - r)*ps[k])); out[k] = prev
    return out
DUP = ['Library visits per capita, Vancouver Public Library', 'Library circulation per capita, Vancouver Public Library', 'Library visits per capita, Calgary Public Library',
       'Library circulation per capita, Calgary Public Library', 'GDP per capita, Vancouver CMA', 'GDP per capita, Calgary CMA', 'Per Capita Regional GDP',
       'Regional GHG emissions per capita', 'Number of transit trips']
EVENT = ['Occurence of shocks', 'Public emergency alert messages issued']
out = {}
for city, fn in [('Vancouver', 'Input.xlsx'), ('Calgary', 'Input_Calgary.xlsx')]:
    full = pd.read_excel(os.path.join(D, fn), sheet_name='Sheet1'); yc = sorted([c for c in full.columns if isinstance(c, (int, np.integer))]); yi = {y: k for k, y in enumerate(yc)}
    Vall = full[yc].apply(pd.to_numeric, errors='coerce').values.astype(float); nobs = np.isfinite(Vall).sum(1)
    names = full['Subindicator'].astype(str).str.strip().values
    def run(drop, label):
        keep = (nobs >= 3) & ~np.isin(names, drop); V = Vall[keep]; x = full[keep].reset_index(drop=True); pr = pc.Pairs(V.shape[0])
        E = pc.NadEngine(V, pr); obs, ne = E.nad(V); draw = pc.make_increment_factory(V); rng = np.random.default_rng(20231)
        null = np.array([E.nad(draw(rng))[0] for _ in range(N)]); r = {}
        for y in REP:
            j = yi[y]; nj = null[:, j]; r[y] = dict(z=float((obs[j]-nj.mean())/nj.std(ddof=1)), P=(int((nj >= obs[j]-1e-12).sum())+1)/(N+1))
        ph = holm({y: v['P'] for y, v in r.items()})
        for y in r: r[y]['P_holm'] = ph[y]
        cov_w = full.assign(k=keep).groupby('Indicator')['k'].mean().to_dict(); dom_of = full.groupby('Indicator')['Domain'].first().to_dict()
        node, (inds, ind), (doms, dom) = pc.scores(V, pr, list(x['Indicator']), cov_w, dom_of, std=True)
        dd = pd.DataFrame(dom, index=doms, columns=yc)[REP]
        peaks = {d: int(dd.loc[d].idxmax()) for d in doms}
        res = dict(label=label, working=int(V.shape[0]), years={int(y): v for y, v in r.items()}, domain_peak_year=peaks,
                   n_domains_peak_2020=int(sum(1 for v in peaks.values() if v == 2020)), D_2020={d: round(float(dd.loc[d, 2020]), 3) for d in doms})
        print(city, label, V.shape[0], {y: (round(v['z'], 2), round(v['P_holm'], 3)) for y, v in r.items() if v['P_holm'] < 0.05}, 'domains peaking 2020:', res['n_domains_peak_2020'], flush=True)
        return res
    # participation in the main working set
    keep = nobs >= 3; V = Vall[keep]; pr = pc.Pairs(V.shape[0]); S = pc.lfo_stats(V, pr); A = V[pr.ia]; B = V[pr.ib]
    el = S['ok'] & np.isfinite(A) & np.isfinite(B); pos = el & (S['n'] > 3); rep_cols = [yi[y] for y in REP]
    def part(mask):
        m = mask[:, rep_cols].any(1); s = np.zeros(V.shape[0], bool); s[pr.ia[m]] = True; s[pr.ib[m]] = True; return s
    pa = part(el); pp = part(pos); cnt = np.isfinite(V).sum(1)
    out[city] = dict(rows=int(len(full)), working=int(V.shape[0]), nad_participation=int(pa.sum()), positive_weight_participation=int(pp.sum()),
                     tier1_ge7=int((cnt >= 7).sum()), tier2_3to6=int(((cnt >= 3) & (cnt <= 6)).sum()), tier2_nad_part=int(pa[(cnt >= 3) & (cnt <= 6)].sum()), tier2_pos_part=int(pp[(cnt >= 3) & (cnt <= 6)].sum()),
                     excluded_lt3=int((nobs < 3).sum()))
    print(city, out[city], flush=True)
    out[city]['sens'] = {}
    for lab, drop in [('main', []), ('one representation per activity', [d for d in DUP if d in names]), ('without event-coded inputs', [e for e in EVENT if e in names]),
                      ('both', [d for d in DUP + EVENT if d in names])]:
        out[city]['sens'][lab] = run(drop, lab); json.dump(out, open(OUT, 'w'), indent=1)
