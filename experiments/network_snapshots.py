"""Indicator-level anomaly network of Vancouver for 2020 and 2023 (descriptive).
Nodes: the 40 indicators, grouped by domain on a circle; size = indicator score A_k (threshold-relative, manuscript convention).
Edges: indicator pairs whose sub-indicator pairs activated at the F threshold; drawn when at least MINACT pairs activated and the
activated share of eligible pairs is at least MINSHARE. Width = activated share. Run from the repository root: python experiments/network_snapshots.py [OUTDIR]"""
import sys, os, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
root = '.'; outdir = sys.argv[1] if len(sys.argv) > 1 else 'results/figures'; sys.path.insert(0, root)
import parenclitic_core as pc
MINACT, MINSHARE, YEARS = 3, 0.5, [2020, 2023]
full = pd.read_excel(f'{root}/data/Input.xlsx', sheet_name='Sheet1'); yc = sorted([c for c in full.columns if isinstance(c, (int, np.integer))])
V = full[yc].apply(pd.to_numeric, errors='coerce').values.astype(float); keep = np.isfinite(V).sum(1) >= 3
x = full[keep].reset_index(drop=True); V = V[keep]; pr = pc.Pairs(V.shape[0])
cov_w = full.assign(k=keep).groupby('Indicator')['k'].mean().to_dict(); dom_of = full.groupby('Indicator')['Domain'].first().to_dict()
ind_of = list(x['Indicator'])
node, (inds, ind), (doms, dom) = pc.scores(V, pr, ind_of, cov_w, dom_of, std=True)
S = pc.lfo_stats(V, pr); A = V[pr.ia]; B = V[pr.ib]; el = S['ok'] & np.isfinite(A) & np.isfinite(B); act = el & (pc.weights(S, A, B) >= S['th'])
ii = {k: n for n, k in enumerate(inds)}; ka = np.array([ii[ind_of[i]] for i in pr.ia]); kb = np.array([ii[ind_of[i]] for i in pr.ib])
lo = np.minimum(ka, kb); hi = np.maximum(ka, kb); K = len(inds)
names = pd.read_csv(f'{root}/data/names.csv', encoding='utf-8-sig'); ab = dict(zip(names['Indicator name'], names['Abbr']))
order = sorted(range(K), key=lambda k: (dom_of[inds[k]], inds[k])); pos = {}; ang = {}
gap = 0.12; n = len(order); doms_o = [dom_of[inds[k]] for k in order]
th = np.linspace(0, 2*np.pi, n, endpoint=False)
cur = 0.0; pos_idx = {}
for r, k in enumerate(order):
    if r > 0 and doms_o[r] != doms_o[r-1]: cur += gap
    pos_idx[k] = cur; cur += (2*np.pi - gap*(len(set(doms_o)))) / n
ang = {k: pos_idx[k] for k in order}
palette = dict(zip(sorted(set(doms_o)), ['#4C78A8', '#F58518', '#54A24B', '#E45756', '#B279A2', '#9D755D', '#72B7B2']))
res = {}; fig, axs = plt.subplots(1, 2, figsize=(13, 6.8))
for ax, y in zip(axs, YEARS):
    j = yc.index(y); cell = lo*K + hi
    Ea = np.bincount(cell, weights=act[:, j].astype(float), minlength=K*K).reshape(K, K); El = np.bincount(cell, weights=el[:, j].astype(float), minlength=K*K).reshape(K, K)
    share = np.where(El > 0, Ea/np.maximum(El, 1), 0.0); edges = [(a, b) for a in range(K) for b in range(a+1, K) if Ea[a, b] >= MINACT and share[a, b] >= MINSHARE]
    nodes_sc = ind[:, j]; sc = np.nan_to_num(nodes_sc, nan=0.0)
    xy = {k: (np.cos(ang[k]), np.sin(ang[k])) for k in order}
    for a, b in edges:
        same = dom_of[inds[a]] == dom_of[inds[b]]
        ax.plot([xy[a][0], xy[b][0]], [xy[a][1], xy[b][1]], color=('#D4A017' if same else '#9A9A9A'), lw=0.5 + 3.0*share[a, b], alpha=0.45, zorder=1)
    for k in order: ax.scatter(*xy[k], s=40 + 330*sc[k], color=palette[dom_of[inds[k]]], edgecolor='white', lw=0.6, zorder=3)
    top = np.argsort(-sc)[:6]
    for k in top: ax.annotate(ab.get(inds[k], inds[k]), xy[k], xytext=(xy[k][0]*1.17, xy[k][1]*1.17), ha='center', va='center', fontsize=8.5, weight='bold')
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4); ax.set_aspect('equal'); ax.axis('off')
    nad = act[:, j].sum()/el[:, j].sum()
    ax.set_title(f'{y}   NAD = {nad:.3f}   ({len(edges)} indicator pairs shown)', fontsize=11, y=0.99)
    res[y] = dict(NAD=float(nad), eligible_pairs=int(el[:, j].sum()), activated_pairs=int(act[:, j].sum()), edges_shown=len(edges),
                  top_nodes=[(inds[k], round(float(sc[k]), 2)) for k in top], max_score=round(float(sc.max()), 2))
from matplotlib.lines import Line2D
h = [Line2D([0], [0], marker='o', ls='', color=palette[d], label=d, markersize=8) for d in sorted(palette)] + [Line2D([0], [0], color='#D4A017', lw=2, label='within domain'), Line2D([0], [0], color='#9A9A9A', lw=2, label='between domains')]
fig.legend(handles=h, loc='lower center', ncol=5, frameon=False, fontsize=9); fig.tight_layout(rect=(0, 0.08, 1, 0.97))
os.makedirs(outdir, exist_ok=True)
for ext in ('pdf', 'png'): fig.savefig(f'{outdir}/FigS2_networks_2020_2023.{ext}', dpi=200)
json.dump(dict(params=dict(MINACT=MINACT, MINSHARE=MINSHARE), years=res), open(f'{outdir}/FigS2_numbers.json', 'w'), indent=1); print(json.dumps(res, indent=1))
