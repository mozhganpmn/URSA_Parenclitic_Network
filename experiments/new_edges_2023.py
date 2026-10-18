# Composition of the 2023 anomaly network: share of the edges active in 2023 that were inactive in 2022, and the
# domains they touch (Vancouver). Usage: python experiments/new_edges_2023.py .
import sys, os, numpy as np, pandas as pd
REPO = sys.argv[1]; sys.path.insert(0, REPO)
import parenclitic_core as pc
D = os.path.join(REPO, 'data')
f = os.path.join(D, 'Input.xlsx')
full = pd.read_excel(f, sheet_name='Sheet1'); yc = sorted([c for c in full.columns if isinstance(c, (int, np.integer))])
V = full[yc].apply(pd.to_numeric, errors='coerce').values.astype(float); keep = np.isfinite(V).sum(1) >= 3
x = full[keep].reset_index(drop=True); V = V[keep]; pr = pc.Pairs(V.shape[0])
S = pc.lfo_stats(V, pr); A = V[pr.ia]; B = V[pr.ib]; w = pc.weights(S, A, B)
el = S['ok'] & np.isfinite(A) & np.isfinite(B)
act = el & (w >= S['th'])
print('working', V.shape[0], 'possible pairs', V.shape[0]*(V.shape[0]-1)//2)
print('unique pairs with >=1 reference', int(el.any(1).sum()), 'pair-years', int(el.sum()))
for y in [2014, 2015, 2022, 2023]:
    j = yc.index(y); print(y, 'eligible', int(el[:, j].sum()), 'active', int(act[:, j].sum()), 'NAD', round(act[:, j].sum()/el[:, j].sum(), 3))
j22, j23 = yc.index(2022), yc.index(2023)
new = act[:, j23] & ~act[:, j22]; print('2023 active', int(act[:, j23].sum()), 'new vs 2022', int(new.sum()), round(new.sum()/act[:, j23].sum(), 3))
dom = x['Domain'].values
cnt = {}
for d in np.unique(dom):
    m = (dom[pr.ia] == d) | (dom[pr.ib] == d)
    cnt[d] = int((new & m).sum())
print('new edges touching each domain', cnt)
# endpoints counted once per edge
node_hits = {}
for d in np.unique(dom):
    node_hits[d] = int(((dom[pr.ia] == d) & new).sum() + ((dom[pr.ib] == d) & new).sum())
print('endpoint counts', node_hits)
print('norm by nodes', {d: round(node_hits[d]/ (dom==d).sum(),1) for d in node_hits})
