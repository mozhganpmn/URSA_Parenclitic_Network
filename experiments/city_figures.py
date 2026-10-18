# Domain and node scores relative to the activation threshold (std=True in parenclitic_core.scores),
# the top-10 node tables for significant city-years, and Figs 5 and 6 of the manuscript
# (run from the repository root: python experiments/city_figures.py).
import sys, os, json, numpy as np, pandas as pd
import matplotlib as mpl; mpl.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.join(HERE, '..'); sys.path.insert(0, REPO)
import parenclitic_core as pc
sys.path.insert(0, HERE); import domain_matrix_panels as dmp
DM = dmp.load(os.path.join(REPO, 'results', 'domain_matrix.json'))
DATA = os.path.join(REPO, 'data'); RES = os.path.join(REPO, 'results'); FIG_DIR = os.path.join(RES, 'figures'); os.makedirs(FIG_DIR, exist_ok=True)
RY = list(range(2015, 2025))
FIG_BLUE, FIG_ORANGE = '#2a78d6', '#eb6834'; FIG_GREY, FIG_LGREY, FIG_INK, FIG_INK2 = '#8a8985', '#d9d8d4', '#0b0b0b', '#52514e'; MM = 1/25.4
FIG_RC = {'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'], 'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6.5, 'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'axes.edgecolor': FIG_INK2, 'axes.labelcolor': FIG_INK, 'xtick.color': FIG_INK2, 'ytick.color': FIG_INK2,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#ecebe8', 'grid.linewidth': 0.5, 'axes.axisbelow': True,
    'lines.linewidth': 1.4, 'lines.markersize': 3.5, 'legend.frameon': False, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 400, 'figure.dpi': 150}
SEQ = LinearSegmentedColormap.from_list('seqblue', ['#f2f7fe', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'])
def letter(ax, s, x=-0.18, y=1.06): ax.text(x, y, s, transform=ax.transAxes, fontsize=8, fontweight='bold', va='bottom', ha='left', color=FIG_INK)
def save(fig, name):
    for ext in ('pdf', 'png'): fig.savefig(os.path.join(FIG_DIR, f'{name}.{ext}'), bbox_inches='tight', dpi=400)
LONG = {'Knowledge Dissemination and Management': 'Knowledge Dissemination', 'Effective Provision of Critical Services': 'Critical Services',
        'Diverse Livelihood and Employment': 'Livelihood and Employment', 'Innovation and Entrepreneurship': 'Innovation',
        'Integrated Development Planning': 'Development Planning', 'Community Engagement & Preparedness': 'Community Engagement'}
def short(s):
    s = LONG.get(s, s)
    for tail in [', Vancouver Public Library', ', Vancouver CMA', ', BC Hydro', ' (in-person + virtual)', ', City of Vancouver', ', Metro Vancouver', ', British Columbia', ', Calgary CMA', ', Calgary Public Library']:
        s = s.replace(tail, '')
    s = s.replace(' (%)', '').strip(); s = s[0].upper() + s[1:].replace('Days', 'days').replace('Rate', 'rate').replace('Resolution', 'resolution').replace('Case Conclusion', 'case conclusion')
    return s if len(s) <= 34 else s[:32] + '…'
def dumbbell(ax, rows, xlabel):
    for k, (lab, v, m) in enumerate(rows):
        ax.plot([m, v], [k, k], color=FIG_LGREY, lw=2.2, solid_capstyle='round', zorder=1); ax.scatter(m, k, s=14, color=FIG_GREY, zorder=2); ax.scatter(v, k, s=20, color=FIG_BLUE, zorder=3)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([short(r[0]) for r in rows]); ax.set_ylim(-0.7, len(rows) - 0.3); ax.grid(axis='y', visible=False); ax.set_xlabel(xlabel)
H = pd.read_excel(os.path.join(DATA, 'Input.xlsx'), sheet_name='Harmonization').set_index('Row')
def prep(city):
    full = pd.read_excel(os.path.join(DATA, {'Vancouver': 'Input.xlsx', 'Calgary': 'Input_Calgary.xlsx'}[city]), sheet_name='Sheet1'); yc = sorted([c for c in full.columns if isinstance(c, (int, np.integer))])
    V = full[yc].apply(pd.to_numeric, errors='coerce').values.astype(float); keep = np.isfinite(V).sum(1) >= 3; x = full[keep].reset_index(drop=True); V = V[keep]
    return full, x, V, yc, pc.Pairs(V.shape[0]), full.assign(k=keep).groupby('Indicator')['k'].mean().to_dict(), full.groupby('Indicator')['Domain'].first().to_dict(), keep
def oth(arr, yi, y): return float(np.nanmedian(arr[[yi[t] for t in RY if t != y]]))
out = {}; C = {}
for city in ['Vancouver', 'Calgary']:
    full, x, V, yc, pr, cov_w, dom_of, keep = prep(city); yi = {y: k for k, y in enumerate(yc)}
    node, (inds, ind), (doms, dom) = pc.scores(V, pr, list(x['Indicator']), cov_w, dom_of, std=True)
    nraw, _, _ = pc.scores(V, pr, list(x['Indicator']), cov_w, dom_of, std=False)
    C[city] = dict(full=full, x=x, node=node, inds=inds, ind=ind, doms=doms, dom=dom, yi=yi, dom_of=dom_of, yc=yc)
    dd = pd.DataFrame(dom, index=doms, columns=yc)[RY]
    r = dict(domain_scores={d: {int(y): round(float(v), 3) for y, v in dd.loc[d].items()} for d in doms}, domain_peak={d: [int(dd.loc[d].idxmax()), round(float(dd.loc[d].max()), 3)] for d in doms})
    for y in ([2020, 2023] if city == 'Vancouver' else [2015, 2020, 2023]):
        j = yi[y]; ex = node[:, j] - np.array([oth(node[k], yi, y) if np.isfinite(node[k]).any() else np.nan for k in range(len(x))]); order = [int(k) for k in np.argsort(-np.where(np.isfinite(ex), ex, -9))[:10]]
        exr = nraw[:, j] - np.array([oth(nraw[k], yi, y) if np.isfinite(nraw[k]).any() else np.nan for k in range(len(x))]); top_r = set(int(k) for k in np.argsort(-np.where(np.isfinite(exr), exr, -9))[:10])
        rows = []
        for k in order:
            rid = int(full.index[keep][k]) + 1; v = V[k]
            rows.append(dict(sub=str(x.loc[k, 'Subindicator']).strip(), domain=str(x.loc[k, 'Domain']), node=round(float(node[k, j]), 2), median_other=round(oth(node[k], yi, y), 2),
                value_prev=None if not np.isfinite(v[j-1]) else float(v[j-1]), value=None if not np.isfinite(v[j]) else float(v[j]), status=str(H.loc[rid, 'Status']).split(':')[0], geography=None if pd.isna(H.loc[rid, 'Geography']) else str(H.loc[rid, 'Geography'])))
        r[f'top10_{y}'] = dict(rows=rows, overlap_with_sd_unit_scores=len(top_r & set(order)))
    out[city] = r
json.dump(out, open(os.path.join(RES, 'city_figure_numbers.json'), 'w'), indent=1)
# ---------------- Fig. 5: domains (Vancouver) ----------------
c = C['Vancouver']; dd = pd.DataFrame(c['dom'], index=c['doms'], columns=c['yc'])[RY]
cand = dd[[y for y in RY if y > 2015]]; DD = cand.max(axis=1).idxmax(); DY = int(cand.loc[DD].idxmax()); j = c['yi'][DY]
ind_rows = sorted([(n, float(c['ind'][k, j]), oth(c['ind'][k], c['yi'], DY)) for k, n in enumerate(c['inds']) if c['dom_of'][n] == DD and np.isfinite(c['ind'][k, j])], key=lambda r: r[1] - r[2])
sub_rows = sorted([(str(c['x'].loc[k, 'Subindicator']).strip(), float(c['node'][k, j]), oth(c['node'][k], c['yi'], DY)) for k in range(len(c['x'])) if c['dom_of'][c['x'].loc[k, 'Indicator']] == DD and np.isfinite(c['node'][k, j])], key=lambda r: r[1] - r[2])[-6:]
print('Fig 5 target', DD, DY, round(float(dd.loc[DD, DY]), 3)); print([(r[0], round(r[1], 2), round(r[2], 2)) for r in ind_rows]); print([(r[0], round(r[1], 2), round(r[2], 2)) for r in sub_rows])
with mpl.rc_context(FIG_RC):
    fig = plt.figure(figsize=(183*MM, 140*MM)); og = fig.add_gridspec(2, 1, height_ratios=[74, 58], hspace=0.42); gs = og[0].subgridspec(2, 4, width_ratios=[1.15, 0.045, 0.95, 1.0], height_ratios=[0.8, 1.2], wspace=0.08, hspace=0.75)
    a = fig.add_subplot(gs[:, 0]); im = a.imshow(dd.values, aspect='auto', cmap=SEQ, vmin=np.nanmin(dd.values), vmax=np.nanmax(dd.values))
    a.set_xticks(range(len(RY))); a.set_xticklabels([str(y)[2:] for y in RY]); a.set_yticks(range(len(dd.index))); a.set_yticklabels(dd.index); a.grid(False); a.set_xlabel('Year (20xx)')
    for d in dd.index:
        yr = int(dd.loc[d].idxmax()); a.add_patch(plt.Rectangle((RY.index(yr)-0.5, list(dd.index).index(d)-0.5), 1, 1, fill=False, ec='white', lw=1.2))
    for s_ in a.spines.values(): s_.set_visible(False)
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[:, 1])); cb.outline.set_visible(False); cb.set_label('Domain score $D_d(t)$', labelpad=2); cb.ax.tick_params(labelsize=6); letter(a, 'a', x=-0.55)
    b = fig.add_subplot(gs[0, 3]); dumbbell(b, ind_rows, f'Indicator score $A_k$ ({DD}, {DY})'); b.text(1.0, 1.04, f'blue, {DY}; grey, median of other years', transform=b.transAxes, ha='right', fontsize=6, color=FIG_INK2); letter(b, 'b', x=-0.78)
    cc = fig.add_subplot(gs[1, 3]); dumbbell(cc, sub_rows, f'Node score $A_i$ (top sub-indicators, {DY})'); letter(cc, 'c', x=-0.78)
    bg = og[1].subgridspec(1, 4, width_ratios=[1, 1, 0.04, 1.25], wspace=0.5)
    for q, (yy, lt) in enumerate([(2020, 'd'), (2023, 'e')]):
        axm = fig.add_subplot(bg[0, q]); imm = dmp.draw(axm, DM, 'Vancouver', yy); letter(axm, lt, x=-0.3, y=1.22)
    cbm = fig.colorbar(imm, cax=fig.add_subplot(bg[0, 2])); cbm.outline.set_visible(False); cbm.ax.tick_params(labelsize=6)
    cbm.set_label('Observed minus null mean', labelpad=2, fontsize=6)
    kt = fig.add_subplot(bg[0, 3]); kt.set_axis_off(); kt.text(0.15, 0.95, 'Cell: observed share of activated pairs\nbetween two domains minus its mean under\nrefitted surrogates; number, $z$ score.\nHatched: fewer than 30 eligible pairs.\nDescriptive display, not a test.', fontsize=5.8, color=FIG_INK2, va='top', linespacing=1.35)
    save(fig, 'Fig5_domains'); plt.close(fig)
# ---------------- Fig. 6: Calgary ----------------
cr = json.load(open(os.path.join(HERE, 'city_replication.json'))); c = C['Calgary']; cd = pd.DataFrame(c['dom'], index=c['doms'], columns=c['yc'])[RY]
j = c['yi'][2015]; crow = sorted([(str(c['x'].loc[k, 'Subindicator']).strip(), float(c['node'][k, j]), oth(c['node'][k], c['yi'], 2015)) for k in range(len(c['x'])) if np.isfinite(c['node'][k, j])], key=lambda r: r[1] - r[2])[-6:]
print('Fig 6c', [(r[0], round(r[1], 2), round(r[2], 2)) for r in crow][::-1])
with mpl.rc_context(FIG_RC):
    fig = plt.figure(figsize=(183*MM, 132*MM)); og = fig.add_gridspec(2, 1, height_ratios=[66, 58], hspace=0.5); gs = og[0].subgridspec(1, 6, width_ratios=[1.1, 0.95, 1.05, 0.05, 1.1, 0.95], wspace=0.04)
    a = fig.add_subplot(gs[0, 0]); a.axhline(0, color=FIG_INK2, lw=0.6)
    for key, col, ls, lw in [('Vancouver_full', FIG_BLUE, '-', 1.8), ('Calgary_full', FIG_ORANGE, '-', 1.8), ('Vancouver_core', FIG_BLUE, ':', 1.0), ('Calgary_core', FIG_ORANGE, ':', 1.0)]:
        yr = cr[key]['increment']['years']; z = [yr[str(y)]['z'] for y in RY]; a.plot(RY, z, ls, color=col, lw=lw, zorder=2)
        if 'full' in key:
            for y, zz in zip(RY, z): a.plot(y, zz, 'o', color=col if yr[str(y)]['P_holm'] < 0.05 else 'white', mec=col, mew=1.0, ms=4, zorder=3)
    a.text(2015.1, 9.3, 'Vancouver', color=FIG_BLUE, fontsize=6.3, fontweight='bold'); a.text(2015.1, 8.2, 'Calgary', color=FIG_ORANGE, fontsize=6.3, fontweight='bold'); a.text(2015.1, 6.5, 'dotted:\nmatched core', color=FIG_INK2, fontsize=5.4, va='bottom')
    a.set_xticks([2015, 2019, 2023]); a.set_xlim(2014.6, 2024.4); a.set_ylabel('NAD relative to null ($z$)'); a.set_xlabel('Year'); letter(a, 'a', x=-0.25)
    b = fig.add_subplot(gs[0, 2]); im = b.imshow(cd.values, aspect='auto', cmap=SEQ, vmin=np.nanmin(cd.values), vmax=np.nanmax(cd.values))
    b.set_xticks(range(len(RY))); b.set_xticklabels([str(y)[2:] for y in RY], fontsize=5.8); b.set_yticks(range(len(cd.index))); b.set_yticklabels(cd.index); b.grid(False); b.set_xlabel('Year (20xx)')
    for d in cd.index: b.add_patch(plt.Rectangle((RY.index(int(cd.loc[d].idxmax()))-0.5, list(cd.index).index(d)-0.5), 1, 1, fill=False, ec='white', lw=1.2))
    for s_ in b.spines.values(): s_.set_visible(False)
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[0, 3])); cb.outline.set_visible(False); cb.ax.set_title('Domain\nscore', fontsize=6, pad=3); cb.ax.tick_params(labelsize=6); letter(b, "b", x=-1.0)
    cc = fig.add_subplot(gs[0, 5]); dumbbell(cc, crow, 'Node score $A_i$ (Calgary, 2015)'); letter(cc, "c", x=-0.6)
    bg = og[1].subgridspec(1, 4, width_ratios=[1, 1, 0.04, 1.25], wspace=0.5)
    for q, (yy, lt) in enumerate([(2015, 'd'), (2023, 'e')]):
        axm = fig.add_subplot(bg[0, q]); imm = dmp.draw(axm, DM, 'Calgary', yy); letter(axm, lt, x=-0.3, y=1.22)
    cbm = fig.colorbar(imm, cax=fig.add_subplot(bg[0, 2])); cbm.outline.set_visible(False); cbm.ax.tick_params(labelsize=6)
    cbm.set_label('Observed minus null mean', labelpad=2, fontsize=6)
    kt = fig.add_subplot(bg[0, 3]); kt.set_axis_off(); kt.text(0.15, 0.95, 'Cell: observed share of activated pairs\nbetween two domains minus its mean under\nrefitted surrogates; number, $z$ score.\nHatched: fewer than 30 eligible pairs.\nDescriptive display, not a test.', fontsize=5.8, color=FIG_INK2, va='top', linespacing=1.35)
    save(fig, 'Fig6_calgary'); plt.close(fig)
print('saved')
