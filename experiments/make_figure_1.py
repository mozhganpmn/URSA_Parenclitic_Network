"""Fig. 1 framework overview with the two-channel schematic (run from the repository root)."""
import json, os
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt
from scipy.stats import f as _fd
from matplotlib.patches import Ellipse, FancyArrowPatch

FIG_BLUE, FIG_ORANGE = '#2a78d6', '#eb6834'
FIG_GREY, FIG_INK, FIG_INK2 = '#8a8985', '#0b0b0b', '#52514e'
MM = 1 / 25.4
FIG_DIR = os.path.join('results', 'figures'); os.makedirs(FIG_DIR, exist_ok=True)
FIG_RC = {
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
    'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7, 'xtick.labelsize': 6.5,
    'ytick.labelsize': 6.5, 'legend.fontsize': 6.5, 'axes.linewidth': 0.5,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.major.size': 2.5,
    'ytick.major.size': 2.5, 'axes.edgecolor': FIG_INK2, 'axes.labelcolor': FIG_INK,
    'xtick.color': FIG_INK2, 'ytick.color': FIG_INK2, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#ecebe8',
    'grid.linewidth': 0.5, 'axes.axisbelow': True, 'lines.linewidth': 1.4,
    'lines.markersize': 3.5, 'legend.frameon': False, 'pdf.fonttype': 42,
    'ps.fonttype': 42, 'savefig.dpi': 400, 'figure.dpi': 150}

def fig_letter(ax, s, x=-0.18, y=1.06):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8, fontweight='bold', va='bottom', ha='left', color=FIG_INK)

def fig_save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG_DIR, f'{name}.{ext}'), bbox_inches='tight', dpi=400)
    print('saved', name)


with mpl.rc_context(FIG_RC):
    _rs = np.random.default_rng(7)
    fig = plt.figure(figsize=(183*MM, 100*MM))
    gs = fig.add_gridspec(2, 4, wspace=0.55, hspace=0.95, left=0.03, right=0.99, bottom=0.04, top=0.9, height_ratios=[1, 1.15])
    # (a) short annual records in a hierarchy
    ax = fig.add_subplot(gs[0, 0]); ax.set_axis_off()
    yrs = np.arange(2011, 2025)
    for k, (lab, col) in enumerate([('sub-indicator $x_i$', FIG_BLUE), ('sub-indicator $x_j$', FIG_ORANGE),
                                    ('sub-indicator $x_k$', FIG_GREY)]):
        base = np.cumsum(_rs.normal(0, 1, len(yrs))); base = (base - base.min())/(np.ptp(base)+1e-9)
        if k == 2: base[:4] = np.nan
        y0 = 2.75 - 1.35*k
        ax.plot(yrs, y0 + 0.6*base, '-o', color=col, lw=1.0, ms=1.8)
        ax.text(2010.6, y0 + 0.78, lab, fontsize=6.2, color=FIG_INK2, ha='left')
    ax.text(2010.6, 4.05, 'domain $\\rightarrow$ indicator $\\rightarrow$ sub-indicator', fontsize=6.2, color=FIG_INK)
    ax.set_xlim(2010.5, 2024.5); ax.set_ylim(-0.1, 4.2)
    ax.text(2017.5, -0.35, 'short annual records (14 years)', ha='center', va='top', fontsize=6.2, color=FIG_INK2)
    fig_letter(ax, 'a', x=-0.02, y=1.12)
    # (b) leave-future-out reference and calibrated threshold
    ax = fig.add_subplot(gs[0, 1])
    n = 9; C = np.array([[1, 0.85], [0.85, 1]])
    tr = _rs.multivariate_normal([0, 0], C, n)
    mu = tr.mean(0); S = np.cov(tr.T, ddof=1)
    w2 = 2*(n+1)*(n-1)/(n*(n-2))*_fd.ppf(0.95, 2, n-2)
    vals, vecs = np.linalg.eigh(S); ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    ax.add_patch(Ellipse(mu, 2*np.sqrt(w2*vals[1]), 2*np.sqrt(w2*vals[0]), angle=ang,
                         fc='#e8f1fc', ec=FIG_BLUE, lw=1.0))
    ax.scatter(tr[:, 0], tr[:, 1], s=9, color=FIG_GREY, zorder=3)
    tgt = mu + np.array([1.9, -1.9])
    ax.scatter(*tgt, s=26, color=FIG_ORANGE, zorder=4, marker='D')
    ax.text(tgt[0]+0.4, tgt[1], 'year $t$', fontsize=6.2, color=FIG_ORANGE, va='center')
    ax.set_xlabel('$x_i$'); ax.set_ylabel('$x_j$'); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    L_ = np.sqrt(w2*vals.max())*0.62 + 0.6; ax.set_xlim(mu[0]-L_, mu[0]+L_); ax.set_ylim(mu[1]-L_, mu[1]+L_); ax.set_aspect('equal')
    ax.text(0.5, -0.2, 'grey, years before $t$ (reference)\nedge if $w_{ij}(t) \\geq w^*(n)$, the exact\n$F(2, n{-}2)$ boundary', transform=ax.transAxes,
            ha='center', va='top', fontsize=6.2, color=FIG_INK2)
    fig_letter(ax, 'b', x=-0.12, y=1.12)
    # (c) anomaly network and hierarchical roll-up
    ax = fig.add_subplot(gs[0, 2]); ax.set_axis_off()
    groups = [(0, 0), (2.4, 0.2), (1.2, 2.1)]; pos = []
    for gi, (cx, cy) in enumerate(groups):
        ax.add_patch(plt.Circle((cx, cy), 0.95, fc='#f3f2ef', ec='none'))
        for k in range(4):
            th = 2*np.pi*k/4 + gi
            pos.append((cx + 0.55*np.cos(th), cy + 0.55*np.sin(th), gi))
    edges = [(0, 5), (1, 9), (2, 6), (5, 10), (3, 4), (7, 11), (1, 2)]
    for a_, b_ in edges:
        ax.plot([pos[a_][0], pos[b_][0]], [pos[a_][1], pos[b_][1]], color=FIG_BLUE, lw=1.0, zorder=2)
    for x_, y_, g in pos:
        ax.scatter(x_, y_, s=16, color=[FIG_GREY, '#b5b4af', '#6f6e6a'][g], zorder=3, edgecolor='white', lw=0.5)
    for (cx, cy), lab, dy, va in zip(groups, ['domain 1', 'domain 2', 'domain 3'], [-1.05, -1.05, 1.05], ['top', 'top', 'bottom']):
        ax.text(cx, cy+dy, lab, ha='center', va=va, fontsize=6, color=FIG_INK2)
    ax.set_xlim(-1.2, 3.6); ax.set_ylim(-1.6, 3.4); ax.set_aspect('equal')
    ax.text(1.2, -2.05, 'anomaly network $G_t$; scores roll up\nto indicators and domains', ha='center',
            va='top', fontsize=6.2, color=FIG_INK2)
    fig_letter(ax, 'c', x=-0.02, y=1.12)
    # (d) surrogate test of network anomaly density
    ax = fig.add_subplot(gs[0, 3])
    null = _rs.normal(0.14, 0.019, 4000)
    ax.hist(null, bins=35, color='#b7b6b1', edgecolor='white', lw=0.3)
    ax.axvline(0.222, color=FIG_BLUE, lw=1.8)
    ax.text(0.218, ax.get_ylim()[1]*0.95, 'observed\nNAD', ha='right', va='top', fontsize=6.2,
            color=FIG_BLUE, fontweight='bold')
    ax.set_yticks([]); ax.set_xlabel('NAD under refitted surrogates'); ax.grid(False)
    ax.spines['left'].set_visible(False)
    ax.text(0.5, -0.36, 'Monte Carlo $P=(k+1)/(N+1)$', transform=ax.transAxes, ha='center', va='top',
            fontsize=6.2, color=FIG_INK2)
    fig_letter(ax, 'd', x=-0.05, y=1.12)
    # arrows between panels
    _axs = list(fig.axes)
    for a1, a2 in zip(_axs[:-1], _axs[1:]):
        x1 = a1.get_position().x1; x2 = a2.get_position().x0; xm = (x1 + x2)/2
        fig.add_artist(FancyArrowPatch((xm-0.012, 0.5), (xm+0.012, 0.5), transform=fig.transFigure,
                                       arrowstyle='-|>', mutation_scale=8, color=FIG_INK2, lw=0.8))

    # (e) the two channels of NET2 on one pair, drawn in the principal-axis frame of its reference
    ax = fig.add_subplot(gs[1, 0:2])
    _r2 = np.random.default_rng(5)
    pts = np.column_stack([_r2.normal(0, 1.5, 9), _r2.normal(0, 0.45, 9)])
    ax.add_patch(Ellipse((0, 0), 2*4.4, 2*1.45, fc='#e8f1fc', ec=FIG_BLUE, lw=1.0))
    ax.scatter(pts[:, 0], pts[:, 1], s=9, color=FIG_GREY, zorder=3)
    ax.axhline(0, color=FIG_INK2, lw=0.5, ls='--', zorder=1); ax.axvline(0, color=FIG_INK2, lw=0.5, ls='--', zorder=1)
    A = (6.2, 0.15); B = (0.9, 2.9)
    ax.scatter(*A, s=26, color=FIG_BLUE, marker='D', zorder=4); ax.scatter(*B, s=26, color=FIG_ORANGE, marker='D', zorder=4)
    ax.text(A[0]+0.3, A[1]-0.55, 'A  along the usual\nco-movement:\noverall channel', fontsize=6.0, color=FIG_BLUE, ha='center', va='top')
    ax.text(B[0]+0.45, B[1], 'B  against it:\nboth channels', fontsize=6.0, color=FIG_ORANGE, ha='left', va='center')
    ax.text(-4.3, -1.75, 'boundary of the\nreference ($F$ law)', fontsize=5.8, color=FIG_BLUE, ha='left', va='top')
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False); ax.set_aspect('equal')
    ax.set_xlabel('major axis of the training correlation'); ax.set_ylabel('minor axis')
    ax.set_xlim(-6.5, 11.0); ax.set_ylim(-4.0, 4.0)
    fig_letter(ax, 'e', x=-0.04, y=1.08)
    # (f) from channels to the two-channel test
    ax = fig.add_subplot(gs[1, 2:4]); ax.set_axis_off(); ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    def _box(x, y, w, h, txt, fc, ec, bold=False):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec=ec, lw=0.8))
        ax.text(x+w/2, y+h/2, txt, ha='center', va='center', fontsize=5.9, color=FIG_INK, fontweight='bold' if bold else 'normal', linespacing=1.25)
    _box(0.0, 3.7, 4.4, 2.0, 'overall channel\nmean $-\\ln p$ over eligible pairs\n(departure in any direction)', '#e8f1fc', FIG_BLUE)
    _box(0.0, 0.5, 4.4, 2.0, 'minor-axis channel\nminor-axis part of the distance\n(pairs with $|r| \\geq 0.7$)', '#fdeee6', FIG_ORANGE)
    _box(5.6, 2.1, 4.4, 2.0, 'NET2\nsmaller of the two $P$ values,\ncalibrated with the same surrogates', '#f3f2ef', FIG_INK2, bold=False)
    for y0 in (4.7, 1.5):
        ax.add_artist(FancyArrowPatch((4.5, y0), (5.5, 3.1), arrowstyle='-|>', mutation_scale=7, color=FIG_INK2, lw=0.8, shrinkA=0, shrinkB=0))
    fig_letter(ax, 'f', x=-0.02, y=1.04)
    fig_save(fig, 'Fig1_framework'); plt.close(fig)
