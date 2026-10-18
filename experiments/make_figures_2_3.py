"""Figs 2, 3 and Supplementary Fig. S1 (run from the repository root: python experiments/make_figures_2_3.py)."""
import json, os
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt
from scipy.stats import f as _fd, chi2 as _c2

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

_mc = json.load(open('results/method_comparison_output.json'))['held_out_fpr']
_sv = json.load(open('results/synthetic_validation_output.json'))
_pwr = pd.DataFrame(json.load(open('experiments/power_pair.json'))['results'])

# ---- Fig. 2: threshold calibration (a, b) and prospective versus whole-record reference (c)
with mpl.rc_context(FIG_RC):
    fig, ax = plt.subplots(1, 3, figsize=(150*MM, 52*MM), gridspec_kw=dict(wspace=0.5, width_ratios=[1.1, 0.9, 1.0]))
    a = ax[0]; _n = [r['n'] for r in _mc]
    a.axhline(0.05, color=FIG_INK2, lw=0.6, zorder=1)
    for k, c, m, lw in [('fpr_chi2', FIG_ORANGE, 'o', 1.1), ('fpr_F', FIG_BLUE, 'o', 2.0)]:
        a.plot(_n, [r[k] for r in _mc], marker=m, color=c, lw=lw, ms=4 if lw > 1.5 else 3.2,
               mec='white', mew=0.6, zorder=3 if lw > 1.5 else 2)
    for txt, xy, c in [('$\\chi^2$', (6.0, 0.25), FIG_ORANGE), ('$F$ (this work)', (3.0, 0.085), FIG_BLUE)]:
        a.text(*xy, txt, color=c, fontsize=6.0, fontweight='bold' if 'this' in txt else 'normal')
    a.set_xscale('log'); a.minorticks_off(); a.set_xticks([3, 5, 10, 20, 50]); a.set_xticklabels(['3', '5', '10', '20', '50'])
    a.set_xlim(2.7, 52.5); a.set_ylim(-0.02, 0.65)
    a.set_xlabel('Training window $n$'); a.set_ylabel('False-positive rate')
    fig_letter(a, 'a', x=-0.3)
    b = ax[1]; _nn = np.arange(3, 51)
    b.plot(_nn, [np.sqrt(2*(k+1)*(k-1)/(k*(k-2))*_fd.ppf(0.95, 2, k-2)) for k in _nn], color=FIG_BLUE, lw=2.0)
    b.axhline(np.sqrt(_c2.ppf(0.95, 2)), color=FIG_ORANGE, lw=1.2)
    b.set_xscale('log'); b.set_yscale('log'); b.minorticks_off()
    b.set_xticks([3, 5, 10, 20, 50]); b.set_xticklabels(['3', '5', '10', '20', '50'])
    b.set_yticks([2, 3, 5, 10, 20, 30]); b.set_yticklabels(['2', '3', '5', '10', '20', '30'])
    b.set_xlim(2.7, 52); b.set_xlabel('Training window $n$'); b.set_ylabel('Threshold $w^*$ ($\\sigma$)')
    b.text(3.4, 31, '32.6 at $n=3$', fontsize=6, color=FIG_INK2, va='center')
    b.text(50, 2.75, '$F$', ha='right', color=FIG_BLUE, fontsize=6.3, fontweight='bold')
    b.text(50, 2.15, '$\\chi^2$ = 2.45', ha='right', color=FIG_ORANGE, fontsize=6.3)
    fig_letter(b, 'b', x=-0.36)
    c = ax[2]; _q = _pwr[_pwr.n == 7].sort_values('delta')
    c.axhline(0.05, color=FIG_INK2, lw=0.6)
    c.plot(_q.delta, _q.sa_lfo, 'o-', color=FIG_BLUE, lw=2.0, ms=4, mec='white', mew=0.6, zorder=3)
    c.plot(_q.delta, _q.sa_whole, 'D-', color='#b5b4af', lw=1.1, ms=3.2, mec='white', mew=0.6)
    c.text(2.2, 0.86, 'LFO reference', color=FIG_BLUE, fontsize=6.0, fontweight='bold', ha='right')
    c.text(6, 0.10, 'whole-record\nreference', color='#9d9c97', fontsize=6.0, ha='right')
    c.set_ylim(0, 1.04); c.set_xlim(-0.2, 6.2); c.set_xticks([0, 1, 2, 3, 4, 6])
    c.set_xlabel('Break size $\\delta$ (SD)'); c.set_ylabel('Power at 5% false positives')
    fig_letter(c, 'c', x=-0.3)
    fig_save(fig, 'Fig2_calibration'); plt.close(fig)

# ---- Fig. 3: calibration of the network-level surrogate test
_cal = json.load(open('experiments/null_calibration.json'))['results']
_REG = ['iid', 'AR(1)', 'independent trends', 'common trend', 'saturating', 'common factor', 'random walk']
_LAB = {'iid': 'independent', 'AR(1)': 'AR(1)', 'independent trends': 'independent trends', 'common trend': 'common trend',
        'saturating': 'saturating trends', 'common factor': 'common factor', 'random walk': 'random walk'}
_FIX, _NEW = 'circular shift, fixed references', 'increment shift, refitted'
with mpl.rc_context(FIG_RC):
    fig, a = plt.subplots(1, 1, figsize=(85*MM, 55*MM))
    a.axvline(0.05, color=FIG_INK2, lw=0.6)
    for k, reg in enumerate(_REG):
        y = len(_REG) - 1 - k
        for key, col, off in [(_FIX, FIG_ORANGE, 0.17), (_NEW, FIG_BLUE, -0.17)]:
            r_ = _cal[reg][key]; lo, hi = r_['ci95']
            a.plot([lo, hi], [y + off]*2, color=col, lw=1.0)
            a.plot(r_['rate'], y + off, 'o', color=col, ms=4 if key == _NEW else 3.4, mec='white', mew=0.5, zorder=3)
    a.set_yticks(range(len(_REG))); a.set_yticklabels([_LAB[r_] for r_ in _REG][::-1]); a.grid(axis='y', visible=False)
    a.set_xlim(-0.01, 0.27); a.set_xlabel('Type-I error rate at $\\alpha = 0.05$')
    a.text(0.27, 1.0, 'fixed references', color=FIG_ORANGE, fontsize=6.0, ha='right')
    a.text(0.27, 0.6, 'refitted increment shift', color=FIG_BLUE, fontsize=6.0, ha='right', fontweight='bold')
    fig_save(fig, 'Fig3_network_test'); plt.close(fig)

# ---- Supplementary Fig. S1: activation probability when a correlation drops from 0.85 to 0
with mpl.rc_context(FIG_RC):
    fig, d = plt.subplots(1, 1, figsize=(75*MM, 52*MM))
    _r = _sv['experiment_B_break_iid']['per_year']; _yy = [r_['year'] for r_ in _r]
    d.axvspan(2013.5, 2019.5, color='#f3f2ef', lw=0, zorder=0); d.axvline(2019.5, color=FIG_INK2, lw=0.6)
    d.axhline(0.05, color=FIG_INK2, lw=0.6, zorder=1)
    d.plot(_yy, [r_['p_act_chi2'] for r_ in _r], 'o-', color=FIG_ORANGE, lw=1.2, mec='white', mew=0.6)
    d.plot(_yy, [r_['p_act_F'] for r_ in _r], 'o-', color=FIG_BLUE, lw=2.0, ms=4, mec='white', mew=0.6)
    d.text(2016.5, 0.63, 'before break', ha='center', fontsize=6, color=FIG_INK2)
    d.text(2022, 0.63, 'after', ha='center', fontsize=6, color=FIG_INK2)
    d.text(2014.0, 0.55, '$\\chi^2$', color=FIG_ORANGE, fontsize=6.3)
    d.text(2014.0, 0.09, '$F$', color=FIG_BLUE, fontsize=6.3, fontweight='bold')
    d.set_ylim(0, 0.66); d.set_xlim(2013.5, 2024.5); d.set_xticks([2014, 2017, 2020, 2023])
    d.set_xlabel('Target year'); d.set_ylabel('Activation probability')
    fig_save(fig, 'FigS1_correlation_drop'); plt.close(fig)
