"""Domain-pair matrix panels (diagnostic visualisation) used in Figs 5 and 6. Reads results/domain_matrix.json."""
import json, os, numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
CODES = {'Critical Infrastructure': 'CI', 'Economics': 'Ec', 'Environment': 'En', 'Health & Well-being': 'HW',
         'Institutional': 'In', 'Leadership & Strategy': 'LS', 'Social & Demographics': 'SD'}
DIV = LinearSegmentedColormap.from_list('div', ['#eb6834', '#f7d3c3', '#ffffff', '#cde2fb', '#6da7ec', '#2a78d6', '#184f95'])
NORM = TwoSlopeNorm(vmin=-0.3, vcenter=0.0, vmax=0.55)
def load(path):
    return json.load(open(path))
def draw(ax, DM, city, year, min_elig=30):
    d = DM[city]; doms = d['domains']; D = len(doms); r = d[str(year)]
    obs = np.array(r['obs'], float); mu = np.array(r['null_mean'], float); sd = np.array(r['null_sd'], float); el = np.array(r['eligible'], float)
    diff = obs - mu; z = diff/sd
    M = np.full((D, D), np.nan)
    for a in range(D):
        for b in range(a, D):
            if el[a, b] >= min_elig and np.isfinite(diff[a, b]): M[a, b] = diff[a, b]
    ax.set_facecolor('white')
    im = ax.imshow(np.ma.masked_invalid(M), cmap=DIV, norm=NORM, aspect='equal')
    for a in range(D):
        for b in range(D):
            if b < a: continue
            if np.isnan(M[a, b]):
                ax.add_patch(__import__('matplotlib').patches.Rectangle((b-0.5, a-0.5), 1, 1, fc='#ecebe8', ec='white', lw=0.5, hatch='////', zorder=1))
            else:
                ax.text(b, a, f'{int(round(z[a, b]))}', ha='center', va='center', fontsize=5.0, color='#0b0b0b' if abs(M[a, b]) < 0.3 else 'white')
    labs = [CODES[x] for x in doms]
    ax.set_xticks(range(D)); ax.set_xticklabels(labs, fontsize=5.8); ax.set_yticks(range(D)); ax.set_yticklabels(labs, fontsize=5.8)
    ax.xaxis.tick_top(); ax.tick_params(length=0); ax.grid(False)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title(f'{city} {year}', fontsize=6.6, pad=14, fontweight='bold', loc='left')
    return im
