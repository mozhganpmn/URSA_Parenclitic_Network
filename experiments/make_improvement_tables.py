"""LaTeX tables (Supplementary Tables S11 to S18 and main-text Table 2) from the confirmatory outputs.

  python make_improvement_tables.py EXPERIMENTS_DIR OUT_SI.tex OUT_MAIN.tex
Reads improvement_summary.json, improvement_gate.json and improvement_real_*.json.
"""
import json, sys, os, glob
D = sys.argv[1]; OUT_SI = sys.argv[2]; OUT_MAIN = sys.argv[3]
S = json.load(open(os.path.join(D, 'improvement_summary.json'))); G = json.load(open(os.path.join(D, 'improvement_gate.json')))
REAL = {}
for f in glob.glob(os.path.join(D, 'improvement_real_*.json')):
    for k, v in json.load(open(f)).items():
        if k in ('Vancouver', 'Calgary'): REAL[k] = v
HN = {'NAD-M': 'NAD', 'CON-M': 'CON', 'REL-M': 'REL', 'NAD-T': 'NAD', 'CON-T': 'CON', 'REL-T': 'REL', 'NET2-M': 'NET2', 'NET2-T': 'NET2',
      'UED-M': 'UED', 'dT2-M': '$\\mathrm{dT}^2$', 'SPE-M': 'SPE', 'UED-T': 'UED', 'dT2-T': '$\\mathrm{dT}^2$', 'SPE-T': 'SPE',
      'OMNI(NAD-M)': 'NAD+SPE', 'OMNI(NAD-T)': 'NAD+SPE'}
def head(keys):
    top = ' & '.join(HN[k] for k in keys); ref = ' & '.join(('M' if k.endswith('M') or k.endswith('M)') else 'T') for k in keys); return top, ref
NETK = ['NAD-M', 'CON-M', 'REL-M', 'NET2-M', 'NAD-T', 'CON-T', 'REL-T', 'NET2-T']
BASEK = ['UED-M', 'dT2-M', 'SPE-M', 'OMNI(NAD-M)', 'UED-T', 'dT2-T', 'SPE-T', 'OMNI(NAD-T)']
RN = {'iid': 'Independent normal', 'AR(1)': 'AR(1), $\\phi = 0.5$', 'independent trends': 'Independent linear trends', 'common trend': 'Common trend',
      'saturating': 'Saturating trends', 'random walk': 'Random walk', 'common factor': 'Stationary common factor'}
FN = {'A': 'Level shift', 'B': 'Relationship break', 'C': 'Opposing shift', 'D': 'Missing values', 'E': 'Sparse pair breaks', 'F': 'Non-linear bases'}
t = []
def tab_gate(keys, cap, label):
    top, ref = head(keys)
    s = ['\\begin{table}[!htbp]\n\\caption{' + cap + '}\n\\label{' + label + '}\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{2pt}\n\\begin{tabular}{@{}p{3.6cm}' + 'c'*len(keys) + '@{}}\n\\toprule']
    s.append('Regime & ' + top + ' \\\\\n & ' + ref + ' \\\\\n\\midrule')
    for r in G['regimes']:
        nm = RN[r['regime']] + (f", {int(round(100*r['miss']))}\\% missing" if r['miss'] != 0.1 else '')
        s.append(nm + ' & ' + ' & '.join(f"{r['results'][k]['rate']:.3f}" for k in keys) + ' \\\\')
    s.append('\\botrule\n\\end{tabular}\n\\end{table}\n'); return '\n'.join(s)
capgate = ('\\textbf{Type-I error of the surrogate test, %s.} Share of tested years with $P \\le 0.05$ (target 0.05) under the refitted increment-shift null, '
           'with %d datasets of 30 series over 14 years per regime, %d surrogates and nine years tested per dataset. M, mean reference; T, linear-trend reference. %s '
           'A statistic passed the pre-specified gate if its rate was at most 0.07 in every regime except the stationary common factor.')
t.append(tab_gate(NETK, capgate % ('network statistics', G['datasets'], G['surrogates'], 'NAD, share of activated pairs; CON, continuous pair score; REL, relationship channel; NET2, minimum-P combination of CON and REL.'), 'tab:gate_net'))
t.append(tab_gate(BASEK, capgate % ('standard monitors and the NAD+SPE combination', G['datasets'], G['surrogates'], 'UED, univariate exceedance density; $\\mathrm{dT}^2$, diagonal Hotelling $T^2$; SPE, principal-component residual; NAD+SPE, minimum-P combination of NAD and SPE.'), 'tab:gate_base'))
# ---------------- power by scenario
def ev(r):
    m = f", {int(round(100*r['miss']))}\\% missing" if r['family'] == 'D' else ''
    base = {'stationary': 'stationary', 'trending': 'trending', 'saturating': 'saturating', 'random walk': 'random walk'}[r['base']]
    if r['kind'] == 'level': e = f"level shift, {int(round(100*r['frac']))}\\%, $\\delta = {r['delta']}$"
    elif r['kind'] == 'break': e = f"break, $\\lambda = {r['lam']}$, {int(round(100*r['frac']))}\\%"
    elif r['kind'] == 'opposing': e = f"opposing, $\\delta = {r['delta']}$"
    else: e = f"pair break, {int(round(r['frac']*15))} of 15"
    return f"{r['family']}, {base}: {e}{m}"
def tab_power(keys, cap, label):
    top, ref = head(keys)
    cols = ' & '.join('\\shortstack{' + HN[k] + '\\\\' + ('M' if k.endswith('M') or k.endswith('M)') else 'T') + '}' for k in keys)
    hd = 'Scenario & ' + cols + ' \\\\'
    s = ['{\\scriptsize\\setlength{\\tabcolsep}{2.5pt}\\begin{longtable}{@{}p{3.7cm}' + 'c'*len(keys) + '@{}}\n\\caption{' + cap + '}\\label{' + label + '}\\\\\n\\toprule']
    s.append(hd + '\n\\midrule\n\\endfirsthead\n\\toprule\n' + hd + '\n\\midrule\n\\endhead\n\\botrule\n\\endfoot')
    last = None
    for r in S['rows']:
        if last is not None and r['family'] != last: s.append('\\addlinespace')
        last = r['family']
        s.append(ev(r) + ' & ' + ' & '.join(f"{r[k]:.2f}" for k in keys) + ' \\\\')
    s.append('\\end{longtable}}\n'); return '\n'.join(s)
def fix(tex): return tex
cappow = ('\\textbf{Detection power at a matched 5\\%% false-positive rate, %s.} Power in the event year (year index 10, so up to ten earlier years) for 30 series over 14 years, '
          'from %d datasets with the event and %d datasets of the same process without it (Monte Carlo standard error at most 0.007). Families: A, level shift; B, relationship break in a three-block factor model; '
          'C, opposing shift in the same model; D, missing values (10, 30 and 50\\%% missing; opposing shifts use $\\delta = 1.5$ and $\\lambda = 0.9$); E, sparse pair breaks (fifteen disjoint pairs with loading 0.9 and absolute within-pair correlation 0.81, one member takes an independent draw), '
          'which favours pairwise methods by design; F, level shift on saturating and random-walk bases. M, mean reference; T, linear-trend reference. %s')
t.append(fix(tab_power(NETK, cappow % ('network statistics', S['D'], S['Dn'], 'NAD, share of activated pairs; CON, continuous pair score; REL, relationship channel; NET2, minimum-P combination of CON and REL.'), 'tab:pow_net')))
t.append(fix(tab_power(BASEK, cappow % ('standard monitors', S['D'], S['Dn'], 'UED, univariate exceedance density; $\\mathrm{dT}^2$, diagonal Hotelling $T^2$; SPE, principal-component residual; NAD+SPE, minimum-P combination of NAD and SPE.'), 'tab:pow_base')))
# ---------------- family means and adoption
fm = S['family_means']; keys = NETK + BASEK
tt = ['\\begin{table}[!htbp]\n\\caption{\\textbf{Mean power by scenario family and the adoption rule.} Mean of the matched-false-positive-rate power over the scenarios of each family. The primary endpoint is the mean over the six families. The admitted statistic with the highest primary endpoint was adopted (a tie within 0.01 goes to the simpler statistic). Gate: x, failed the pre-specified type-I gate.}\n\\label{tab:fam}\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{2pt}\n\\begin{tabular}{@{}p{3.3cm}' + 'c'*len(NETK) + '@{}}\n\\toprule']
tt.append('Family & ' + ' & '.join('\\shortstack{' + HN[k] + '\\\\' + ('M' if k.endswith('M') else 'T') + '}' for k in NETK) + ' \\\\\n\\midrule')
for f in 'ABCDEF': tt.append(f"{f}, {FN[f]} & " + ' & '.join(f"{fm[f][k]:.2f}" for k in NETK) + ' \\\\')
tt.append('\\midrule\nPrimary endpoint & ' + ' & '.join((f"{S['primary_endpoint'][k]:.3f}" + ('$^{\\mathrm{x}}$' if S['gate'][k]['admitted'] is False else '')) for k in NETK) + ' \\\\')
tt.append('\\botrule\n\\end{tabular}\n\\end{table}\n')
tb = ['\\begin{table}[!htbp]\n\\caption{\\textbf{Mean power of the standard monitors by scenario family.} As for Supplementary Table S15.}\n\\label{tab:fam_base}\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{2pt}\n\\begin{tabular}{@{}p{3.3cm}' + 'c'*len(BASEK) + '@{}}\n\\toprule']
tb.append('Family & ' + ' & '.join('\\shortstack{' + HN[k] + '\\\\' + ('M' if k.endswith('M') or k.endswith('M)') else 'T') + '}' for k in BASEK) + ' \\\\\n\\midrule')
for f in 'ABCDEF': tb.append(f"{f}, {FN[f]} & " + ' & '.join(f"{fm[f][k]:.2f}" for k in BASEK) + ' \\\\')
tb.append('\\botrule\n\\end{tabular}\n\\end{table}\n')
t.append('\n'.join(tt)); t.append('\n'.join(tb))
# ---------------- real data
RK1 = ['NAD-M', 'CON-M', 'REL-M', 'NET2-M', 'NAD-T', 'NET2-T']
RK2 = ['UED-M', 'dT2-M', 'SPE-M', 'SPE-T', 'OMNI(NAD-M)', 'OMNI(NAD-T)']
def real_block(d, cols):
    s = ['\\begin{tabular}{@{}l' + 'c'*len(cols) + '@{}}\n\\toprule']
    s.append('Year & ' + ' & '.join('\\shortstack{' + HN.get(k, k) + '\\\\' + ('M' if k.endswith('M') or k.endswith('M)') else 'T') + '}' for k in cols) + ' \\\\\n\\midrule')
    for y in range(2015, 2025):
        cells = []
        for k in cols:
            c = d[k].get(str(y))
            if c is None: cells.append('--'); continue
            p = c['P_holm']; ps = '<0.001' if p < 0.001 else f'{p:.3f}'; cells.append('\\textbf{' + ps + '}' if p < 0.05 else ps)
        s.append(f'{y} & ' + ' & '.join(cells) + ' \\\\')
    s.append('\\botrule\n\\end{tabular}'); return '\n'.join(s)
def real_table(city):
    d = REAL[city]
    return ('\\begin{table}[!htbp]\n\\caption{\\textbf{Network and standard monitors applied to %s, 2015--2024.} Entries give Holm-adjusted $P$ across the ten reporting years for all %d working sub-indicators, with $10^4$ refitted increment-shift surrogates shared by all statistics. Bold, adjusted $P < 0.05$. Upper block, network statistics; lower block, standard monitors and the NAD+SPE combination. M, mean reference; T, linear-trend reference.}\n\\label{tab:real_%s}\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n' % (city, d['working'], city.lower())
            + real_block(d, RK1) + '\n\n\\medskip\n' + real_block(d, RK2) + '\n\\end{table}\n')
for city in ['Vancouver', 'Calgary']:
    if city in REAL: t.append(real_table(city))
open(OUT_SI, 'w').write('\n'.join(t)); print('written', OUT_SI)

# ---------------- main-text Table 2
import numpy as np
BASEV = {'UED': ('UED-M', 'UED-T'), 'dT2': ('dT2-M', 'dT2-T'), 'SPE': ('SPE-M', 'SPE-T')}
NS = {'A': 12, 'B': 12, 'C': 6, 'D': 9, 'E': 6, 'F': 4}
def best(f, b): return max(fm[f][BASEV[b][0]], fm[f][BASEV[b][1]])
cols = {'NAD': lambda f: fm[f]['NAD-M'], 'NET2': lambda f: fm[f]['NET2-M'], 'UED': lambda f: best(f, 'UED'), 'dT2': lambda f: best(f, 'dT2'), 'SPE': lambda f: best(f, 'SPE')}
gate = {r['regime'] + '|' + str(r['miss']): r for r in G['regimes']}
def worst(k): return max(r['results'][k]['rate'] for r in G['regimes'] if r['regime'] != 'common factor')
def cf(k): return [r for r in G['regimes'] if r['regime'] == 'common factor'][0]['results'][k]['rate']
KM = {'NAD': 'NAD-M', 'NET2': 'NET2-M', 'UED': 'UED-M', 'dT2': 'dT2-M', 'SPE': 'SPE-M'}
def flagged(city, k):
    ys = [y for y in range(2015, 2025) if str(y) in REAL[city][k] and REAL[city][k][str(y)]['P_holm'] < 0.05]
    if not ys: return 'none'
    out, i = [], 0
    while i < len(ys):
        j = i
        while j + 1 < len(ys) and ys[j + 1] == ys[j] + 1: j += 1
        out.append(str(ys[i]) if i == j else (f'{ys[i]}, {ys[j]}' if j == i + 1 else f'{ys[i]}--{ys[j]}')); i = j + 1
    return ', '.join(out)
m = ['\\begin{table}[!htbp]\n\\caption{\\textbf{Comparison of the two-channel test with NAD and three standard monitors.} Power is at a matched 5\\% false-positive rate for 30 series over 14 years, averaged over the scenarios of each family (5000 datasets per scenario; Monte Carlo standard error of a scenario at most 0.007). '
     'NAD, share of activated pairs; NET2, two-channel test (level channel and relationship channel); UED, univariate exceedance density; $\\mathrm{dT}^2$, diagonal Hotelling $T^2$; SPE, principal-component residual. NAD and NET2 use the mean reference. For the three standard monitors the power is that of the better of the mean and linear-trend versions in each family. '
     'Type-I error and the flagged years use the mean reference, and years are those with Holm-adjusted $P < 0.05$ on the full working set of each city. Scenario definitions and all results are in Supplementary Tables S11--S18.}\n\\label{tab:bench}\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{@{}p{4.1cm}>{\\centering\\arraybackslash}p{1.9cm}>{\\centering\\arraybackslash}p{1.9cm}ccc@{}}\n\\toprule\n & NAD & NET2 & UED & $\\mathrm{dT}^2$ & SPE \\\\\n\\midrule\n\\multicolumn{6}{@{}l}{\\textit{Power at a matched 5\\% false-positive rate (mean over the scenarios of a family)}} \\\\']
for f in 'ABCDEF': m.append(f"{FN[f]} ({NS[f]}) & " + ' & '.join(f"{cols[c](f):.2f}" for c in cols) + ' \\\\')
m.append('Mean of the six families & ' + ' & '.join(f"{np.mean([cols[c](f) for f in 'ABCDEF']):.2f}" for c in cols) + ' \\\\')
m.append('\\midrule\n\\multicolumn{6}{@{}l}{\\textit{Type-I error of the surrogate test (\\%)}} \\\\')
m.append('Highest of eight regimes without common factor & ' + ' & '.join(f"{100*worst(KM[c]):.1f}" for c in cols) + ' \\\\')
m.append('Stationary common factor & ' + ' & '.join(f"{100*cf(KM[c]):.1f}" for c in cols) + ' \\\\')
if REAL:
    m.append('\\midrule\n\\multicolumn{6}{@{}l}{\\textit{Years flagged in the two cities}} \\\\')
    for city in ['Vancouver', 'Calgary']: m.append(city + ' & ' + ' & '.join(flagged(city, KM[c]) for c in cols) + ' \\\\')
m.append('\\botrule\n\\end{tabular}\n\\end{table}\n')
open(OUT_MAIN, 'w').write('\n'.join(m)); print('written', OUT_MAIN)
