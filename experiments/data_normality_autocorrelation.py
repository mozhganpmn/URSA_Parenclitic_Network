"""Shapiro-Wilk pass share and mean lag-1 autocorrelation of the working-set series (>= MINV observed values).
Run from the repository root: python experiments/data_normality_autocorrelation.py [OUT.json]"""
import sys, json, numpy as np, scipy.stats as st
root = '.'; sys.path.insert(0, root)
import parenclitic_core as pc
def acf1(v):
    ok = np.isfinite(v); m = v[ok].mean(); d = v - m
    num = sum(d[t]*d[t+1] for t in range(len(v)-1) if ok[t] and ok[t+1]); den = np.sum(d[ok]**2)
    npair = sum(1 for t in range(len(v)-1) if ok[t] and ok[t+1])
    return (num/den if den > 0 else np.nan), npair
def summ(V, minv, what):
    rows = []
    for r in V:
        x = r[np.isfinite(r)]
        if what == 'inc':
            # increments between consecutive observed values
            x = np.diff(x)
        if len(x) < minv or np.ptp(x) == 0: continue
        p = st.shapiro(x).pvalue
        rr = r if what == 'lvl' else None
        if what == 'lvl': a, npair = acf1(r)
        else: a = acf1(x.astype(float))[0]
        rows.append((p, a))
    p = np.array([a for a, b in rows]); a = np.array([b for a, b in rows]); a = a[np.isfinite(a)]
    return dict(N=len(p), share_pass=float((p >= 0.05).mean()), n_pass=int((p >= 0.05).sum()), mean_lag1=float(a.mean()), median_lag1=float(np.median(a)), share_pos_lag1=float((a > 0).mean()))
out = {}
for city, f in [('Vancouver', 'Input.xlsx'), ('Calgary', 'Input_Calgary.xlsx')]:
    X, V, yc = pc.load_input(f'{root}/data/{f}')
    out[city] = dict(working_set=int(V.shape[0]), n_ge8_values=int((np.isfinite(V).sum(1) >= 8).sum()))
    for minv in (8, 7):
        out[city][f'levels_min{minv}'] = summ(V, minv, 'lvl')
    out[city]['increments_min8'] = summ(V, 7, 'inc')   # >=8 values give >=7 increments
out['note'] = 'Shapiro-Wilk on observed values; lag-1 autocorrelation uses the standard estimator with consecutive observed years only; constant series dropped. increments_min8: series with >=8 values, tested on year-to-year increments of consecutive observed values.'
json.dump(out, open((sys.argv[1] if len(sys.argv) > 1 else 'results/data_normality_autocorrelation.json'), 'w'), indent=1); print(json.dumps(out, indent=1))
