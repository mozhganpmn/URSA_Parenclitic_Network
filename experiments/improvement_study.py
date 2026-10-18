"""Confirmatory study of PLAN_improvements.md.

  python improvement_study.py gate  D N [workers]       calibration gate (type-I error of the surrogate test)
  python improvement_study.py power D Dn [workers]      matched-false-positive-rate power, raw statistics at the event year
  python improvement_study.py summary                   tables, family means, gate, adoption rule (reads the JSON/npy outputs)
  python improvement_study.py real N                    Vancouver and Calgary (run last)

Raw statistic arrays go to improvement_raw/ so that every summary can be recomputed without simulation.
"""
import sys, os, json, time, hashlib, numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.stats import rankdata
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..')); sys.path.insert(0, HERE)
from parenclitic_core import Pairs, make_increment_factory
import variants as V
import benchmark_monitors as bm
from synthetic_regimes import gen, kinds, M, T
J = 10
RAW = os.path.join(HERE, 'improvement_raw'); os.makedirs(RAW, exist_ok=True)
NAMES = V.AllStats.names                       # 12 statistics
# Omnibus statistics: the smaller of the two P values (min-p). OMNI(X) pairs a NAD-family variant with the SPE of the same
# reference (companion monitor); NET2 pairs the level channel (CON) and the relationship channel (REL) of the network.
OMNI_SPEC = {'OMNI(' + k + ')': (k, 'SPE-' + k[-1]) for k in V.NAMES_NAD}
OMNI_SPEC.update({'NET2-M': ('CON-M', 'REL-M'), 'NET2-T': ('CON-T', 'REL-T')})
OMNI = list(OMNI_SPEC)
NEG = -1e9                                       # an undefined statistic counts as no evidence
SEEDS = dict(gate_data=20261101, gate_surr=20261102, null_base=5000000, event_base=6000000)
PLAN_SHA = hashlib.sha256(open(os.path.join(HERE, 'PLAN_improvements.md'), 'rb').read()).hexdigest()

# ------------------------------------------------------------------ generators (families A to C are those of benchmark_monitors.py)
ar1 = bm.ar1
def base_noise(rng, base):
    if base == 'random walk': return np.cumsum(rng.normal(size=(M, T)), axis=1)
    x = ar1(rng, (M, T)); tt = np.arange(T)
    if base == 'trending': x = x + rng.normal(0, 0.3, M)[:, None]*tt
    elif base == 'saturating': x = x + rng.uniform(1, 4, M)[:, None]*(1 - np.exp(-tt/rng.uniform(2, 6, M)[:, None]))
    return x
def gen_level(rng, base, frac, delta, miss):
    x = base_noise(rng, base); k = int(round(frac*M))
    if k: idx = rng.choice(M, k, replace=False); x[idx, J] += delta*rng.choice([-1, 1], k)
    return bm.finish(x, rng, miss)
def gen_pairs(rng, base, share, rho, miss):
    """Fifteen disjoint pairs with within-pair correlation rho. In the event year one member of a share of the pairs takes
    an independent draw of the shared component (same variance, so no marginal changes)."""
    K = M//2; f = ar1(rng, (K, T)); eta = ar1(rng, (M, T)); sg = rng.choice([-1, 1], M); pid = np.repeat(np.arange(K), 2)
    x = rho*sg[:, None]*f[pid] + np.sqrt(1 - rho**2)*eta
    k = int(round(share*K))
    if k: sel = rng.choice(K, k, replace=False); idx = 2*sel; x[idx, J] = rho*rng.normal(size=k) + np.sqrt(1 - rho**2)*eta[idx, J]
    if base == 'trending': x = x + rng.normal(0, 0.3, M)[:, None]*np.arange(T)
    return bm.finish(x, rng, miss)
SL = {'stationary': 0.0, 'trending': 0.3}
def scenarios():
    sc = []
    def add(fam, kind, base, miss, **kw): sc.append(dict(family=fam, kind=kind, base=base, miss=miss, **kw))
    for base in ['stationary', 'trending']:
        for frac in [0.1, 0.3]:
            for delta in [2, 3, 4]: add('A', 'level', base, 0.10, frac=frac, delta=delta)
        for lam in [0.9, 0.7]:
            for frac in [0.3, 0.6, 1.0]: add('B', 'break', base, 0.10, frac=frac, lam=lam)
        for delta in [1.0, 1.5, 2.0]: add('C', 'opposing', base, 0.10, delta=delta, lam=0.9)
    for base in ['stationary', 'trending']:
        for miss in [0.1, 0.3, 0.5]: add('D', 'level', base, miss, frac=0.3, delta=2.5)
    for miss in [0.1, 0.3, 0.5]: add('D', 'opposing', 'stationary', miss, delta=1.5, lam=0.9)
    for base in ['stationary', 'trending']:
        for share in [1/3, 2/3, 1.0]: add('E', 'pairs', base, 0.10, frac=share, rho=0.9)
    for base in ['saturating', 'random walk']:
        for delta in [2.5, 3]: add('F', 'level', base, 0.10, frac=0.3, delta=delta)
    for i, s in enumerate(sc): s['id'] = i; s['group'] = group_key(s)
    return sc
def group_key(s):
    if s['kind'] == 'level': return ('level', s['base'], s['miss'])
    if s['kind'] in ('break', 'opposing'): return ('factor', s['base'], s['lam'], s['miss'])
    return ('pairs', s['base'], s['miss'])
def draw_event(s, rng):
    if s['kind'] == 'level': return gen_level(rng, s['base'], s['frac'], s['delta'], s['miss'])
    if s['kind'] == 'break': return bm.gen_break(rng, s['frac'], s['lam'], SL[s['base']], s['miss'])
    if s['kind'] == 'opposing': return bm.gen_opposing(rng, s['delta'], SL[s['base']], s['miss'])
    return gen_pairs(rng, s['base'], s['frac'], s['rho'], s['miss'])
def draw_null(g, rng):
    if g[0] == 'level': return gen_level(rng, g[1], 0.0, 0, g[2])
    if g[0] == 'factor': return bm.gen_break(rng, 0.0, g[2], SL[g[1]], g[3])
    return gen_pairs(rng, g[1], 0.0, 0.9, g[2])
def stat_J(x, pr):
    s = V.AllStats(x, pr)(x, years=[J]); return [float(s[k][J]) for k in NAMES]

# ------------------------------------------------------------------ power
def _null_task(a):
    g, gi, Dn = a; f = os.path.join(RAW, f'null_{gi}_{Dn}.npy')
    if os.path.exists(f): return gi
    rng = np.random.default_rng(SEEDS['null_base'] + gi); pr = Pairs(M)
    np.save(f, np.array([stat_J(draw_null(g, rng), pr) for _ in range(Dn)])); return gi
def _alt_task(a):
    s, D = a; f = os.path.join(RAW, f'alt_{s["id"]}_{D}.npy')
    if os.path.exists(f): return s['id']
    rng = np.random.default_rng(SEEDS['event_base'] + s['id']); pr = Pairs(M)
    np.save(f, np.array([stat_J(draw_event(s, rng), pr) for _ in range(D)])); return s['id']
def run_power(D, Dn, workers):
    sc = scenarios(); groups = sorted(set(s['group'] for s in sc), key=str); t0 = time.time()
    json.dump(dict(groups=[list(g) for g in groups], scenarios=sc, D=D, Dn=Dn, plan_sha256=PLAN_SHA, seeds=SEEDS, J=J, names=NAMES), open(os.path.join(HERE, 'improvement_design.json'), 'w'), indent=1)
    with Pool(workers) as p:
        for gi in p.imap_unordered(_null_task, [(g, i, Dn) for i, g in enumerate(groups)]): print('null group', gi, f'{time.time()-t0:.0f}s', flush=True)
        for si in p.imap_unordered(_alt_task, [(s, D) for s in sc]): print('scenario', si, f'{time.time()-t0:.0f}s', flush=True)

def mid_p(null, x):
    """Upper-tail mid-p of values x against the null sample (NaN stays NaN)."""
    srt = np.sort(null[np.isfinite(null)]); lo = np.searchsorted(srt, x, 'left'); hi = np.searchsorted(srt, x, 'right'); n = len(srt)
    return ((n - hi) + 0.5*(hi - lo) + 0.5)/(n + 1)
def omni_columns(nul, alt):
    """Omnibus statistic = minus the smaller of the two mid-p values against the process null (a monotone transform of
    min-p, so that larger is more extreme like the other statistics). nul, alt: arrays (datasets x 12)."""
    cols_n, cols_a = {}, {}
    nul = np.nan_to_num(nul, nan=NEG); alt = np.nan_to_num(alt, nan=NEG)
    for name, (k, k2) in OMNI_SPEC.items():
        i, j = NAMES.index(k), NAMES.index(k2)
        cols_n[name] = -np.minimum(mid_p(nul[:, i], nul[:, i]), mid_p(nul[:, j], nul[:, j]))
        cols_a[name] = -np.minimum(mid_p(nul[:, i], alt[:, i]), mid_p(nul[:, j], alt[:, j]))
    return cols_n, cols_a
def power_table(D, Dn):
    des = json.load(open(os.path.join(HERE, 'improvement_design.json'))); groups = [tuple(g) for g in des['groups']]; rows = []
    for s in des['scenarios']:
        g = tuple(s['group']) if isinstance(s['group'], list) else s['group']; g = tuple(g); gi = groups.index(g)
        nul = np.load(os.path.join(RAW, f'null_{gi}_{Dn}.npy')); alt = np.load(os.path.join(RAW, f'alt_{s["id"]}_{D}.npy'))
        cn, ca = omni_columns(nul, alt); r = dict(s)
        for i, k in enumerate(NAMES): r[k] = round(bm.matched_power(np.nan_to_num(nul[:, i], nan=NEG), np.nan_to_num(alt[:, i], nan=NEG)), 4)
        for k in OMNI: r[k] = round(bm.matched_power(cn[k], ca[k]), 4)
        rows.append(r)
    return rows

# ------------------------------------------------------------------ calibration gate
GATE_REGIMES = [(k, 0.10) for k in kinds] + [('iid', 0.5), ('independent trends', 0.5)]
def omni_P(obs, null, rng, conservative=False):
    """Westfall-Young min-p omnibus for one year. obs: dict stat -> value; null: dict stat -> array (N,). Returns dict
    name -> P. The N surrogates and the observed value are treated as an exchangeable set of N+1. An undefined channel
    counts as no evidence; if both channels are undefined for the observed data the year is not tested."""
    out = {}
    for name, pair in OMNI_SPEC.items():
        if not any(np.isfinite(obs[k]) for k in pair): out[name] = np.nan; continue
        n1 = len(null[pair[0]]) + 1; minp = np.full(n1, 1.0)
        for k in pair:
            a = np.nan_to_num(np.concatenate([null[k], [obs[k]]]), nan=NEG)
            aj = a if conservative else a + 1e-12*rng.random(n1)
            minp = np.minimum(minp, rankdata(-aj, method='min')/n1)
        out[name] = float((minp <= minp[-1] + 1e-15).mean())
    return out
def surrogate_test(X, pr, years, N, rng, conservative=False):
    A = V.AllStats(X, pr); obs = A(X); draw = make_increment_factory(X)
    null = {k: np.full((N, X.shape[1]), np.nan) for k in NAMES}
    for i in range(N):
        s = A(draw(rng))
        for k in NAMES: null[k][i] = s[k]
    res = {k: {} for k in NAMES + OMNI}
    for j in years:
        for k in NAMES:
            nj = null[k][:, j]; nj = nj[np.isfinite(nj)]; o = obs[k][j]
            if not np.isfinite(o) or len(nj) < 10: continue
            gt = int((nj > o + 1e-12).sum()); eq = int((np.abs(nj - o) <= 1e-12).sum())
            p = (gt + eq + 1)/(len(nj) + 1) if conservative else (gt + rng.random()*eq + 1)/(len(nj) + 1)
            res[k][j] = dict(obs=float(o), null_mean=float(nj.mean()), null_sd=float(nj.std(ddof=1)), P=float(p), z=float((o - nj.mean())/nj.std(ddof=1)) if nj.std(ddof=1) > 0 else float('nan'))
        po = omni_P({k: obs[k][j] for k in NAMES}, {k: null[k][:, j] for k in NAMES}, rng, conservative)
        for k, v in po.items():
            if np.isfinite(v): res[k][j] = dict(P=float(v))
    return res
def _gate_task(a):
    kind, miss, D, N = a; f = os.path.join(RAW, f'gate_{kind.replace(" ", "_")}_{miss}_{D}_{N}.json')
    if os.path.exists(f): return json.load(open(f))
    rng = np.random.default_rng(SEEDS['gate_data'] + int(1000*miss) + 7*kinds.index(kind)); data = [gen(kind, rng, miss) for _ in range(D)]
    r2 = np.random.default_rng(SEEDS['gate_surr'] + int(1000*miss) + 7*kinds.index(kind)); pr = Pairs(M); rates = {k: [] for k in NAMES + OMNI}; t0 = time.time()
    for x in data:
        res = surrogate_test(x, pr, range(5, T), N, r2)
        for k in rates: rates[k].append(np.mean([v['P'] <= 0.05 for v in res[k].values()]) if len(res[k]) else np.nan)
    out = {}
    for k in rates:
        a_ = np.array(rates[k]); a_ = a_[np.isfinite(a_)]; bs = [a_[r2.integers(0, len(a_), len(a_))].mean() for _ in range(2000)]
        out[k] = dict(rate=round(float(a_.mean()), 4), ci95=[round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)], datasets=int(len(a_)))
    rec = dict(regime=kind, miss=miss, results=out, seconds=round(time.time() - t0)); json.dump(rec, open(f, 'w')); return rec
def run_gate(D, N, workers):
    with Pool(workers) as p:
        recs = []
        for rec in p.imap_unordered(_gate_task, [(k, m, D, N) for k, m in GATE_REGIMES]):
            print(rec['regime'], rec['miss'], {k: v['rate'] for k, v in rec['results'].items() if k in NAMES}, rec['seconds'], 's', flush=True); recs.append(rec)
    order = {(k, m): i for i, (k, m) in enumerate(GATE_REGIMES)}; recs.sort(key=lambda r: order[(r['regime'], r['miss'])])
    json.dump(dict(datasets=D, surrogates=N, plan_sha256=PLAN_SHA, tested_year_indices=list(range(5, T)), regimes=recs), open(os.path.join(HERE, 'improvement_gate.json'), 'w'), indent=1)

# ------------------------------------------------------------------ real data
_D = os.path.join(HERE, '..', 'data'); FILES = {'Vancouver': os.path.join(_D, 'Input.xlsx'), 'Calgary': os.path.join(_D, 'Input_Calgary.xlsx')}
def run_real(N, cities=('Vancouver', 'Calgary')):
    out = dict(plan_sha256=PLAN_SHA)
    for city in cities:
        full = pd.read_excel(FILES[city], sheet_name='Sheet1'); yc = sorted([c for c in full.columns if isinstance(c, (int, np.integer))])
        Vv = full[yc].apply(pd.to_numeric, errors='coerce').values.astype(float); Vv = Vv[np.isfinite(Vv).sum(1) >= 3]
        pr = Pairs(Vv.shape[0]); yi = {y: k for k, y in enumerate(yc)}; rep = list(range(2015, 2025)); rng = np.random.default_rng(20231); t0 = time.time()
        res = surrogate_test_real(Vv, pr, [yi[y] for y in rep], N, rng, yc)
        out[city] = dict(working=int(Vv.shape[0]), surrogates=N, years=yc)
        for k in NAMES + OMNI:
            yr = {y: res[k][yi[y]] for y in rep if yi[y] in res[k]}; ph = bm.holm({y: yr[y]['P'] for y in yr})
            for y in yr: yr[y]['P_holm'] = ph[y]
            out[city][k] = {str(y): {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()} for y, v in yr.items()}
        print(city, f'{time.time()-t0:.0f}s', {k: [y for y in rep if str(y) in out[city][k] and out[city][k][str(y)]['P_holm'] < 0.05] for k in NAMES + OMNI}, flush=True)
    json.dump(out, open(os.path.join(HERE, 'improvement_real_' + '_'.join(cities) + '.json'), 'w'), indent=1)
def surrogate_test_real(X, pr, years, N, rng, yc):
    A = V.AllStats(X, pr, times=yc); obs = A(X); draw = make_increment_factory(X)
    null = {k: np.full((N, X.shape[1]), np.nan) for k in NAMES}
    for i in range(N):
        s = A(draw(rng))
        for k in NAMES: null[k][i] = s[k]
    res = {k: {} for k in NAMES + OMNI}
    for j in years:
        for k in NAMES:
            nj = null[k][:, j]; nj = nj[np.isfinite(nj)]; o = obs[k][j]
            if not np.isfinite(o) or len(nj) < 10: continue
            gt = int((nj > o + 1e-12).sum()); eq = int((np.abs(nj - o) <= 1e-12).sum()); p = (gt + eq + 1)/(len(nj) + 1)
            res[k][j] = dict(obs=float(o), null_mean=float(nj.mean()), null_sd=float(nj.std(ddof=1)), P=float(p), z=float((o - nj.mean())/nj.std(ddof=1)) if nj.std(ddof=1) > 0 else float('nan'))
        po = omni_P({k: obs[k][j] for k in NAMES}, {k: null[k][:, j] for k in NAMES}, rng, True)
        for k, v in po.items():
            if np.isfinite(v): res[k][j] = dict(P=float(v))
    return res

if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'gate': run_gate(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 2)
    elif mode == 'power': run_power(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 2)
    elif mode == 'real': run_real(int(sys.argv[2]), tuple(sys.argv[3:]) if len(sys.argv) > 3 else ('Vancouver', 'Calgary'))
