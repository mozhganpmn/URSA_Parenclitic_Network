"""Applies the rules of PLAN_improvements.md section 5 to the confirmatory outputs (no choices are made here by hand).

  python improvement_summary.py     writes improvement_summary.json and prints the tables
"""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..')); sys.path.insert(0, HERE)
import improvement_study as S
NAMES, OMNI = S.NAMES, S.OMNI
SINGLE = ['NAD-M', 'CON-M', 'NAD-T', 'CON-T', 'REL-M', 'REL-T']          # order = simplicity order of the tie rule
CAND = SINGLE + ['NET2-M', 'NET2-T']
BASE = {'UED': ('UED-M', 'UED-T'), 'dT2': ('dT2-M', 'dT2-T'), 'SPE': ('SPE-M', 'SPE-T')}
FAM = ['A', 'B', 'C', 'D', 'E', 'F']
FAMNAME = {'A': 'Level shift', 'B': 'Relationship break (factor model)', 'C': 'Opposing shift', 'D': 'Missing values',
           'E': 'Sparse pair breaks', 'F': 'Non-linear bases'}

def load(D, Dn):
    rows = S.power_table(D, Dn); gate = json.load(open(os.path.join(HERE, 'improvement_gate.json'))) if os.path.exists(os.path.join(HERE, 'improvement_gate.json')) else None
    return rows, gate
def family_means(rows, keys):
    return {f: {k: float(np.mean([r[k] for r in rows if r['family'] == f])) for k in keys} for f in FAM}
def admitted(gate, thr=0.07):
    """Gate: rate <= thr in every regime except the stationary common factor."""
    if gate is None: return None
    out = {}
    for k in NAMES + OMNI:
        rs = [(r['regime'], r['miss'], r['results'][k]['rate']) for r in gate['regimes'] if r['regime'] != 'common factor']
        out[k] = dict(admitted=all(x[2] <= thr for x in rs), worst=max(rs, key=lambda x: x[2]), rates={f'{a}|{b}': c for a, b, c in rs},
                      common_factor=[r['results'][k]['rate'] for r in gate['regimes'] if r['regime'] == 'common factor'][0])
    return out
def adopt(fm, adm):
    pe = {k: float(np.mean([fm[f][k] for f in FAM])) for k in CAND}
    ok = [k for k in CAND if adm is None or adm[k]['admitted']]
    best = max(pe[k] for k in ok); pick = [k for k in CAND if k in ok and pe[k] >= best - 0.01][0]
    return pe, ok, pick
def wording(rows, fam, var, base):
    """PLAN section 5 rule 3. The comparator is the better of the M and T versions of the baseline in the family."""
    rs = [r for r in rows if r['family'] == fam]; versions = BASE[base]
    comp = max(versions, key=lambda v: np.mean([r[v] for r in rs]))
    d = np.mean([r[var] for r in rs]) - np.mean([r[comp] for r in rs]); ahead = np.mean([r[var] > r[comp] for r in rs]); behind = np.mean([r[var] < r[comp] for r in rs])
    if d >= 0.05 and ahead >= 2/3: w = 'more powerful'
    elif d <= -0.05 and behind >= 2/3: w = 'less powerful'
    else: w = 'comparable'
    return dict(comparator=comp, diff=round(float(d), 3), ahead=round(float(ahead), 2), behind=round(float(behind), 2), wording=w)

def main(D=5000, Dn=10000):
    rows, gate = load(D, Dn); keys = NAMES + OMNI; fm = family_means(rows, keys); adm = admitted(gate, 0.07)
    pe, ok, pick = adopt(fm, adm)
    out = dict(plan_sha256=S.PLAN_SHA, D=D, Dn=Dn, rows=rows, family_means=fm, primary_endpoint=pe, admitted_candidates=ok, adopted=pick,
               gate=adm, wording={f: {b: wording(rows, f, pick, b) for b in BASE} for f in FAM})
    # also the wording for every candidate, for the supplement
    out['wording_all'] = {v: {f: {b: wording(rows, f, v, b) for b in BASE} for f in FAM} for v in CAND}
    json.dump(out, open(os.path.join(HERE, 'improvement_summary.json'), 'w'), indent=1)
    print('family means'); print(' ' * 8 + ' '.join(f'{k:>11s}' for k in keys))
    for f in FAM: print(f'{f:8s}' + ' '.join(f'{fm[f][k]:11.3f}' for k in keys))
    print('primary endpoint (candidates):', {k: round(v, 3) for k, v in pe.items()})
    if adm: print('gate (admitted, worst regime):', {k: (adm[k]['admitted'], adm[k]['worst'][2]) for k in NAMES + OMNI})
    print('admitted candidates:', ok, '| adopted:', pick)
    for f in FAM: print(f, {b: out['wording'][f][b]['wording'] + f" ({out['wording'][f][b]['diff']:+.2f} vs {out['wording'][f][b]['comparator']})" for b in BASE})

if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5000, int(sys.argv[2]) if len(sys.argv) > 2 else 10000)
