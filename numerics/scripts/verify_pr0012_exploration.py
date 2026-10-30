# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Scalar checks against preserved aggregate inputs and Markdown displays."""
from pathlib import Path
import hashlib
import json
import math
import re
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'numerics/analysis/pr0012_64bf0f2c7697_confirmation_v1'
OUT = ROOT / 'numerics/analysis/pr0012_64bf0f2c7697_exploration_v2'
G = ['B_L', 'K2_M', 'K200_L', 'V05_L']
C = ['clean', 'GM800', 'GP800', 'BAL800', 'BAL400', 'TRI800']
E = ['normal', 'GM05', 'GP05', 'P02']
N = [20000, 5000, 5000, 5000]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def load(p):
    return json.loads(p.read_text())

def close(a, b):
    if a is None or b is None:
        assert a is b
    else:
        assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-14), (a, b)

def display(got, expected):
    if expected is None:
        assert got in ('NA', 'None')
    else:
        value = float(got.replace('−', '-'))
        assert abs(value - expected) <= .000051, (value, expected)

def pick(rows, **kw):
    found = [r for r in rows if all(r[k] == v for k, v in kw.items())]
    assert len(found) == 1
    return found[0]

def main():
    result = load(OUT / 'results.json')
    assert result == load(OUT.parent / 'pr0012_64bf0f2c7697_exploration_v1/results.json')
    assert set(load(OUT / 'validation.json')['errors']) == {'mean_closure', 'variance_closure'}
    snap = load(OUT / 'start_snapshot.json')
    # Inputs, not the mutable navigation files, are used by this verifier.
    input_hashes = {}
    for path in [SRC/'outer_arrays.npz', *sorted((SRC/'details').glob('rep_*.json'))]:
        rel = str(path.relative_to(ROOT))
        assert sha(path) == snap['anchors'][rel]
        input_hashes[rel] = sha(path)
    a = dict(np.load(SRC/'outer_arrays.npz', allow_pickle=False))
    details = [load(SRC/f'details/rep_{i:02d}.json') for i in range(50)]
    for r in result['decomposition']:
        g, c, e, ai = G.index(r['group']), C.index(r['candidate']), E.index(r['evaluation']), [.01, .001].index(r['alpha'])
        vals = [(int(a['policy'][i,g,c,e,ai,2,0])-int(a['policy'][i,g,c,e,ai,0,0]))/N[e] for i in range(50)]
        selections = [int(a['selection'][i,g,c,ai,2]) for i in range(50)]
        assert vals == r['per_repeat'] and selections == r['selection']
        mu = sum(vals)/50
        close(r['mean'], mu)
        within = between = 0.
        for part, key in zip(r['pieces'], [0, c+1, 7, -1]):
            ids = [i for i, v in enumerate(selections) if v == key]
            assert len(ids) == part['n']
            cond = sum(vals[i] for i in ids)/len(ids) if ids else None
            close(part['conditional_delta'], cond)
            close(part['contribution'], sum(vals[i] for i in ids)/50)
            if ids:
                for policy, name in [(0, 'P0'), (2, 'P2')]:
                    close(part[name], sum(int(a['policy'][i,g,c,e,ai,policy,0]) for i in ids)/(N[e]*len(ids)))
                within += sum((vals[i]-cond)**2 for i in ids)/50
                between += len(ids)*(cond-mu)**2/50
        total = sum((v-mu)**2 for v in vals)/50
        for key, value in [('within', within), ('between', between), ('variance', total), ('between_fraction', between/total if total else None)]:
            close(r[key], value)
    for r in result['transfer']:
        g, c, b = G.index(r['group']), C.index(r['candidate']), ['G-M','G-MC'].index(r['gate'])
        cut = 2.333890445586029 if g == 1 else [2.454714455705117, 3.0310874379402915][b]
        pairs = []
        for i, d in enumerate(details):
            rows = [x for x in d if x['group'] == r['group'] and x['candidate'] == r['candidate']]
            scores = {x['role']:x['scores'][b] for x in rows}
            ah, uh = scores['audit'] > cut, scores['update'] > cut
            assert ah == bool(a['gates'][i,g,c,b])
            pairs.append((ah, uh))
        m = [[sum(x == bool(row) and y == bool(col) for x,y in pairs) for col in range(2)] for row in range(2)]
        assert m == r['matrix']
        assert [sum(x for x,y in pairs),sum(y for x,y in pairs),m[0][1],m[1][0],m[0][1]+m[1][0]] == [r[k] for k in ['A_holds','U_shadow_exceeds','A_pass_U_exceed','A_hold_U_pass','disagree']]
    for r in result['branches']:
        g, e, ai, ci = G.index(r['group']), E.index(r['evaluation']), [.01,.001].index(r['alpha']), {'F':0,'clean_U':1,'C':7}[r['context']]
        hist = [sum(int(a['wcounts'][i,g,e,ci,ai,k]) for i in range(50)) for k in range(9)]
        assert sum(hist) == r['denominator'] == 50*N[e] and hist[8] == 0
        counts = dict(alarm=sum(hist[1:8]), G_only=sum(hist[k] for k in range(1,8) if k&3 and not k&4), P_only=hist[4], both=sum(hist[k] for k in range(1,8) if k&3 and k&4))
        assert counts == r['counts']
        assert sum(counts[k] for k in ('G_only','P_only','both')) == counts['alarm']
        for k,v in counts.items(): close(r[k], v/r['denominator'])
        for k in ['P_only','G_only']: close(r[k+'_share'], counts[k]/counts['alarm'] if counts['alarm'] else None)
    for r in result['passrisk']:
        g,c,e,ai = G.index(r['group']),C.index(r['candidate']),E.index(r['evaluation']),[.01,.001].index(r['alpha'])
        ids = [i for i in range(50) if not a['gates'][i,g,c,1]]
        assert len(ids) == r['passed']
        def counts(context): return [sum(int(a['wcounts'][i,g,e,context,ai,k]) for k in range(1,8)) for i in ids]
        u, clean = counts(c+1), counts(1)
        mean = lambda xs: sum(xs)/(len(ids)*N[e]) if ids else None
        close(r['U'],mean(u));close(r['clean_U'],mean(clean));close(r['delta'],mean([x-y for x,y in zip(u,clean)]))
        assert r['lower_than_clean'] == sum(x<y for x,y in zip(u,clean))
        assert r['higher_than_clean'] == sum(x>y for x,y in zip(u,clean))
        assert r['empirical_above_cap'] == (sum(x/N[e]>1.25*r['alpha'] for x in u) if e == 0 else None)
    # All 1,104 supplementary table rows: keys, counts and rounded values.
    lines = [x for x in (OUT/'ALL_EXPLORATORY_RESULTS.md').read_text().splitlines() if x.startswith('| ') and x.split('|')[1].strip() in G]
    assert len(lines) == 1104
    expected = []
    p = lambda v:'NA' if v is None else f'{100*v:.4f}'
    for r in result['decomposition']:
        for s in r['pieces']:
            expected.append([r['group'],r['candidate'],r['alpha'],r['evaluation'],s['path'],s['n'],p(s['conditional_delta']),p(s['contribution']),p(r['between_fraction'])])
    for r in result['transfer']:
        expected.append([r['group'],r['candidate'],r['gate'],r['matrix'],r['A_holds'],r['U_shadow_exceeds'],r['A_pass_U_exceed'],r['A_hold_U_pass']])
    for r in result['branches']:
        expected.append([r['group'],r['alpha'],r['context'],r['evaluation']]+[p(r[k]) for k in ['alarm','G_only','P_only','both','P_only_share']])
    for r in result['passrisk']:
        expected.append([r['group'],r['candidate'],r['alpha'],r['evaluation'],r['passed'],p(r['U']),p(r['clean_U']),p(r['delta']),r['lower_than_clean'],r['higher_than_clean'],r['empirical_above_cap']])
    for line,row in zip(lines,expected): assert [x.strip() for x in line.split('|')[1:-1]] == list(map(str,row))
    body = (OUT/'PR0012_EXPLORATORY_INSIGHTS_JA.md').read_text()
    def section(n): return body.split(f'## {n}.',1)[1].split('\n## ',1)[0]
    rows = [[x.strip() for x in line.split('|')[1:-1]] for line in section(3).splitlines() if line.startswith('| 1%') or line.startswith('| 0.1%')]
    assert len(rows) == 8
    for line,(alpha,e) in zip(rows,[(alpha,e) for alpha in [.01,.001] for e in E]):
        r=pick(result['passrisk'],group='V05_L',candidate='BAL400',alpha=alpha,evaluation=e)
        for got,key in zip(line[2:],['clean_U','U','delta']):display(got,100*r[key])
    rows = [[x.strip() for x in line.split('|')[1:-1]] for line in section(4).splitlines() if line.startswith('| ') and line.split('|')[1].strip() in ['正常','負ゲイン','正ゲイン','私的入力']]
    assert len(rows)==4
    for line,e in zip(rows,E):
        r=pick(result['decomposition'],group='K2_M',candidate='TRI800',alpha=.01,evaluation=e)
        for got,v in zip(line[1:],[r['mean'],r['pieces'][2]['contribution'],r['pieces'][3]['contribution'],r['between_fraction']]):display(got,100*v)
    rows = [[x.strip() for x in line.split('|')[1:-1]] for line in section(5).splitlines() if line.startswith('| K200') or line.startswith('| V05')]
    assert len(rows)==5
    for line in rows:
        group,candidate=line[0].split('・')
        r=pick(result['transfer'],group=group,candidate=candidate,gate='G-MC')
        assert list(map(int,line[1:]))==[r[k] for k in ['A_holds','U_shadow_exceeds','A_pass_U_exceed','A_hold_U_pass','disagree']]
    rows = [[x.strip() for x in line.split('|')[1:-1]] for line in section(6).splitlines() if line.startswith('| ') and line.split('|')[1].strip() in G]
    assert len(rows)==4
    for line in rows:
        rr=[pick(result['branches'],group=line[0],evaluation=e,alpha=.01,context='F') for e in ['GM05','GP05']]
        for got,v in zip(line[1:],[rr[0]['alarm'],rr[0]['P_only_share'],rr[1]['alarm'],rr[1]['P_only_share']]):display(got,100*v)
    assert '12支持・R05未解決' in body and '31/50' in body and '21/50' in body
    # Positive and negative controls for displayed precision; no scientific data modified.
    display('−0.5779',-.5778571428571428)
    try: display('−0.5789',-.5778571428571428)
    except AssertionError: pass
    else: raise AssertionError('Incorrect display accepted')
    figures=sorted((OUT/'figures').glob('*.png'))
    assert len(figures)==7
    v=dict(status='PASS_DELIVERY',input_hashes=input_hashes,scalar_checks={k:len(v) for k,v in result.items()},supplement_rows=1104,main_numeric_rows=21,v1_v2_results_equal=True,unused_counters_absent=True,display_negative_control=True,figures={str(p.relative_to(ROOT)):sha(p) for p in figures},visual_inspection='All seven figures displayed by the same agent; 02b replaces the overly categorical title and rounded 100.00% in 02 for the report.',raw_archive_scan_rerun=False,raw_decision_recomputation_rerun=False,scientific_generator_calls=0,refits=0,policy_changes=0,independent_reviewer=False)
    target=OUT/'delivery_validation.json'
    assert not target.exists()
    target.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
    print('PASS: 528 result records, 1104 supplement rows, 21 main numeric rows, 7 figures')

if __name__ == '__main__': main()
