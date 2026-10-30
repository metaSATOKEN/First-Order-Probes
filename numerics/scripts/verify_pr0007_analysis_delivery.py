# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Second-path delivery checks: integer margins, report numbers, links, preservation."""
from pathlib import Path
import hashlib
import json
import re
import math
import numpy as np
from PIL import Image
import pr0007_spec as spec

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'numerics/analysis/pr0007_60acab7af20d_performance_v1'
def read(n):return json.loads((OUT/n).read_text())
def digest(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def percent(x):return f'{100*x:.4f}'


def main():
    v=read('validation.json');assert v['status']=='PASS_SAVED_DATA_RECOMPUTATION'
    src=ROOT/'numerics/scripts/analyze_pr0007_calibration.py';assert digest(src)==v['source_sha256']
    rates=read('recomputed_rates.json');policy=read('recomputed_calibration_contrasts.json');pairs=read('recomputed_within_methods.json')
    L={(r['context']['index'],r['method'],r['alpha']):r for r in rates}
    checked=0
    with np.load(OUT/'recomputed_counts.npz',allow_pickle=False) as x:
        for a,b in [('counts','available'),('pairs','pair_n'),('contrasts','contrast_n')]:assert np.array_equal(x[a].sum(-1),x[b])
        for r in rates:
            c=r['context'];i=c['index'];m=spec.METHODS.index(r['method']);a=spec.ALPHA_DENOMINATORS.index(round(1/r['alpha']))
            n=20000 if '__' not in c['case'] else 5000
            assert (x['available'][:,i,a,m]==n).all()
            values=(n-x['counts'][:,i,a,m,0])/n
            assert np.array_equal(values,np.array(r['rate']['values']))
            assert np.allclose(1-values,r['miss']['values'],atol=0,rtol=0)
            checked+=1
        for r in policy:
            left=L[r['left']['index'],r['method'],r['alpha']]['rate']['values']
            right=L[r['right']['index'],r['method'],r['alpha']]['rate']['values']
            assert np.allclose(np.subtract(left,right),r['difference']['values'],atol=2e-16,rtol=0)
            cells=np.array(r['counts']);den=np.array(r['joint_available'])
            assert np.array_equal(cells.sum(-1),den)
            for j,k in enumerate(('both','gained','lost','neither')):
                assert np.array_equal(cells[:,j]/den,np.array(r['cells'][k]['values']))
            checked+=1
        for r in pairs:
            c=r['context']['index'];left=L[c,r['left'],r['alpha']]['rate']['values'];right=L[c,r['right'],r['alpha']]['rate']['values']
            assert np.allclose(np.subtract(left,right),r['difference']['values'],atol=2e-16,rtol=0)
            checked+=1
    text=(OUT/'PR0007_CALIBRATION_REPORT_JA.md').read_text()
    primary=read('report_primary.json')
    for r in primary:
        line=next(l for l in text.splitlines() if l.startswith('| '+r['id']+' '))
        expected=[str(r['mix']),percent(r['clean']),percent(r['mixed']),percent(r['delta']).replace('-','−'),
            '['+percent(r['interval']['lower']).replace('-','−')+', '+percent(r['interval']['upper']).replace('-','−')+']',percent(r['mixed_fpr'])]
        assert [s.strip() for s in line.split('|')][2:-1]==expected,(r['id'],line,expected)
    # Check the other principal numeric tables directly against rate rows.
    byid={(r['context']['id'],r['context']['case'],r['method'],r['alpha']):r for r in rates}
    def rate(calid,case,method='W0',alpha=.01):return byid[calid,case,method,alpha]['rate']['mean']
    for method in spec.METHODS:
        prefix='| '+method+' | '
        line=next(l for l in text.splitlines() if l.startswith(prefix))
        vals=[rate('B_L__mix_P02_800','B_L__'+f,method) for f in spec.FAULTS]+[rate('B_L__mix_P02_800','B_L',method)]
        assert [s.strip() for s in line.split('|')][2:-1]==[percent(x) for x in vals]
    for r in read('report_saturation.json'):
        if r['group'] not in ('K2_M','V15_L') or r['alpha']!=.001:continue
        expected='| '+' | '.join([r['group'],f"{r['m']:,}",str(r['inactive']),percent(r['fpr']),percent(r['private_detection']),percent(r['w0_private_detection'])])+' |'
        assert expected in text,expected
    for g in ('K2_M','V15_L'):
        for a in (.01,.001):
            expected='| '+' | '.join([g,f'{a*100:g}',percent(rate(g+'__F_15999',g,alpha=a)),percent(rate(spec.clean_id(g,15999),g,alpha=a))])+' |'
            assert expected in text
    # Audit the descriptive all-series summaries rather than accepting prose alone.
    ts=read('report_threshold_stability.json');d={}
    for r in ts:
        if r['method']!='LF0':d.setdefault((r['group'],r['method'],r['alpha'],r['branch']),{})[r['m']]=r['sd']
    ratio=[r[31999]/r[7999] for r in d.values()]
    assert len(ratio)==180 and all(x<1 for x in ratio) and f'{np.median(ratio):.4f}' in text
    ds=read('report_stability.json');d={}
    for r in ds:
        if r['method']!='LF0':d.setdefault((r['group'],r['method'],r['alpha']),{})[r['m']]=r['sd_fpr']
    ratio=[r[31999]/r[7999] for r in d.values()]
    assert len(ratio)==60 and sum(x<1 for x in ratio)==58 and f'{np.median(ratio):.4f}' in text
    figures=sorted((OUT/'figures').glob('*.png'));assert len(figures)==18
    for p in figures:
        with Image.open(p) as im:assert min(im.size)>500;im.verify()
    manifest=read('figure_manifest.json');assert manifest['cross_scale_pp']==[-100,100]
    assert len(manifest['cross'])==12 and len({(r['method'],r['alpha']) for r in manifest['cross']})==12
    links=0
    for p in OUT.glob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if target.startswith(('https:','#')):continue
            destination=(p.parent/target.split('#')[0]).resolve()
            if destination==OUT/'delivery_validation.json':continue
            assert destination.is_file(),(p.name,target)
            links+=1
    start=read('start_snapshot.json')
    for p,h in start['protected'].items():assert digest(ROOT/p)==h,p
    for row in read('archive_inventory.json'):assert digest(ROOT/row['path'])==row['sha256']
    ledger=(ROOT/'audits/ledger_2026-10.md').read_bytes()
    assert hashlib.sha256(ledger[:start['ledger_bytes']]).hexdigest()==start['ledger_sha256']
    outputs={str(p.relative_to(ROOT)):digest(p) for p in OUT.rglob('*') if p.is_file() and p.name!='delivery_validation.json'}
    sources={str(p.relative_to(ROOT)):digest(p) for p in [src,ROOT/'numerics/scripts/report_pr0007_calibration.py',Path(__file__).resolve()]}
    result=dict(status='PASS_DELIVERY',date='2026-10-18',date_basis='user-provided Asia/Tokyo date',
        numeric_series_checked=checked,primary_table_rows=8,main_budget_rows=6,saturation_table_rows=6,source_shift_rows=4,
        figure_count=18,links_checked=links,protected_unchanged=True,source_zip_unchanged=True,
        ledger_prefix=dict(bytes=start['ledger_bytes'],sha256=start['ledger_sha256']),
        outputs=outputs,sources=sources,
        navigation_hashes={p:digest(ROOT/p) for p in ('audits/README.md','audits/prereg/README.md','HANDOFF/LATEST.md','audits/ledger_2026-10.md')},
        scientific_records_generated=0,manuscript_edited=False,
        audit_type='Same-agent second calculation path and manual image review, not an independent reviewer',
        visual_review='Primary plot, W0 1% and LF0 0.1% cross grids, FPR, variability, threshold, saturation and tradeoff figures inspected.')
    (OUT/'delivery_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('PASS_DELIVERY',checked,'series;',links,'links; 18 figures')


if __name__=='__main__':main()
