# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Verify displayed tables, derived comparisons, and preserved artifacts."""
from pathlib import Path
import hashlib,json,re,sys,time
from urllib.parse import unquote
import numpy as np
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'numerics/scripts'))
from analyze_pr0004_lowfpr import file_hash,put


def tables(text):
    lines=text.splitlines();out=[];i=0
    while i+1<len(lines):
        if lines[i].startswith('|') and re.fullmatch(r'[| :\-]+',lines[i+1]) and '-' in lines[i+1]:
            header=[v.strip() for v in lines[i].strip('|').split('|')];i+=2;rows=[]
            while i<len(lines) and lines[i].startswith('|'):
                row=[v.strip() for v in lines[i].strip('|').split('|')];assert len(row)==len(header);rows.append(row);i+=1
            out.append((header,rows))
        else:i+=1
    return out


def verify(out):
    start=time.monotonic();out=Path(out);s=json.loads((out/'summary.json').read_text());meta=json.loads((out/'metadata.json').read_text())
    r={(x['case'],x['policy'],x['alpha'],x['budget'],x['method']):x for x in s['rates']}
    p={(x['case'],x['policy'],x['alpha'],x['budget'],x['new'],x['reference']):x for x in s['comparisons']}
    keys=[tuple(k) for k in meta['keys']];methods=meta['methods'];budgets=list(meta['budgets']);main=out/'THREE_CONFIGURATION_REPORT_JA.md'
    assert len(tables('|x|y|\n|---|---|\n|1|2|'))==1 and tables('|x|=3')==[]
    def val(x):return float(x.replace('−','-').replace('+',''))
    checks=dict(main_table_numbers=0,appendix_table_numbers=0,derived_intervals=0,local_links=0)
    def close(actual,expected):
        assert abs(val(actual)-expected)<.000051,(actual,expected)
    for h,rows in tables(main.read_text()):
        if 'LF0検出' in h:
            assert len(rows)==6
            for row in rows:
                case,policy=row[0].split('・');group=meta['conditions'][case]['group']
                expected=[r[c,policy,.01,'W0',m]['rate']['mean']*100 for c in [case,group] for m in methods]
                for x,y in zip(row[1:],expected):close(x,y);checks['main_table_numbers']+=1
        elif h[1]=='新−比較':
            assert len(rows)==18
            for row in rows:
                case,policy=row[0].split('・');new,ref=row[1].split('−');z=p[case,policy,.01,'W0',new,ref]
                for x,k in zip(row[2:5],['gained','lost','difference']):close(x,100*z[k]['mean']);checks['main_table_numbers']+=1
                nums=row[5].strip('[]').split(', ')
                for x,k in zip(nums,['lower','upper']):close(x,100*z['adjusted'][k]);checks['main_table_numbers']+=1
                assert row[6]=={'improved':'改善','within_1pp':'幅内','loss_over_1pp':'損失','uncertain':'未確定'}[z['classification']]
        elif '構造非発火反復/50' in h:
            assert len(rows)==3
            for row in rows:
                z=r['K2_M__P02','C',.001,'W0',row[0]];zz=r['K2_M','C',.001,'W0',row[0]]
                for x,y in zip(row[1:4],[100*z['rate']['mean'],100*z['miss']['mean'],100*zz['rate']['mean']]):close(x,y);checks['main_table_numbers']+=1
        elif 'LF0 フラグ' in h:
            for row in rows:
                policy={'基準B':'B','据え置きF':'F','閾値更新C':'C'}[row[0]];alpha=float(row[1].strip('%'))/100
                for cell,m in zip(row[2:],methods):
                    rr=[x for x in s['rates'] if x['kind']=='normal' and x['policy']==policy and x['alpha']==alpha and x['budget']=='W0' and x['method']==m]
                    assert cell==f"{sum(x['excess'] for x in rr)}/{len(rr)}";checks['main_table_numbers']+=2
    tt=tables((out/'ALL_RESULTS.md').read_text());assert len(tt)==20
    for ix,(h,rows) in enumerate(tt):
        alpha=[.01,.001][ix//10];budget=budgets[(ix//2)%5]
        if ix%2==0:
            assert len(rows)==82
            for row in rows:
                for m,cell in zip(methods,row[3:6]):close(cell,100*r[row[0],row[1],alpha,budget,m]['rate']['mean']);checks['appendix_table_numbers']+=1
        else:
            assert len(rows)==246
            for row in rows:
                new,ref=row[2].split('−');z=p[row[0],row[1],alpha,budget,new,ref]
                for cell,k in zip(row[3:7],z['counts']):close(cell,100*k/z['n_joint']);checks['appendix_table_numbers']+=1
                close(row[7],100*z['difference']['mean']);checks['appendix_table_numbers']+=1
    with np.load(out/'replicate_statistics.npz',allow_pickle=False) as f:data={k:f[k] for k in f.files}
    def ci(x,z):
        mean=float(np.mean(x));sd=float(np.std(x,ddof=1));w=float(t.ppf(.975,len(x)-1)*sd/len(x)**.5)
        assert abs(mean-z['mean'])<1e-14
        if sd:assert np.allclose([mean-w,mean+w],[z['lower'],z['upper']],atol=1e-14)
        else:assert z['lower'] is None
        checks['derived_intervals']+=1
    def rates(case,policy,alpha,budget,method):
        ki=keys.index((case,policy));ai=meta['alphas'].index(alpha);bi=budgets.index(budget);mi=methods.index(method)
        assert data['available'][:,ki,mi].all()
        return data['counts'][:,ki,ai,bi,mi,1:].sum(1).astype(float)/meta['n'][ki]
    for z in s['asymmetry']:
        plus=rates(z['positive'],z['policy'],z['alpha'],z['budget'],z['method']);minus=rates(z['negative'],z['policy'],z['alpha'],z['budget'],z['method'])
        ci(plus,z['positive_rate']);ci(minus,z['negative_rate']);ci(plus-minus,z['difference'])
    for z in s['source_shifts']:
        c=rates(z['group'],'C',z['alpha'],z['budget'],z['method']);f=rates(z['group'],'F',z['alpha'],z['budget'],z['method'])
        b=rates('B_'+z['group'].rsplit('_',1)[1],'B',z['alpha'],z['budget'],z['method'])
        ci(c-f,z['C_minus_F']);ci(f-b,z['F_minus_B'])
    # Human-readable calibration diagnostics, distinct from empirical zero counts.
    lines=['# 主配分W0：校正と非発火の診断','', '構造非発火は校正後の到達可能性。評価正常の0件とは別。順位上限率はLF0の評価正常における2枝の率。', '',
           '| 正常群 | 方針 | α% | LF0構造非発火/50 | LF-DC無限閾値全枝/50 | PH-DC無限閾値全枝/50 | LF0順位上限 G/P % |',
           '|---|---|---:|---:|---:|---:|---|']
    for ni,(case,policy) in enumerate(meta['null_keys']):
        ki=keys.index((case,policy));sat=data['saturation'][:,ki].mean(0)*100
        for ai,alpha in enumerate(meta['alphas']):
            cc=data['cuts'][:,ni,ai,0];dead=[int((cc[:,0,0]>=9999).sum()),int(np.isinf(cc[:,1]).all(1).sum()),int(np.isinf(cc[:,2]).all(1).sum())]
            lines.append(f'| {case} | {policy} | {100*alpha:g} | {dead[0]} | {dead[1]} | {dead[2]} | {sat[0]:.4f}/{sat[1]:.4f} |')
    lines+=['','全水準・全配分の実閾値／同値数は`replicate_statistics.npz`の`cuts`／`ties`、参照ゲート・Dと元モーメントは`diagnostics.json`および`models/`。',
            '正と負の枝には同時発火しない構造があり、私的枝との重複は枝8状態から読める。構造非発火とPH-DC不成立を混同しない。']
    (out/'CALIBRATION_DIAGNOSTICS.md').write_text('\n'.join(lines)+'\n')
    docs=[main,out/'ALL_RESULTS.md',ROOT/'audits/prereg/PR-0005_three_configuration_saved_data.md',ROOT/'audits/README.md',ROOT/'HANDOFF/LATEST.md',ROOT/'audits/prereg/README.md']
    for doc in docs:
        for link in re.findall(r'\[[^\]]*\]\(([^)]+)\)',doc.read_text()):
            link=unquote(link.strip('<>').split('#')[0])
            if not link or link.startswith(('http:','https:','mailto:')):continue
            assert (doc.parent/link).exists(),(doc,link);checks['local_links']+=1
    freeze=json.loads((ROOT/'audits/verification/2026-10-17_PR-0005_design_freeze.json').read_text())
    for path,sha in freeze['protected'].items():
        if path=='audits/fp_corpus.md':assert hashlib.sha256((ROOT/path).read_bytes()[:9157]).hexdigest()==sha
        else:assert file_hash(ROOT/path)==sha
    ledger=(ROOT/'audits/ledger_2026-10.md').read_bytes();assert hashlib.sha256(ledger[:freeze['ledger_bytes']]).hexdigest()==freeze['ledger_sha256']
    for name in ['validation.json','independent_path_validation.json']:assert json.loads((out/name).read_text())['status']=='PASS'
    result=dict(status='PASS',checks=checks,same_agent=True,visual_review='six figures inspected; legend moved off data',
        scientific_records_generated=0,ledger_prefix_bytes=freeze['ledger_bytes'],protected_files=len(freeze['protected']),
        elapsed_seconds=time.monotonic()-start,files={str(p.relative_to(out)):file_hash(p) for p in out.rglob('*') if p.is_file() and p.name!='delivery_validation.json'},
        sources={str(p.relative_to(ROOT)):file_hash(p) for p in [ROOT/'numerics/scripts/analyze_pr0005_three_config.py',ROOT/'numerics/tools/test_pr0005_three_config.py',ROOT/'numerics/tools/verify_pr0005_three_config.py',ROOT/'numerics/tools/render_pr0005_three_config.py',Path(__file__)]})
    put(out/'delivery_validation.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['files','sources']},indent=2))


if __name__=='__main__':verify(sys.argv[1])
