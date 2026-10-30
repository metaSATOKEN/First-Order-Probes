# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Check report numbers, saved arrays, packaged evidence and no-op restart."""
from pathlib import Path
import hashlib,json,re,subprocess,sys,zipfile
import numpy as np
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'numerics/analysis/pr0011_9adfa7a8c6e0_recovery_v2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def main():
    s=load(OUT/'schema.json');rr=load(OUT/'rates_all.json');pp=load(OUT/'paired_all.json')
    names=['counts','paired','actions','v_labels','old_labels','e_labels','base']
    d={k:np.stack([np.load(OUT/'recomputed'/f'rep_{r:02d}.npz')[k] for r in range(50)]) for k in names}
    keys=lambda z:(s['groups'].index(z['group']),next(i for i,c in enumerate(s['candidates']) if c['id']==z['candidate']),s['gates'].index(z['gate']),s['alphas'].index(z['alpha']),s['evaluations'].index(z['evaluation']))
    lut={};pairs={};tc=t.ppf(.975,49)
    for z in rr:
        g,k,h,a,e=keys(z);p=s['policies'].index(z['policy']);x=d['counts'][:,g,k,h,a,p,e];n=z['per_rep_N']
        assert np.all(x.sum(1)==n) and np.array_equal(x.sum(0),z['totals'])
        for col,name in enumerate(['alarm','no_alarm','defer','invalid']):
            v=x[:,col]/n;m=v.mean();err=tc*v.std(ddof=1)/np.sqrt(50)
            assert np.allclose([m,m-err,m+err],[z[name]['mean'],*z[name]['ci95']],atol=1e-14,rtol=0)
        aa=x[:,0];bb=x[:,:2].sum(1);q=z['issued_rate']
        if bb.sum()==0:assert q['value'] is None and q['ci95'] is None
        else:
            theta=aa.sum()/bb.sum();assert abs(theta-q['value'])<1e-14
            u=aa-theta*bb
            if q['ci95'] is not None:
                se=u.std(ddof=1)/(np.sqrt(50)*bb.mean());assert np.allclose(q['ci95'],[theta-tc*se,theta+tc*se],atol=1e-14,rtol=0)
        lut[g,k,h,a,p,e]=z
    for z in pp:
        g,k,h,a,e=keys(z);pi=s['pairs'].index([s['policies'].index(z['left']),s['policies'].index(z['right'])])
        j=d['paired'][:,g,k,h,a,pi,e];n=z['per_rep_N'];assert np.array_equal(j.sum(0),z['table'])
        gain=j[:,0,1:].sum(1)/n;loss=j[:,1:,0].sum(1)/n
        for v,name in [(gain,'gain_alarm'),(loss,'loss_alarm'),(gain-loss,'alarm_difference')]:
            assert abs(v.mean()-z[name]['mean'])<1e-14
        pairs[g,k,h,a,pi,e]=z
    full=(OUT/'ALL_CONDITIONS.md').read_text().splitlines()
    full=[x for x in full if x.startswith('|') and x.split('|')[1] in s['groups']]
    assert len(full)==len(rr)==15984
    for line,z in zip(full,rr):
        vals=line.strip('|').split('|')
        assert vals[:6]==[z['group'],z['candidate'],z['gate'],str(z['alpha']),z['policy'],z['evaluation']]
        expect=[100*z[k]['mean'] for k in ['alarm','no_alarm','defer','invalid','coverage']]
        assert np.allclose(list(map(float,vals[6:11])),expect,atol=.000501,rtol=0)
    text=(OUT/'PR0011_STOP_RECOVERY_REPORT_JA.md').read_text();linechecks=0
    bodyrows=[l.strip('|').split('|') for l in text.splitlines() if l.startswith('|')]
    def equal_display(a,b):
        if str(a)==str(b):return True
        try:return abs(float(str(a).replace('−','-'))-float(str(b).replace('−','-')))<=.000100001
        except ValueError:return False
    def has(cells):
        nonlocal linechecks
        line='|'+'|'.join(map(str,cells))+'|'
        assert any(len(r)==len(cells) and all(equal_display(x,y) for x,y in zip(r,cells)) for r in bodyrows),line
        linechecks+=1
    assert equal_display('0.9468','0.9467') and not equal_display('0.9478','0.9467')
    for g,group in enumerate(s['groups']):
        labs=[]
        for a in range(2):
            for c in range(2):labs.append('/'.join(map(str,np.bincount(d['v_labels'][:,g,c,a],minlength=4)[:3])))
        has([group,*labs])
        for p in ([0] if g in [3,4] else range(3)):
            x=[lut[g,36,2,0,p,e] for e in range(4)]
            has([group,'P0/P1/P2' if g in [3,4] else f'P{p}',f"{100*x[0]['coverage']['mean']:.0f}",*[f"{100*z['alarm']['mean']:.4f}" for z in x]])
        cov='/'.join(f"{100*lut[g,36,2,1,p,0]['coverage']['mean']:.0f}" for p in range(3))
        has([group,cov,*[f"{100*lut[g,36,2,1,2,e]['alarm']['mean']:.4f}" for e in range(4)]])
    strata=load(OUT/'recovery_strata_triple800.json')
    for g,group in enumerate(s['groups']):
        for a,alpha in enumerate(s['alphas']):
            x=[z for z in strata if z['group']==group and z['alpha']==alpha]
            assert len(x)==4
            if x[0]['recovery_replicates']:
                has(['1%' if a==0 else '0.1%',group,x[0]['recovery_replicates'],*[f"{100*z['recovery_issued']['value']:.4f}" for z in x]])
            for e,z in enumerate(x):
                rec=d['actions'][:,g,36,2,a,2]==2;df=d['actions'][:,g,36,2,a,2]==3
                j=d['paired'][:,g,36,2,a,2,e]
                assert z['C0_switch_gain']==int(j[rec,0,1:].sum())
                assert z['C0_switch_loss']==int(j[rec,1:,0].sum())
                assert z['defer_lost_P0_alarm']==int(j[df,2,0].sum())
                assert z['defer_lost_P0_noalarm']==int(j[df,2,1].sum())
    for g in [2,5]:
        for e,ev in enumerate(['負ゲイン','正ゲイン','私的入力'],1):
            z=pairs[g,36,2,0,2,e];ci=np.array(z['alarm_difference']['ci95'])*100
            cell=f"{100*z['alarm_difference']['mean']:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]".replace('-','−')
            has([s['groups'][g]+'・'+ev,f"{100*z['gain_alarm']['mean']:.4f}",f"{100*z['loss_alarm']['mean']:.4f}",cell])
    npz_members=0;reps=set();zips={}
    for p in sorted((OUT/'bundles').glob('*.zip')):
        with zipfile.ZipFile(p) as z:
            assert z.testzip() is None
            for n in z.namelist():assert hashlib.sha256(z.read(n)).hexdigest()==sha(OUT/n)
            r={int(Path(n).stem.split('_')[1]) for n in z.namelist()};assert len(r)==5 and not reps&r;reps|=r
            npz_members+=sum(n.endswith('.npz') for n in z.namelist())
        assert p.stat().st_size<128000000;zips[p.name]=dict(bytes=p.stat().st_size,sha256=sha(p))
    assert reps==set(range(50)) and len(zips)==10
    before={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()}
    run=subprocess.run([sys.executable,str(ROOT/'numerics/scripts/analyze_pr0011_recovery.py')],capture_output=True,text=True)
    assert run.returncode==0 and 'COMPLETE_VERIFIED_NOOP' in run.stdout
    after={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()};assert before==after
    for rel in re.findall(r'\]\(([^)]+)\)',text):
        p=OUT/rel
        if p.name not in ['delivery_validation.json','closeout_validation.json']:assert p.exists(),rel
    snap=load(OUT/'start_snapshot.json')
    for p,h in {**snap['protected'],**snap['inputs']}.items():assert sha(ROOT/p)==h,p
    out=dict(status='PASS_DELIVERY',all_rate_rows=len(rr),paired_rows=len(pp),all_condition_MD_rows=len(full),
        main_report_table_rows=linechecks,recovery_strata=48,zip_members_npz=npz_members,bundles=zips,
        complete_noop=run.stdout.strip(),code_sha256=sha(__file__),report_sha256=sha(OUT/'PR0011_STOP_RECOVERY_REPORT_JA.md'),
        protected_count=len(snap['protected']),scientific_missing=0,scientific_excluded=0,independent_reviewer=False)
    (OUT/'delivery_validation.json').write_text(json.dumps(out,indent=2)+'\n')
    print('DELIVERY PASS',linechecks,'report rows;',len(rr),'full rows; NOOP exact')
if __name__=='__main__':main()
