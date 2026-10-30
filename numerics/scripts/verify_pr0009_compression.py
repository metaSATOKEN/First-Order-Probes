# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Independent integer/run-length audit of saved LF representations and report.

No generation, fitting, new calibration, or evaluation-based selection.
Archive is reused only for authenticated read access, not analysis arithmetic.
"""
from pathlib import Path
import hashlib, json, re, time
import numpy as np
from analyze_pr0009_mechanisms import Archive

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'numerics/analysis/pr0009_9adfa7a8c6e0_compression_v1'
OLD=ROOT/'numerics/analysis/pr0009_9adfa7a8c6e0_two_axis_v1'
G=['B_M','B_L','K2_M','K200_L','V05_L','V15_L']
EV=['normal','GM05','GP05','P02'];CAL=['F','C'];AL=[.01,.001]
def h(b):return hashlib.sha256(b).hexdigest()
def load(n):return json.loads((OUT/n).read_text())
def npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def stats(actual,expected):
    x=np.asarray(actual,float)
    for k,v in dict(mean=x.mean(),sd=x.std(ddof=1),min=x.min(),max=x.max()).items():
        assert np.isclose(v,expected[k],atol=2e-14,rtol=2e-12),(k,v,expected[k])
def run_lengths(keys,labels):
    # Sort keys, then sum labels over contiguous runs; no inverse-key bincount.
    order=np.argsort(keys,kind='stable');keys=keys[order];y=labels[order].astype(np.int64)
    starts=np.r_[0,np.flatnonzero(keys[1:]!=keys[:-1])+1]
    total=np.diff(np.r_[starts,len(keys)]);yes=np.add.reduceat(y,starts);no=total-yes
    mixed=(yes!=0)&(no!=0)
    return [len(starts),int(mixed.sum()),int(total[mixed].sum()),int(np.minimum(yes,no).sum())]

def main():
    t=time.monotonic();valid=load('validation.json');assert valid['status']=='PASS_ANALYSIS'
    rows=load('comparison.json');asym=load('gain_asymmetry.json');extra=load('other_budgets.json')
    modelrows=load('models.json');models={(m['replicate'],m['base']):m for m in modelrows}
    assert len(models)==100 and len(rows)==96 and len(asym)==48 and len(extra)==576
    arc=Archive();allrep=[];joint_checks=0;collision_checks=0
    for r in range(50):
        folder=f'shards/{r//5:02d}/rep_{r:02d}';d=npz(OUT/'recomputed'/f'rep_{r:02d}.npz')
        prior=npz(OLD/'recomputed'/f'rep_{r:02d}.npz');allrep.append(d)
        cache={}
        for base in ['B_M','B_L']:
            mod=arc.npz(f'{folder}/models/{base}.npz');state=json.loads(str(mod['state']))['model']
            W=np.array(state['weights']);mu=np.array(state['mean']);ordered=mod['ordered'];m=models[r,base]
            assert np.array_equal(W,m['W']) and np.array_equal(mu,m['mean'])
            scaled=W*np.sqrt(np.diag(state['covariance']))
            assert np.linalg.matrix_rank(scaled)==2
            assert np.allclose(np.linalg.solve(W,W),np.eye(2),atol=1e-14,rtol=0)
            assert np.isclose(np.linalg.cond(scaled),m['scaled_condition'])
            cache[base]=(W,mu,ordered)
        for gi,g in enumerate(G):
            base='B_M' if g.endswith('_M') else 'B_L';W,mu,ordered=cache[base]
            for ci in range(2):
                for ai,minimum in enumerate([160,16]):
                    off=d['cal_top'][gi,ci,2]>=minimum
                    assert off==d['inactive'][gi,ci,ai]
                    assert off==(d['rank_cut'][gi,ci,ai]==9999)
                    assert np.all(np.isinf(d['cut'][gi,ci,ai,1]))==off
                    if not off:
                        j=d['rank_cut'][gi,ci,ai]
                        assert np.array_equal(d['cut'][gi,ci,ai,1],ordered[j,[0,0,1]])
            for ei,ev in enumerate(EV):
                case=g if ei==0 else g+'__'+ev;role='null_eval' if ei==0 else 'fault_eval'
                v=arc.values(folder,case,role);n=len(v);q=v[:,[11,12]]
                # Explicit scalar sum, rather than BLAS projection in analysis.
                sg=(q[:,0]-mu[0])*W[0,0]+(q[:,1]-mu[1])*W[0,1]
                sp=(q[:,0]-mu[0])*W[1,0]+(q[:,1]-mu[1])*W[1,1]
                rg=np.searchsorted(ordered[:,0],abs(sg),side='left')
                rp=np.searchsorted(ordered[:,1],sp,side='left')
                pair=rg*10000+rp;keys=[2*pair+(sg>=0),pair,np.maximum(rg,rp)]
                bits=np.concatenate([arc.npz(f'{folder}/decisions/{case}/{start:09d}.npz')['packed'][:,:2]
                                     for start in range(0,n,4096)])
                for ci in range(2):
                    for ai in range(2):
                        b=bits[:,ci,ai];assert np.all((b&8)!=0) and np.all(((b>>20)&8)!=0)
                        dc=(b&7);lf=(b>>20)&7
                        joint=np.zeros((8,8),np.int64)
                        np.add.at(joint,(dc,lf),1)
                        assert np.array_equal(joint,d['joint'][gi,ci,ei,ai]);joint_checks+=1
                        assert np.array_equal(joint.sum(1),d['hist'][gi,ci,ei,ai,0])
                        assert np.array_equal(joint.sum(0),d['hist'][gi,ci,ei,ai,1])
                        assert int((dc>0).sum()-(lf>0).sum())==int(joint[1:,0].sum()-joint[0,1:].sum())
                        for ki,key in enumerate(keys):
                            assert np.array_equal(run_lengths(key,dc>0),d['collisions'][gi,ci,ei,ai,ki])
                            collision_checks+=1
                        for mi in range(6):
                            nib=(b>>(4*mi))&15;assert np.all(nib>=8)
                            counted=np.array([np.count_nonzero((nib&7)==p) for p in range(8)])
                            assert np.array_equal(counted,d['extra_hist'][gi,ci,ei,ai,mi])
                            assert np.array_equal(counted,prior['counts'][gi,ei,ci,ai,mi])
                        # Saved LF0 is also reproduced by strict rank threshold.
                        cr=d['rank_cut'][gi,ci,ai];assert np.array_equal((np.maximum(rg,rp)>cr),lf>0)
        if (r+1)%10==0:print(f'{r+1}/50: original E bits, run-length collisions, old counts match',flush=True)
    a={k:np.stack([d[k] for d in allrep]) for k in allrep[0]}
    assert arc.records==10500000 and joint_checks==4800 and collision_checks==14400
    assert np.all(np.diff(a['collisions'][...,-1],axis=-1)>=0)
    ix={}
    for row in rows:
        gi=G.index(row['group']);ci=CAL.index(row['calibration']);ai=AL.index(row['alpha']);ei=EV.index(row['evaluation']);n=row['n_per_rep']
        jt=a['joint'][:,gi,ci,ei,ai];ht=a['hist'][:,gi,ci,ei,ai]
        yes=ht[...,1:].sum(-1);gain=jt[:,1:,0].sum(1);loss=jt[:,0,1:].sum(1)
        mask=np.ones((8,8),bool);mask[0,:]=False;mask[:,0]=False;np.fill_diagonal(mask,False)
        for key,values in [('W0',yes[:,0]),('LF0',yes[:,1]),('difference',yes[:,0]-yes[:,1]),
                           ('W0_only',gain),('LF0_only',loss),('OR_masks_pattern_difference',jt[:,mask].sum(1))]:
            stats(values/n,row[key])
        assert np.array_equal(jt.sum(0),row['pattern_joint'])
        assert np.array_equal(ht[:,0].sum(0),row['W0_patterns']) and np.array_equal(ht[:,1].sum(0),row['LF0_patterns'])
        off=a['inactive'][:,gi,ci,ai];assert int(off.sum())==row['LF0_inactive']
        assert int(gain[off].sum())==row['W0_only_in_inactive']
        assert np.array_equal(a['collisions'][:,gi,ci,ei,ai].sum(0),row['collision_sums'])
        assert np.allclose((yes[:,0]-yes[:,1])/n,row['per_rep_difference'],rtol=0,atol=2e-16)
        ix[row['group'],row['calibration'],row['alpha'],row['evaluation']]=row
    for row in asym:
        gi=G.index(row['group']);ci=CAL.index(row['calibration']);ai=AL.index(row['alpha']);mi=['W0','LF0'].index(row['method'])
        diff=a['hist'][:,gi,ci,2,ai,mi]-a['hist'][:,gi,ci,1,ai,mi]
        g=diff[:,[1,2,3,5,6,7]].sum(1);p=diff[:,4];total=diff[:,1:].sum(1)
        assert np.array_equal(g+p,total)
        for key,values in [('GP_minus_GM',total),('G_component',g),('P_only_component',p)]:stats(values/5000,row[key])
    # Names in the archived spec, not an assumed spelling of the auxiliary methods.
    from pr0009_spec import METHODS as saved_methods
    for row in extra:
        gi=G.index(row['group']);ci=CAL.index(row['calibration']);ai=AL.index(row['alpha']);ei=EV.index(row['evaluation']);mi=list(saved_methods).index(row['method'])
        n=20000 if ei==0 else 5000
        stats(a['extra_hist'][:,gi,ci,ei,ai,mi,1:].sum(1)/n,row['rate'])
    for source in load('source_summary.json'):
        gi=G.index(source['group'])
        assert np.allclose(source['normal_mean'],a['mean'][:,gi,0].mean(0),atol=1e-14)
        assert np.allclose(source['normal_sd_mean'],a['sd'][:,gi,0].mean(0),atol=1e-14)
    # Deterministic boundary/negative controls, not scientific observations.
    ref=np.array([0.,1.,1.,2.]);x=np.array([-1.,0.,.5,1.,1.5,2.,3.])
    rank=np.searchsorted(ref,x,side='left')
    for c in range(5):
        threshold=ref[c] if c<4 else np.inf
        assert np.array_equal(rank>c,x>threshold)
    assert np.any((rank>1)!=(x>=ref[1]))
    assert run_lengths(np.array([0,0,1]),np.array([0,1,1]))==[2,1,2,1]
    assert run_lengths(np.array([0,0,1]),np.array([0,0,1]))==[2,0,0,0]
    report=(OUT/'PR0009_COMPRESSION_REPORT_JA.md').read_text()
    full=(OUT/'ALL_COMPRESSION_RESULTS.md').read_text();published=0
    for row in rows:
        p=row['pattern_joint'];neither=p[0][0]/(50*row['n_per_rep'])
        vals=[row[k]['mean'] for k in ['W0','LF0','difference']]+[row['difference']['sd']]+[row[k]['mean'] for k in ['W0_only','LF0_only']]+[neither,row['OR_masks_pattern_difference']['mean']]
        expected='| '+' | '.join([row['group'],row['calibration'],str(row['alpha']),row['evaluation']]+[f'{100*v:.4f}' for v in vals]+[str(row['LF0_inactive'])])+' |'
        assert expected in full;published+=1
        expected='| '+' | '.join([row['group'],row['calibration'],str(row['alpha']),row['evaluation']]+['/'.join(str(q[k]) for q in row['collision_sums']) for k in [1,2,3]])+' |'
        assert expected in full;published+=1
    for row in asym:
        expected='| '+' | '.join([row['group'],row['calibration'],str(row['alpha']),row['method']]+[f"{100*row[k]['mean']:.4f}" for k in ['GP_minus_GM','G_component','P_only_component']])+' |'
        assert expected in full;published+=1
    for row in extra:
        expected='| '+' | '.join([row['group'],row['calibration'],str(row['alpha']),row['evaluation'],row['method'],f"{100*row['rate']['mean']:.4f}",f"{100*row['rate']['sd']:.4f}"])+' |'
        assert expected in full;published+=1
    assert published==816
    # Literal main-table numbers, including typographic minus, must follow data.
    normalized=report.replace('−','-')
    from pr0007_spec import conditions
    conditions_now=conditions()
    for gi,g in enumerate(G):
        p=conditions_now[g]['params']
        assert f'| {g} | {p["kappa"]:g} | {p["variance"]:g} | {p["T"]:.3f} |' in report
        sd=a['sd'][:,gi,0].mean(0)
        assert f'| {g} | {sd[2]:.4f} | {sd[3]:.4f} |' in report
        top=a['cal_top'][:,gi,1,2]
        expected=f'| {g} | {top.mean():.2f} | {top.min()}〜{top.max()} | {a["inactive"][:,gi,0,1].sum()} | {a["inactive"][:,gi,1,1].sum()} |'
        assert expected in report
    for g,alpha,ev in [('K2_M',.001,'P02'),('K200_L',.001,'P02'),('V05_L',.01,'GM05'),('V05_L',.01,'GP05'),('V15_L',.001,'P02')]:
        row=ix[g,'C',alpha,ev];normal=ix[g,'C',alpha,'normal']
        vals=[f'{row[k]["mean"]*100:.4f}' for k in ['W0','LF0']]+[f'{row["difference"]["mean"]*100:+.4f}']+[f'{row[k]["mean"]*100:.4f}' for k in ['W0_only','LF0_only']]
        expected=f'| {g}・{ev} | {alpha*100:g}% | '+' | '.join(vals)+f' | {normal["W0"]["mean"]*100:.4f}/{normal["LF0"]["mean"]*100:.4f} |'
        assert expected in normalized,expected
    for g,alpha,ev in [('K2_M',.001,'P02'),('K200_L',.001,'P02'),('V05_L',.01,'GP05')]:
        row=ix[g,'C',alpha,ev]
        assert f'| {g}・{ev} | {alpha*100:g}% | '+' | '.join(f'{c[3]:,}' for c in row['collision_sums'])+' |' in report
    for row in asym:
        if row['calibration']=='C' and row['alpha']==.01 and row['group'] in ['B_L','K2_M','V05_L']:
            assert '| '+' | '.join([row['group'],row['method']]+[f"{100*row[k]['mean']:.4f}" for k in ['GP_minus_GM','G_component','P_only_component']])+' |' in report
    for ei,label in [(1,'-5%ゲイン'),(2,'+5%ゲイン')]:
        shift=(a['mean'][:,4,ei]-a['mean'][:,4,0]).mean(0)
        assert f'| {label} | {shift[2]:+.4f} | {shift[3]:+.4f} |' in normalized
    for text in [report,full]:
        for block in re.findall(r'(?:^\|.*\n)+',text,re.M):
            assert len(set(len(line.split('|')) for line in block.splitlines()))==1
    links=0
    for p in OUT.glob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if '://' in target or target.startswith('#'):continue
            dest=(p.parent/target.split('#')[0]).resolve()
            if dest.name in ['delivery_validation.json','closeout_validation.json']:continue
            assert dest.exists(),(p,target);links+=1
    figures=sorted((OUT/'figures').glob('*.png'));assert len(figures)==7
    for p in figures:assert p.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    snap=load('start_snapshot.json')
    for n,digest in snap['protected'].items():assert h((ROOT/n).read_bytes())==digest,n
    ledger=(ROOT/'audits/ledger_2026-10.md').read_bytes()
    assert h(ledger[:snap['ledger_bytes']])==snap['ledger_sha256']
    outputs={str(p.relative_to(ROOT)):h(p.read_bytes()) for p in OUT.rglob('*') if p.is_file() and p.name not in ['delivery_validation.json','closeout_validation.json']}
    sources={str(p.relative_to(ROOT)):h(p.read_bytes()) for p in (ROOT/'numerics/scripts').glob('*pr0009_compression.py')}
    assert len(sources)==3
    result=dict(status='PASS_DELIVERY',same_agent_self_check=True,independent_reviewer=False,
        original_e_records_reread=arc.records,original_archives_rehashed=12,authenticated_members=len(arc.checked),
        joint_tables=joint_checks,run_length_collision_checks=collision_checks,comparison_rows=96,asymmetry_rows=48,
        auxiliary_rows=576,appendix_rows=published,figures=7,existing_local_links_checked=links,
        deterministic_boundary_negative_controls='PASS',protected_files=len(snap['protected']),ledger_prefix_preserved=True,
        outputs=outputs,sources=sources,elapsed_seconds=time.monotonic()-t)
    (OUT/'delivery_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('PASS_DELIVERY',joint_checks,'joint tables;',collision_checks,'collision checks;',published,'published rows',flush=True)

if __name__=='__main__':main()
