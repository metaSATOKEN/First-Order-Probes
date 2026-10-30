# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Saved-data audit of LF representations; no generator or detector fitting."""
from pathlib import Path
import hashlib,json,time
import numpy as np
from analyze_pr0009_mechanisms import Archive
import pr0009_spec as spec
from pr0009_storage import canonical

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'numerics/analysis/pr0009_9adfa7a8c6e0_two_axis_v1'
PREV=ROOT/'numerics/analysis/pr0009_9adfa7a8c6e0_v_budget_v1'
OUT=ROOT/'numerics/analysis/pr0009_9adfa7a8c6e0_compression_v1'
G=list(spec.groups());EV=['normal','GM05','GP05','P02'];CAL=['F','C'];AL=[.01,.001]
PROBS=np.array([.001,.01,.5,.99,.995,.9975,.999,.9995,.99975])
def h(b):return hashlib.sha256(b).hexdigest()
def load(p):return json.loads(p.read_text())
def safe(x):
    if isinstance(x,np.ndarray):return safe(x.tolist())
    if isinstance(x,np.generic):return safe(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    if isinstance(x,dict):return {k:safe(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [safe(v) for v in x]
    return x
def dump(n,v):(OUT/n).write_text(json.dumps(safe(v),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def stat(x):
    a=np.asarray(x,float);return dict(mean=float(a.mean()),sd=float(a.std(ddof=1)),min=float(a.min()),max=float(a.max()))
def project(q,mod):return (q-np.array(mod['mean']))@np.array(mod['weights']).T
def ranks(s,ordered):return np.column_stack([np.searchsorted(ordered[:,0],np.abs(s[:,0]),side='left'),np.searchsorted(ordered[:,1],s[:,1],side='left')])
def pattern(branch):return (branch.astype(np.int64)*[1,2,4]).sum(1)
def rank_keys(r,s,nref):
    pair=r[:,0]*(nref+1)+r[:,1]
    return [np.unique(k,return_inverse=True) for k in [pair*2+(s[:,0]>=0),pair,r.max(1)]]
def collision(key,alarm):
    unique,inv=key;total=np.bincount(inv);yes=np.bincount(inv,weights=alarm).astype(np.int64);no=total-yes
    mixed=(yes>0)&(no>0)
    return np.array([len(unique),mixed.sum(),total[mixed].sum(),np.minimum(yes,no).sum()],np.int64)

def main():
    t=time.monotonic();assert not (OUT/'validation.json').exists(),'New output version required'
    (OUT/'recomputed').mkdir(exist_ok=True)
    previous=load(PREV/'closeout_validation.json');snap=load(PREV/'start_snapshot.json')
    protected=dict(snap['protected'])
    # Navigation is mutable, but corpus/prereg/requirements and all historical outputs stay protected.
    for key in ['outputs','sources','frozen_sources']:protected.update(previous.get(key,{}))
    for n,digest in protected.items():assert h((ROOT/n).read_bytes())==digest,n
    ledger=(ROOT/'audits/ledger_2026-10.md').read_bytes()
    dump('start_snapshot.json',dict(protected=protected,ledger_bytes=len(ledger),ledger_sha256=h(ledger),protocol_sha256=h((OUT/'PROTOCOL.md').read_bytes())))
    arc=Archive();models=[];allrep=[];examples={};max_reconstruction=0.;max_equivalence=0.;raw_e=0;raw_cal=0;checked_bits=0
    for rep in range(50):
        folder=f'shards/{rep//5:02d}/rep_{rep:02d}';frozen=arc.js(folder+'/CALIBRATION_FROZEN.json');fh=h(canonical(frozen).encode())
        with np.load(OLD/'recomputed'/f'rep_{rep:02d}.npz') as z:old={k:z[k] for k in z.files}
        d=dict(hist=np.zeros((6,2,4,2,2,8),np.int64),joint=np.zeros((6,2,4,2,8,8),np.int64),
            collisions=np.zeros((6,2,4,2,3,4),np.int64),top=np.zeros((6,4,3),np.int64),
            cal_top=np.zeros((6,2,3),np.int64),rank_cut=np.zeros((6,2,2),np.int64),
            cut=np.zeros((6,2,2,2,3)),inactive=np.zeros((6,2,2),bool),
            mean=np.zeros((6,4,4)),sd=np.zeros((6,4,4)),cov=np.zeros((6,4,4,4)),
            quantile=np.zeros((6,4,len(PROBS),4)),cal_quantile=np.zeros((6,2,len(PROBS),3)),
            extra_hist=np.zeros((6,2,4,2,6,8),np.int64))
        cache={}
        for base in ['B_M','B_L']:
            a=arc.npz(f'{folder}/models/{base}.npz');state=json.loads(str(a['state']));assert state['valid']
            mod=state['model'];ordered=a['ordered'];assert ordered.shape==(9999,2) and np.all(np.diff(ordered,axis=0)>=0)
            assert not mod['disabled'],'Disabled direction requires explicit separate analysis'
            W=np.array(mod['weights']);scale=np.sqrt(np.diag(mod['covariance']));Ws=W*scale[None,:]
            rank=int(np.linalg.matrix_rank(Ws));assert rank==2,'Noninvertible model must not be silently excluded'
            modelinfo=dict(replicate=rep,base=base,rank=rank,W=W,mean=mod['mean'],q_scale=scale,
                scaled_condition=float(np.linalg.cond(Ws)),determinant=float(np.linalg.det(Ws)),
                row_cosine=float(np.dot(Ws[0],Ws[1])/np.linalg.norm(Ws[0])/np.linalg.norm(Ws[1])))
            models.append(modelinfo)
            oldcal=arc.npz(f'{folder}/old_cal/{base}.npz');cv=arc.values(folder,base,'old_cal');raw_cal+=len(cv)
            cache[base]=(mod,ordered,oldcal,cv)
            if rep==0:examples[base+'_model']=np.array(mod['weights']);examples[base+'_mean']=np.array(mod['mean']);examples[base+'_scale']=scale
        for gi,group in enumerate(G):
            mod,ordered,oldcal,oldv=cache[spec.baseline(group)];W=np.array(mod['weights']);mu=np.array(mod['mean']);scale=np.sqrt(np.diag(mod['covariance']))
            newcal=arc.npz(f'{folder}/calibration/{group}/candidate_0.npz');newv=arc.values(folder,group,'update_normal');raw_cal+=len(newv)
            for ci,(cal,cv) in enumerate([(oldcal,oldv),(newcal,newv)]):
                meta=json.loads(str(cal['metadata']));assert meta['valid'] and len(cv)==15999
                s=project(cv[:,[11,12]],mod);z=np.column_stack((s[:,0],-s[:,0],s[:,1]));rk=ranks(s,ordered);maximum=rk.max(1)
                d['cal_top'][gi,ci]=[(rk[:,0]==9999).sum(),(rk[:,1]==9999).sum(),(maximum==9999).sum()]
                d['cal_quantile'][gi,ci]=np.quantile(z,PROBS,axis=0)
                for ai,den in enumerate([100,1000]):
                    k=16000-16000//den;cr=int(cal['cut'][ai,5,0]);assert np.all(cal['cut'][ai,5]==cr)
                    assert np.partition(maximum,k-1)[k-1]==cr
                    inactive=cr>=9999
                    assert inactive==(d['cal_top'][gi,ci,2]>=15999-k+1)
                    assert inactive==meta['lf0_structurally_inactive'][ai]
                    lf=np.full(3,np.inf) if inactive else np.array([ordered[cr,0],ordered[cr,0],ordered[cr,1]])
                    dc=cal['cut'][ai,0]
                    for b,part in enumerate([2,2,4]):
                        kb=16000-16000*part//(8*den)
                        assert cal['indices'][ai,0,b]==kb
                        assert np.allclose(np.partition(z[:,b],kb-1)[kb-1],dc[b],rtol=2e-12,atol=2e-13)
                    d['cut'][gi,ci,ai]=[dc,lf];d['rank_cut'][gi,ci,ai]=cr;d['inactive'][gi,ci,ai]=inactive
                    assert inactive==bool(old['inactive'][gi,ci,ai])
            for ei,ev in enumerate(EV):
                case=group if ei==0 else group+'__'+ev;role='null_eval' if ei==0 else 'fault_eval'
                values=arc.values(folder,case,role);raw_e+=len(values);n=len(values);q=values[:,[11,12]];s=project(q,mod)
                nu=spec.nominal_nu();assert np.allclose(q[:,0],values[:,8]-(1-nu)*values[:,6]-nu*values[:,7],rtol=2e-10,atol=2e-13)
                assert np.allclose(q[:,1],values[:,14]+values[:,15]-2*values[:,16],rtol=2e-10,atol=2e-13)
                restored=s@np.linalg.inv(W).T+mu
                err=float(np.max(np.abs((restored-q)/scale)));max_reconstruction=max(max_reconstruction,err);assert err<1e-10
                signed_scalar=np.column_stack([(q[:,0]-mu[0])*w[0]+(q[:,1]-mu[1])*w[1] for w in W])
                max_equivalence=max(max_equivalence,float(np.max(np.abs(signed_scalar-s))))
                features=np.column_stack((q,s));d['mean'][gi,ei]=features.mean(0);d['sd'][gi,ei]=features.std(0,ddof=1)
                d['cov'][gi,ei]=np.cov(features,rowvar=False,ddof=1);d['quantile'][gi,ei]=np.quantile(features,PROBS,axis=0)
                rk=ranks(s,ordered);z=np.column_stack((s[:,0],-s[:,0],s[:,1]));zs=np.column_stack((signed_scalar[:,0],-signed_scalar[:,0],signed_scalar[:,1]))
                d['top'][gi,ei]=[(rk[:,0]==9999).sum(),(rk[:,1]==9999).sum(),(rk.max(1)==9999).sum()]
                keys=rank_keys(rk,s,9999)
                bits=[]
                for start in range(0,n,4096):
                    a=arc.npz(f'{folder}/decisions/{case}/{start:09d}.npz');md=json.loads(str(a['metadata']));bb=a['packed']
                    assert md['task']['case']==case and md['task']['role']==role and md['task']['replicate']==rep
                    assert md['start']==start and md['calibration_sha256']==fh and md['raw_sha256']==h(values[start:start+len(bb)].tobytes())
                    assert md['packed_sha256']==h(bb.tobytes());bits.append(bb[:,:2])
                bits=np.concatenate(bits);assert bits.shape==(n,2,2)
                for ci in range(2):
                    for ai in range(2):
                        pats=[]
                        for mi,shift in enumerate([0,20]):
                            nib=(bits[:,ci,ai]>>shift)&15;assert np.all(nib&8)
                            saved=(nib&7).astype(np.int64);br=z>d['cut'][gi,ci,ai,mi];pp=pattern(br)
                            assert np.array_equal(pp,saved),(rep,group,ev,ci,ai,mi,'continuous halfplanes')
                            assert np.array_equal(pattern(zs>d['cut'][gi,ci,ai,mi]),saved)
                            if mi==1:
                                cut=d['rank_cut'][gi,ci,ai];bg=rk[:,0]>cut;bp=rk[:,1]>cut
                                pr=pattern(np.column_stack((bg&(s[:,0]>=0),bg&(s[:,0]<0),bp)))
                                assert np.array_equal(pr,pp) and np.array_equal(rk.max(1)>cut,pp>0)
                            pats.append(pp);d['hist'][gi,ci,ei,ai,mi]=np.bincount(pp,minlength=8);checked_bits+=n
                        jt=np.bincount(8*pats[0]+pats[1],minlength=64).reshape(8,8);d['joint'][gi,ci,ei,ai]=jt
                        for ki,key in enumerate(keys):d['collisions'][gi,ci,ei,ai,ki]=collision(key,pats[0]>0)
                        assert np.all(np.diff(d['collisions'][gi,ci,ei,ai,:,3])>=0)
                        for mi in range(6):
                            pp=((bits[:,ci,ai]>>(mi*4))&7).astype(int);d['extra_hist'][gi,ci,ei,ai,mi]=np.bincount(pp,minlength=8)
                assert np.array_equal(d['extra_hist'][gi,:,ei],old['counts'][gi,ei,:2])
                if rep==0:examples[group+'__'+ev]=features[:512]
        np.savez_compressed(OUT/'recomputed'/f'rep_{rep:02d}.npz',**d);allrep.append(d)
        if (rep+1)%5==0:print(f'{rep+1}/50: q, projections, rank/halfplanes, F/C branches matched',flush=True)
    a={k:np.stack([d[k] for d in allrep]) for k in allrep[0]};rows=[];asym=[];extra=[];source=[]
    for gi,group in enumerate(G):
        for ci,cal in enumerate(CAL):
            for ai,alpha in enumerate(AL):
                for ei,ev in enumerate(EV):
                    n=20000 if ei==0 else 5000;ht=a['hist'][:,gi,ci,ei,ai];jt=a['joint'][:,gi,ci,ei,ai]
                    rates=ht[:,: ,1:].sum(-1)/n;gain=jt[:,1:,0].sum(1)/n;loss=jt[:,0,1:].sum(1)/n
                    assert np.allclose(rates[:,0]-rates[:,1],gain-loss,atol=2e-14,rtol=0)
                    masked=(jt[:,1:,1:].sum((1,2))-np.trace(jt,axis1=1,axis2=2)+jt[:,0,0])/n
                    row=dict(group=group,calibration=cal,alpha=alpha,evaluation=ev,n_per_rep=n,
                        W0=stat(rates[:,0]),LF0=stat(rates[:,1]),difference=stat(rates[:,0]-rates[:,1]),
                        W0_only=stat(gain),LF0_only=stat(loss),OR_masks_pattern_difference=stat(masked),
                        pattern_joint=jt.sum(0),W0_patterns=ht[:,0].sum(0),LF0_patterns=ht[:,1].sum(0),
                        LF0_inactive=int(a['inactive'][:,gi,ci,ai].sum()),
                        W0_only_in_inactive=int(jt[a['inactive'][:,gi,ci,ai],1:,0].sum()),
                        W0_zero_replicates=int((rates[:,0]==0).sum()),LF0_zero_replicates=int((rates[:,1]==0).sum()),
                        collision_sums=a['collisions'][:,gi,ci,ei,ai].sum(0),
                        per_rep_difference=rates[:,0]-rates[:,1])
                    rows.append(row)
                    for mi,meth in enumerate(spec.METHODS):extra.append(dict(group=group,calibration=cal,alpha=alpha,evaluation=ev,method=meth,
                        rate=stat(a['extra_hist'][:,gi,ci,ei,ai,mi,1:].sum(-1)/n)))
                for mi,method in enumerate(['W0','LF0']):
                    ph=a['hist'][:,gi,ci,2,ai,mi];mh=a['hist'][:,gi,ci,1,ai,mi]
                    gb=np.array([(k&3)>0 for k in range(8)]);po=np.array([k==4 for k in range(8)])
                    dg=(ph[:,gb].sum(1)-mh[:,gb].sum(1))/5000;dp=(ph[:,po].sum(1)-mh[:,po].sum(1))/5000
                    total=(ph[:,1:].sum(1)-mh[:,1:].sum(1))/5000
                    assert np.allclose(dg+dp,total,atol=2e-14,rtol=0)
                    asym.append(dict(group=group,calibration=cal,alpha=alpha,method=method,GP_minus_GM=stat(total),G_component=stat(dg),P_only_component=stat(dp)))
        source.append(dict(group=group,normal_mean=a['mean'][:,gi,0].mean(0),normal_sd_mean=a['sd'][:,gi,0].mean(0),
            mean_cov=a['cov'][:,gi,0].mean(0),quantile_mean=a['quantile'][:,gi].mean(0),
            cut_finite_mean=[[[stat(a['cut'][:,gi,ci,ai,mi,b][np.isfinite(a['cut'][:,gi,ci,ai,mi,b])]) if np.isfinite(a['cut'][:,gi,ci,ai,mi,b]).sum()>1 else None for b in range(3)] for mi in range(2)] for ci in range(2) for ai in range(2)],
            inactive=a['inactive'][:,gi].sum(0),cal_top_mean=a['cal_top'][:,gi].mean(0),eval_top_fraction=a['top'][:,gi].mean(0)/np.array([20000,5000,5000,5000])[:,None]))
    dump('models.json',models);dump('comparison.json',rows);dump('gain_asymmetry.json',asym);dump('other_budgets.json',extra);dump('source_summary.json',source)
    np.savez_compressed(OUT/'example_rep00.npz',**examples)
    for name,digest in protected.items():assert h((ROOT/name).read_bytes())==digest,name
    dump('validation.json',dict(status='PASS_ANALYSIS',models=100,raw_evaluation_records=raw_e,raw_calibration_records=raw_cal,
        raw_records=arc.records,raw_chunks=arc.chunks,record_method_alpha_checks=checked_bits,
        scalar_halfplane_checks=checked_bits,rank_equivalence_checks=checked_bits//2,
        max_scaled_q_reconstruction_error=max_reconstruction,max_scalar_projection_difference=max_equivalence,
        comparison_rows=len(rows),asymmetry_rows=len(asym),extra_method_rows=len(extra),
        missing=0,excluded=0,invalid=0,archives=arc.archives,checked_members=arc.checked,
        new_scientific_records=0,new_detectors=0,manuscript_edits=0,elapsed_seconds=time.monotonic()-t))
    print('PASS_ANALYSIS',len(rows),'comparisons;',raw_e,'E records;',checked_bits,'method-alpha records',flush=True)

if __name__=='__main__':main()
