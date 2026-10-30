# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Post-hoc PR-0009 mechanism audit. Saved data only; no simulation or tuning."""
from pathlib import Path
import hashlib, io, json, zipfile, time
import numpy as np
import pr0009_spec as spec

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'numerics/analysis/pr0009_9adfa7a8c6e0_two_axis_v1'
OUT=ROOT/'numerics/analysis/pr0009_9adfa7a8c6e0_mechanisms_v1'
GROUPS=spec.groups(); CL=['clean','GM160','GM800','GP160','GP800','P160','P800']
EVAL=['normal','GM05','GP05','P02']; METHODS=list(spec.METHODS)

def sha(b):return hashlib.sha256(b).hexdigest()
def dump(name,x):
    (OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def load(p):return json.loads(p.read_text())
def stat(x):
    a=np.asarray(x,float)
    return dict(n=int(a.size),mean=float(a.mean()) if a.size else None,
                sd=float(a.std(ddof=1)) if a.size>1 else None,
                median=float(np.median(a)) if a.size else None,
                min=float(a.min()) if a.size else None,max=float(a.max()) if a.size else None)

class Archive:
    def __init__(self):
        self.z={};self.index={};self.checked={};self.records=0;self.chunks=0
        self.archives=[]
        for item in load(OLD/'receipt.json')['archives']:
            p=ROOT/'Pr0009'/item['file']
            with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
            assert h==item['sha256'] and p.stat().st_size==item['bytes']
            z=zipfile.ZipFile(p);idx=json.loads(z.read('ARCHIVE_INDEX.json'))
            self.z[p.name]=z
            for n,e in idx['entries'].items():
                if n in self.index and n.startswith(('shards/','carry/','sources/')):
                    assert self.index[n][1]==e
                self.index[n]=(p.name,e)
            self.archives.append(dict(file=p.name,sha256=h,bytes=p.stat().st_size))
    def read(self,n):
        f,e=self.index[n];b=self.z[f].read(n)
        assert len(b)==e['size'] and sha(b)==e['sha256'],n
        self.checked[n]=e['sha256'];return b
    def js(self,n):return json.loads(self.read(n))
    def npz(self,n):
        with np.load(io.BytesIO(self.read(n)),allow_pickle=False) as z:return {k:z[k] for k in z.files}
    def values(self,folder,case,role):
        parts=[]
        for start in range(0,spec.COUNTS[role],spec.CHUNK):
            a=self.npz(f'{folder}/chunks/{case}/{role}/{start:09d}.npz')
            v=a['values'];md=json.loads(str(a['metadata']))
            assert v.dtype==np.float64 and v.shape[1]==17 and np.isfinite(v).all()
            assert md['start']==start and md['task']['case']==case and md['task']['role']==role
            assert md['values_sha256']==sha(v.tobytes()) and md['count']==len(v)
            parts.append(v);self.records+=len(v);self.chunks+=1
        v=np.concatenate(parts);assert len(v)==spec.COUNTS[role];return v

def main():
    start=time.monotonic();(OUT/'recomputed').mkdir(exist_ok=True)
    baseline=load(OLD/'closeout_validation.json')
    protected={**baseline['outputs'],**baseline['sources']}
    for n in ['docs/THROUGHLINE.md','AGENTS.md','audits/fp_corpus.md',
              'audits/prereg/PR-0009_two_axis_independent_confirmation.md',
              'Covariance_Consistency_Tests_for_First-Order_Probes_with_Shared_Measurement_Noise_v1.md']:
        protected[n]=sha((ROOT/n).read_bytes())
    for n,h in protected.items():assert sha((ROOT/n).read_bytes())==h,n
    ledger=(ROOT/'audits/ledger_2026-10.md').read_bytes()
    dump('start_snapshot.json',dict(protected=protected,ledger_bytes=len(ledger),ledger_sha256=sha(ledger),
                                  protocol_sha256=sha((OUT/'PROTOCOL.md').read_bytes())))
    arc=Archive();allrows=[];events=[];focus=[];maxerr=0.
    for r in range(50):
        folder=f'shards/{r//5:02d}/rep_{r:02d}'
        details={(v['group'],v['candidate']):v for v in arc.js(folder+'/gate_details.json') if v['role']=='audit'}
        with np.load(OLD/'recomputed'/f'rep_{r:02d}.npz') as z:old={k:z[k] for k in z.files}
        row=dict(mean=np.zeros((6,7,2)),sd=np.zeros((6,7,2)),T=np.zeros((6,7,2)),
                 scores=np.zeros((6,7,3)),states=np.zeros((6,7,3),int),
                 mean_effect=np.zeros((6,7,2)),sd_effect=np.zeros((6,7,2)),
                 variance_parts=np.zeros((6,7,3,2)),mean_replacement=np.zeros((6,7,2)),
                 cuts=np.zeros((6,8,2,6,3)),
                 joint=np.zeros((6,7,4,2,6,8,8),np.int64))
        for g,group in enumerate(GROUPS):
            win=0 if group.endswith('_M') else 1;cut=spec.GATE_CUTS[win]
            normal=arc.values(folder,group,'audit_normal')
            orders=arc.npz(f'{folder}/orders/{group}_audit.npz');order=orders['normal_order'];pos=orders['replacement_order']
            assert np.array_equal(np.sort(order),np.arange(15999)) and np.array_equal(np.sort(pos),np.arange(15999))
            q0=normal[order][:,[11,12]];m=len(q0);mu0=q0.mean(0);sd0=q0.std(0,ddof=1)
            faults={f:arc.values(folder,group+'__'+f,'audit_fault')[:,[11,12]] for f in spec.FAULTS}
            model=arc.npz(f'{folder}/models/{spec.baseline(group)}.npz')
            state=json.loads(str(model['state']));assert state['valid'];mod=state['model']
            oldcal=arc.npz(f'{folder}/old_cal/{spec.baseline(group)}.npz');row['cuts'][g,0]=oldcal['cut']
            for ci,(fault,n) in enumerate(spec.CANDIDATES):
                q=q0.copy()
                if n:q[pos[:n]]=faults[fault][:n]
                mu=q.mean(0);sd=q.std(0,ddof=1);T=np.sqrt(m)*mu/sd
                d=details[group,ci];saved=arc.npz(f'{folder}/audit/{group}/candidate_{ci}.npz')
                assert np.array_equal(saved['origin'].astype(bool),np.isin(np.arange(m),pos[:n]))
                for k,a in [('mean',mu),('sd',sd),('signed_T',T)]:
                    assert np.allclose(a,d['details'][k],rtol=2e-11,atol=2e-13),(r,group,ci,k)
                # G-old is reconstructed on the audit batch using the scalar LF coordinates.
                s=(q-np.array(mod['mean']))@np.array(mod['weights']).T
                z=np.column_stack((s[:,0],-s[:,0],s[:,1]))
                old_alarm=np.any(z>oldcal['cut'][0,0],axis=1).mean()
                scores=np.array([old_alarm,abs(T[0]),max(abs(T[0]),T[1])]);st=(scores>cut).astype(int)
                assert np.allclose(scores,saved['scores'],rtol=2e-11,atol=2e-13)
                assert np.array_equal(st,saved['states']) and np.array_equal(st,old['gate_states'][g,ci])
                row['mean'][g,ci]=mu;row['sd'][g,ci]=sd;row['T'][g,ci]=T
                row['scores'][g,ci]=scores;row['states'][g,ci]=st
                # Symmetric (two-order average) decomposition of signed T changes.
                nm=np.sqrt(m)/2*(mu-mu0)*(1/sd0+1/sd)
                ds=np.sqrt(m)/2*(mu+mu0)*(1/sd-1/sd0)
                err=np.max(np.abs(T-np.sqrt(m)*mu0/sd0-nm-ds));maxerr=max(maxerr,float(err));assert err<1e-10
                row['mean_effect'][g,ci]=nm;row['sd_effect'][g,ci]=ds
                parts=np.zeros((3,2))
                if n:
                    retained=np.delete(q0,pos[:n],axis=0);added=faults[fault][:n]
                    parts[0]=((retained-retained.mean(0))**2).sum(0)/(m-1)
                    parts[1]=((added-added.mean(0))**2).sum(0)/(m-1)
                    parts[2]=n*(m-n)/m*(added.mean(0)-retained.mean(0))**2/(m-1)
                    dm=(added-q0[pos[:n]]).sum(0)/m
                    assert np.allclose(dm,mu-mu0,rtol=2e-10,atol=1e-14)
                    row['mean_replacement'][g,ci]=dm
                else:parts[0]=sd**2
                assert np.allclose(parts.sum(0),sd**2,rtol=2e-11,atol=1e-16)
                row['variance_parts'][g,ci]=parts
                row['cuts'][g,ci+1]=arc.npz(f'{folder}/calibration/{group}/candidate_{ci}.npz')['cut']
                fixed=np.sqrt(m)*mu/sd0
                cls=('both-hold' if st[1] and st[2] else 'M-only' if st[1] else 'MC-only' if st[2] else 'both-pass')
                if cls=='MC-only':assert T[1]>cut[2] and abs(T[0])<=cut[1]
                if cls=='M-only':assert cut[1]<abs(T[0])<=cut[2] and T[1]<=cut[2] and win==1
                events.append(dict(replicate=r,group=group,candidate=ci,name=CL[ci],mixed=n>0,
                    T=T.tolist(),mean=mu.tolist(),sd=sd.tolist(),sd_ratio=(sd/sd0).tolist(),
                    scores=scores.tolist(),states=st.tolist(),M_MC_class=cls,
                    MC_margin=float(scores[2]-cut[2]),fixed_clean_sd_T=fixed.tolist(),
                    fixed_sd_crosses=bool(max(abs(fixed[0]),fixed[1])>cut[2]),
                    mean_effect=nm.tolist(),sd_effect=ds.tolist(),
                    between_variance_fraction=(parts[2]/sd**2).tolist()))
            for e,ev in enumerate(EVAL):
                case=group if e==0 else group+'__'+ev;n=20000 if e==0 else 5000
                chunks=[]
                for lo in range(0,n,spec.CHUNK):
                    d=arc.npz(f'{folder}/decisions/{case}/{lo:09d}.npz');b=d['packed'];md=json.loads(str(d['metadata']))
                    assert md['start']==lo and md['task']['case']==case and md['packed_sha256']==sha(b.tobytes())
                    chunks.append(b)
                bits=np.concatenate(chunks);assert bits.shape==(n,8,2)
                pat=((bits[:,:,:,None]>>(4*np.arange(6,dtype=np.uint32)))&15).astype(np.uint8)
                assert np.all(pat&8);pat &=7
                hist=np.zeros((8,2,6,8),np.int64)
                for j in range(8):
                    for ai in range(2):
                        for mi in range(6):hist[j,ai,mi]=np.bincount(pat[:,j,ai,mi],minlength=8)
                assert np.array_equal(hist,old['counts'][g,e])
                for ci in range(7):
                    for ai in range(2):
                        for mi in range(6):
                            jt=np.bincount((pat[:,0,ai,mi]*8+pat[:,ci+1,ai,mi]).astype(int),minlength=64).reshape(8,8)
                            row['joint'][g,ci,e,ai,mi]=jt
                            assert np.array_equal(jt.sum(1),hist[0,ai,mi]) and np.array_equal(jt.sum(0),hist[ci+1,ai,mi])
                            for gi in range(3):
                                if row['states'][g,ci,gi]:
                                    pair=np.array([jt[1:,1:].sum(),jt[1:,0].sum(),jt[0,1:].sum(),jt[0,0]])
                                else:
                                    alarms=hist[ci+1,ai,mi,1:].sum();pair=np.array([alarms,0,0,n-alarms])
                                assert np.array_equal(pair,old['paired'][g,ci,e,gi,0,ai,mi])
                # Detailed score/threshold audit for the preselected source group with cross loss.
                if group=='V05_L':
                    values=arc.values(folder,case,'null_eval' if e==0 else 'fault_eval')
                    q=values[:,[11,12]]-np.array(mod['mean'])
                    s=np.column_stack([q[:,0]*w[0]+q[:,1]*w[1] for w in mod['weights']])
                    z=np.column_stack((s[:,0],-s[:,0],s[:,1]))
                    for j in range(8):
                        for ai in range(2):
                            b=z>row['cuts'][g,j,ai,0];p=(b*np.array([1,2,4])).sum(1)
                            assert np.array_equal(p,pat[:,j,ai,0])
                    # Primary cross-loss: F versus negative-gain-contaminated U800 (base index3).
                    f=pat[:,0,0,0];u=pat[:,3,0,0]
                    for label,mask in [('all',np.ones(n,bool)),('F-only',(f>0)&(u==0)),('U-only',(f==0)&(u>0))]:
                        focus.append(dict(replicate=r,evaluation=ev,subset=label,n=int(mask.sum()),
                            score_medians=[float(np.median(s[mask,k])) if mask.any() else None for k in range(2)],
                            margin_F_medians=[float(np.median(z[mask,k]-row['cuts'][g,0,0,0,k])) if mask.any() else None for k in range(3)],
                            margin_U_medians=[float(np.median(z[mask,k]-row['cuts'][g,3,0,0,k])) if mask.any() else None for k in range(3)]))
        np.savez_compressed(OUT/'recomputed'/f'rep_{r:02d}.npz',**row)
        allrows.append(row);print('REPRODUCED',r,round(time.monotonic()-start,1),flush=True)
    a={k:np.stack([v[k] for v in allrows]) for k in allrows[0]}
    mixed=[v for v in events if v['mixed']];clean=[v for v in events if not v['mixed']]
    totals=dict(mixed=len(mixed),clean=len(clean),MC_slips=sum(not v['states'][2] for v in mixed),
                MC_only=sum(v['M_MC_class']=='MC-only' for v in mixed),M_only=sum(v['M_MC_class']=='M-only' for v in mixed),
                clean_MC_holds=sum(v['states'][2] for v in clean))
    assert totals==dict(mixed=1800,clean=300,MC_slips=285,MC_only=386,M_only=46,clean_MC_holds=2)
    gs=[];cross=[];decomp=[]
    for g,group in enumerate(GROUPS):
        for ci in range(7):
            subset=[v for v in events if v['group']==group and v['candidate']==ci]
            slips=[v for v in subset if not v['states'][2]]
            gs.append(dict(group=group,candidate=ci,name=CL[ci],gate_holds=a['states'][:,g,ci].sum(0).tolist(),
                MC_slips=len(slips),MC_only=sum(v['M_MC_class']=='MC-only' for v in subset),
                M_only=sum(v['M_MC_class']=='M-only' for v in subset),
                slip_fixed_sd_crosses=sum(v['fixed_sd_crosses'] for v in slips),
                slip_margin=stat([v['MC_margin'] for v in slips]),
                T_mean=a['T'][:,g,ci].mean(0).tolist(),T_sd=a['T'][:,g,ci].std(0,ddof=1).tolist(),
                sd_ratio_mean=(a['sd'][:,g,ci]/a['sd'][:,g,0]).mean(0).tolist(),
                mean_effect=a['mean_effect'][:,g,ci].mean(0).tolist(),sd_effect=a['sd_effect'][:,g,ci].mean(0).tolist(),
                between_variance_fraction=(a['variance_parts'][:,g,ci,2]/a['sd'][:,g,ci]**2).mean(0).tolist()))
            for l,r in [(0,1),(0,2),(1,2)]:
                s=a['states'][:,g,ci]
                counts=[int(((s[:,l]==x)&(s[:,r]==y)).sum()) for x,y in [(0,0),(0,1),(1,0),(1,1)]]
                cross.append(dict(group=group,candidate=ci,left=spec.GATES[l],right=spec.GATES[r],
                                  cells_passpass_passhold_holdpass_holdhold=counts))
            for e,ev in enumerate(EVAL):
                n=20000 if e==0 else 5000
                for ai,den in enumerate(spec.ALPHA_DENOMINATORS):
                    for mi,method in enumerate(METHODS):
                        jt=a['joint'][:,g,ci,e,ai,mi]
                        fg=jt[:,1:,0].sum(1)/n;fl=jt[:,0,1:].sum(1)/n
                        F=jt[:,1:,:].sum((1,2))/n;U=jt[:,:,1:].sum((1,2))/n
                        cleanjt=a['joint'][:,g,0,e,ai,mi];C=cleanjt[:,:,1:].sum((1,2))/n
                        for gi,gate in enumerate(spec.GATES):
                            h=a['states'][:,g,ci,gi].astype(bool);p=h.mean();diff=F-U
                            delta=h*diff;selected=np.where(h,F,U)
                            assert np.allclose(delta,selected-U,atol=2e-16)
                            assert np.allclose((selected-C),delta+(U-C),atol=2e-16)
                            assert abs(delta.mean()-(p*diff[h].mean() if h.any() else 0))<2e-14
                            gained=h*fg;lost=h*fl
                            assert np.allclose(gained-lost,delta,atol=2e-16)
                            # Joint branch patterns are exclusive: no double-counting overlapping tails.
                            gainpat=(jt[:,:,0]*h[:,None]).sum(0);losspat=(jt[:,0,:]*h[:,None]).sum(0)
                            gainpat[0]=0;losspat[0]=0
                            decomp.append(dict(group=group,candidate=ci,name=CL[ci],evaluation=ev,alpha=1/den,method=method,gate=gate,
                                hold_n=int(h.sum()),n_per_rep=n,delta=stat(delta),delta_per_rep=delta.tolist(),
                                F=float(F.mean()),U=float(U.mean()),C_clean=float(C.mean()),G=float(selected.mean()),
                                conditional_F_minus_U=stat(diff[h]),nonhold_F_minus_U=stat(diff[~h]),
                                unweighted_F_minus_U=float(diff.mean()),selection_covariance=float(delta.mean()-p*diff.mean()),
                                G_minus_C=float((selected-C).mean()),U_minus_C=float((U-C).mean()),
                                gained=float(gained.mean()),lost=float(lost.mean()),
                                gain_patterns=gainpat.tolist(),loss_patterns=losspat.tolist(),
                                cut_F_minus_U_mean=(a['cuts'][:,g,0,ai,mi]-a['cuts'][:,g,ci+1,ai,mi]).mean(0).tolist()))
    dump('gate_events.json',events);dump('gate_summary.json',gs);dump('gate_cross_tables.json',cross)
    dump('policy_decomposition.json',decomp);dump('focus_score_margins.json',focus)
    dump('summary.json',totals)
    for n,h in protected.items():assert sha((ROOT/n).read_bytes())==h,n
    dump('validation.json',dict(status='PASS_MECHANISM_RECONSTRUCTION',archives=arc.archives,
        verified_read_members=len(arc.checked),read_member_hashes=arc.checked,
        audit_batches_reconstructed=2100,E_records_branch_recounted=50*6*35000,
        V05_E_records_W0_two_alpha_recomputed=50*35000,raw_records_read=arc.records,raw_chunks_read=arc.chunks,
        mean_sd_decomposition_max_abs_error=maxerr,policy_rows=len(decomp),gate_events=len(events),
        all_gate_states_and_branch_margins_match=True,all_old_policy_paired_counts_match=True,
        source_protected_unchanged=len(protected),missing=0,unavailable=0,excluded=0,
        generated_time_series=0,detector_changes=0,independent_reviewer=False,
        elapsed_seconds=time.monotonic()-start))
    print('PASS',totals,'policy_rows',len(decomp),flush=True)

if __name__=='__main__':main()
