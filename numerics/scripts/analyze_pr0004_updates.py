# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Decompose existing C/R decisions and boundaries, without hybrid detectors."""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import sys
import time
import zipfile
import numpy as np
from analyze_pr0004_lowfpr import ROOT, file_hash, load_arrays, canonical, ci, put, order, check_state, ranks_from_cal
import pr0004_spec as spec

ALPHAS=[.001,.01]
METHODS=['M','CR','B','J','LF']
ROC=ROOT/'numerics/analysis/pr0004_e0af6fcba596_lowfpr_roc_v1'
TERMS=['direction','center','reference_table','calibration_rank']


def cutoffs(r,alpha):
    def cut(x,a):
        k=order(a)
        return 9999 if k>len(x) else int(np.sort(x)[k-1])
    j=r['J'];c=np.full((5,2),9999)
    c[0,0]=cut(j[:,0],alpha);c[1,1]=cut(j[:,1],alpha)
    c[2]=[cut(j[:,i],alpha/2) for i in range(2)]
    c[3]=cut(j.max(1),alpha);c[4]=cut(r['LF'].max(1),alpha)
    return c


def geometry(m,o,cuts,mu,sd):
    w=np.array(m['weights']);l=np.linalg.norm(w*sd,axis=1)
    assert np.all(l>0) and m['disabled']==[]
    u=w*sd/l[:,None];b=(w@(mu-np.array(m['mean'])))/l
    tt=np.array([np.inf if cuts[4,j]>=9999 else o['LF'][cuts[4,j],j] for j in range(2)])
    h=tt/l
    return dict(u=u,b=b,h=h,L=l,t=tt,U=np.stack((u[0],-u[0],u[1])),
        B=np.array([b[0],-b[0],b[1]]),H=np.array([h[0],h[0],h[1]]))


def split_threshold(gc,gr,oc,orr,cc,cr):
    table=[];rank=[]
    for j in range(2):
        c,r=int(cc[4,j]),int(cr[4,j]);assert max(c,r)<9999
        tc=oc['LF'][:,j]/gc['L'][j];tr=orr['LF'][:,j]/gr['L'][j]
        table.append(.5*((tr[c]-tc[c])+(tr[r]-tc[r])))
        rank.append(.5*((tc[r]-tc[c])+(tr[r]-tr[c])))
    table=np.array(table);rank=np.array(rank)
    assert np.allclose(table+rank,gr['h']-gc['h'],rtol=1e-13,atol=1e-13)
    return table,rank


def pack(flags):return (flags.astype(np.uint8)*(1<<np.arange(5,dtype=np.uint8))).sum(1).astype(np.uint8)


def run(archive,out):
    archive,out=Path(archive),Path(out);assert (out/'PROTOCOL.md').exists() and not (out/'validation.json').exists()
    start=time.monotonic();state=check_state()
    roc_delivery=json.loads((ROC/'delivery_validation.json').read_text())
    for p,h in roc_delivery['files'].items():assert file_hash(ROC/p)==h
    state['previous']|={str((ROC/p).relative_to(ROOT)):h for p,h in roc_delivery['files'].items()}
    val=dict(status='IN_PROGRESS',new_scientific_records=0,hybrid_detectors=0,post_hoc=True,
        protocol_sha256=file_hash(out/'PROTOCOL.md'),preserved=state,
        ledger_before_bytes=(ROOT/'audits/ledger_2026-10.md').stat().st_size,
        ledger_before_sha256=file_hash(ROOT/'audits/ledger_2026-10.md'),corpus_sha256=file_hash(ROOT/'audits/fp_corpus.md'))
    put(out/'validation.json',val)
    try:
        val['archive_sha256']=file_hash(archive)
        assert val['archive_sha256']=='7e07909140a0d9080ccdf60837fe1c84ec15d52004c227d66e6f7472feb7a589'
        conditions={k:c for k,c in spec.conditions().items() if not c['group'].startswith('B_')}
        groups=[k for k,c in conditions.items() if c['kind']=='normal'];rows=[];trans=[];margins=[];geos=[];example={}
        checks=dict(calibration_records=0,evaluation_records=0,chunks=0,decision_sets=0,
                    saved_main_sets=0,boundary_records=0,finite_margin_records=0,max_margin_identity_error=0.)
        (out/'decisions').mkdir()
        with zipfile.ZipFile(archive) as z:
            index=json.loads(z.read('ARCHIVE_INDEX.json'))['entries']
            def read(p):
                b=z.read(p);assert hashlib.sha256(b).hexdigest()==index[p]['sha256'],p
                return b
            def js(p):return json.loads(read(p))
            def arr(p):return load_arrays(read(p))
            assert canonical(js('specification.json'))==canonical(spec.specification())
            sh=hashlib.sha256(canonical(spec.specification()).encode()).hexdigest()
            tasks=spec.tasks()
            for rep in range(50):
                done=js(f'analysis/repeats/{rep:02d}/done.json');pre=f'analysis/repeats/{rep:02d}/'+done['attempt']+'/'
                taskmap={(t['case'],t['role']):t for t in tasks if t['replicate']==rep}
                def values(case,role):
                    task=taskmap[case,role];parts=[]
                    for s in range(0,task['count'],4096):
                        a=arr('chunks/'+task['id']+f'/{s:09d}.npz');v=a['values'];meta=json.loads(str(a['metadata']))
                        assert meta['task']==task and meta['start']==s and meta['specification_sha256']==sh
                        assert v.shape==(min(4096,task['count']-s),17) and v.dtype==np.float64 and np.isfinite(v).all()
                        assert hashlib.sha256(v.tobytes()).hexdigest()==meta['values_sha256']
                        checks['chunks']+=1;parts.append(v)
                    checks['calibration_records' if role=='cal' else 'evaluation_records']+=task['count']
                    return np.concatenate(parts)
                models={};ordered={};calibration={};geom={};delta={};saved_out={}
                for group in ['B_M','B_L']+groups:
                    models[group]=js(pre+f'models/{group}/model.json');ordered[group]=arr(pre+f'models/{group}/reference_ranks.npz')
                for group in groups:
                    mR=models[group];mu=np.array(mR['mean']);sd=np.sqrt(np.diag(mR['covariance']))
                    cv=values(group,'cal');calibration[group]={};geom[group]={}
                    for policy in ['C','R']:
                        mg=spec.baseline(group) if policy=='C' else group;m=models[mg];o=ordered[mg]
                        r=ranks_from_cal(m,o,cv);cuts=np.stack([cutoffs(r,a) for a in ALPHAS])
                        assert np.array_equal(cuts[1],np.array(js(pre+f'models/{group}/{policy}_calibration.json')['cutoffs'])[:5])
                        calibration[group][policy]=cuts
                        geom[group][policy]=[geometry(m,o,c,mu,sd) for c in cuts]
                    gc,gr=geom[group]['C'][1],geom[group]['R'][1]
                    assert np.isfinite(gc['h']).all() and np.isfinite(gr['h']).all()
                    dt,dk=split_threshold(gc,gr,ordered[spec.baseline(group)],ordered[group],calibration[group]['C'][1],calibration[group]['R'][1])
                    delta[group]=(dt,dk)
                    for ai,alpha in enumerate(ALPHAS):
                        cg,rg=geom[group]['C'][ai],geom[group]['R'][ai]
                        angle=np.arctan2(rg['u'][:,1],rg['u'][:,0])-np.arctan2(cg['u'][:,1],cg['u'][:,0])
                        angle=np.degrees(np.arctan2(np.sin(angle),np.cos(angle)))
                        def serial(g):
                            return {k:[None if not np.isfinite(x) else float(x) for x in g[k]] if g[k].ndim==1 else g[k].tolist() for k in ['u','b','h','L','t']}
                        geos.append(dict(replicate=rep,group=group,alpha=alpha,C=serial(cg),R=serial(rg),
                            angle_difference=angle.tolist(),center_shift_standardized=((np.array(models[spec.baseline(group)]['mean'])-mu)/sd).tolist(),
                            cuts_C=calibration[group]['C'][ai].tolist(),cuts_R=calibration[group]['R'][ai].tolist(),
                            delta_h_table=dt.tolist() if ai==1 else None,delta_h_cut=dk.tolist() if ai==1 else None))
                for case,c in conditions.items():
                    group=c['group'];v=values(case,'null' if c['kind']=='normal' else 'fault');q=v[:,[11,12]];n=len(v)
                    mu=np.array(models[group]['mean']);sd=np.sqrt(np.diag(models[group]['covariance']));zz=(q-mu)/sd
                    bits=np.zeros((n,2,2),np.uint8);branches=np.zeros_like(bits);margin={}
                    for pi,policy in enumerate(['C','R']):
                        mg=spec.baseline(group) if policy=='C' else group;m=models[mg];o=ordered[mg]
                        ranks=ranks_from_cal(m,o,v);sv=arr(pre+f'decisions/{case}_{policy}.npz')
                        assert np.array_equal(ranks['J'],sv['J']) and np.array_equal(ranks['LF'],sv['LF'])
                        for ai,alpha in enumerate(ALPHAS):
                            cuts=calibration[group][policy][ai]
                            flags=np.column_stack([(ranks['J' if j<4 else 'LF']>cuts[j]).any(1) for j in range(5)])
                            bits[:,ai,pi]=pack(flags);br=ranks['LF']>cuts[4]
                            branches[:,ai,pi]=br[:,0].astype(np.uint8)+2*br[:,1].astype(np.uint8)
                            g=geom[group][policy][ai];mm=zz@g['U'].T+g['B']-g['H']
                            assert np.array_equal(mm.max(1)>0,flags[:,4])
                            assert np.array_equal((mm[:,0]>0)|(mm[:,1]>0),br[:,0]) and np.array_equal(mm[:,2]>0,br[:,1])
                            checks['boundary_records']+=n;checks['decision_sets']+=1
                            if ai==1:
                                assert np.array_equal(bits[:,ai,pi],sv['bits']&31);checks['saved_main_sets']+=1
                                margin[policy]=mm
                    for ai,alpha in enumerate(ALPHAS):
                        for j,method in enumerate(METHODS):
                            cf=(bits[:,ai,0]&(1<<j))!=0;rf=(bits[:,ai,1]&(1<<j))!=0
                            count=[int(x.sum()) for x in [cf&rf,~cf&rf,cf&~rf,~cf&~rf]]
                            rows.append(dict(replicate=rep,case=case,group=group,kind=c['kind'],method=method,alpha=alpha,n=n,counts=count))
                        tr=np.bincount(4*branches[:,ai,0]+branches[:,ai,1],minlength=16).reshape(4,4)
                        trans.append(dict(replicate=rep,case=case,group=group,kind=c['kind'],alpha=alpha,n=n,table=tr.tolist()))
                    gc,gr=geom[group]['C'][1],geom[group]['R'][1];dt,dk=delta[group]
                    du=gr['U']-gc['U'];db=gr['B']-gc['B'];dht=np.array([dt[0],dt[0],dt[1]]);dhk=np.array([dk[0],dk[0],dk[1]])
                    rotation=zz@du.T
                    difference=margin['R']-margin['C'];pred=rotation+db-dht-dhk
                    err=float(np.max(abs(pred-difference)));checks['max_margin_identity_error']=max(err,checks['max_margin_identity_error'])
                    assert np.allclose(pred,difference,rtol=2e-12,atol=2e-12)
                    checks['finite_margin_records']+=n
                    for typ,win,lose,sign in [('R_only','R','C',1),('C_only','C','R',-1)]:
                        mask=(margin[win].max(1)>0)&(margin[lose].max(1)<=0);ii=np.flatnonzero(mask)
                        face=margin[win][ii].argmax(1)
                        vals=sign*np.column_stack((rotation[ii,face],db[face],-dht[face],-dhk[face]))
                        exact=sign*difference[ii,face]
                        assert np.all(exact>0) and np.allclose(vals.sum(1),exact,atol=2e-12,rtol=2e-12)
                        counts=np.bincount(face,minlength=3)
                        sums=np.stack([vals[face==k].sum(0) for k in range(3)])
                        margins.append(dict(replicate=rep,case=case,group=group,kind=c['kind'],direction=typ,n=n,
                            flip_count=len(ii),face_counts=counts.tolist(),term_sums=sums.tolist(),margin_sum=float(exact.sum())))
                    saved_out[case+'_bits']=bits;saved_out[case+'_branches']=branches
                    if rep==0 and case in ['K200_L','K200_L__P02','K2_M','K2_M__P02','V05_L','V05_L__GP05','V05_L__GM05']:
                        example[case+'_z']=zz;example[case+'_bits']=bits;example[case+'_branches']=branches
                np.savez_compressed(out/f'decisions/rep_{rep:02d}.npz',**saved_out)
                print(f'Update decomposition {rep+1}/50; saved main sets {checks["saved_main_sets"]}',flush=True)
        assert checks['calibration_records']==4000000 and checks['evaluation_records']==14000000
        assert checks['decision_sets']==6400 and checks['saved_main_sets']==3200 and checks['finite_margin_records']==14000000
        assert checks['boundary_records']==56000000
        old=json.loads((ROC/'summary.json').read_text())
        ref_rates={(r['case'],r['policy'],r['method'],r['beta']):r['nominal_tpr']['mean'] for r in old['rates']}
        ref_rates|={(r['group'],r['policy'],r['method'],r['beta']):r['nominal_fpr']['mean'] for r in old['normal']}
        grouped=defaultdict(list)
        for r in rows:grouped[r['case'],r['method'],r['alpha']].append(r)
        summary=[]
        for (case,method,alpha),rr in grouped.items():
            ar=np.array([r['counts'] for r in rr])/rr[0]['n']
            metrics={name:ci(ar[:,j]) for j,name in enumerate(['both','R_only','C_only','neither'])}
            metrics.update(C=ci(ar[:,0]+ar[:,2]),R=ci(ar[:,0]+ar[:,1]),difference=ci(ar[:,1]-ar[:,2]))
            for p in ['C','R']:assert abs(metrics[p]['mean']-ref_rates[case,p,method,alpha])<2e-14
            summary.append(dict(case=case,group=rr[0]['group'],kind=rr[0]['kind'],method=method,alpha=alpha,metrics=metrics))
        tg=defaultdict(list)
        for r in trans:tg[r['case'],r['alpha']].append(r)
        tsum=[]
        for (case,alpha),rr in tg.items():
            a=np.array([r['table'] for r in rr])/rr[0]['n']
            met={name:ci(vals) for name,vals in [('R_g_only',a[:,0,1]),('R_p_only',a[:,0,2]),('R_both',a[:,0,3]),
                ('C_g_only',a[:,1,0]),('C_p_only',a[:,2,0]),('C_both',a[:,3,0])]}
            tsum.append(dict(case=case,group=rr[0]['group'],kind=rr[0]['kind'],alpha=alpha,
                table_counts=np.sum([r['table'] for r in rr],axis=0).tolist(),n_total=sum(r['n'] for r in rr),metrics=met))
        mg=defaultdict(list)
        for r in margins:mg[r['case'],r['direction']].append(r)
        msum=[]
        for (case,direction),rr in mg.items():
            sums=np.sum([r['term_sums'] for r in rr],axis=0);count=sum(r['flip_count'] for r in rr)
            per_record=np.array([np.sum(r['term_sums'],axis=0)/r['n'] for r in rr])
            msum.append(dict(case=case,direction=direction,kind=rr[0]['kind'],flip_count=count,
                face_counts=np.sum([r['face_counts'] for r in rr],axis=0).tolist(),term_sums=sums.tolist(),
                conditional_mean=(sums.sum(0)/count).tolist() if count else None,
                per_record=[ci(per_record[:,j]) for j in range(4)],margin_sum=sum(r['margin_sum'] for r in rr)))
        put(out/'replicate_rates.json',rows);put(out/'replicate_transitions.json',trans);put(out/'replicate_margins.json',margins)
        put(out/'geometry.json',geos);np.savez_compressed(out/'examples_rep00.npz',**example)
        put(out/'summary.json',dict(rates=summary,transitions=tsum,margins=msum,terms=TERMS,methods=METHODS,alphas=ALPHAS,
            interval='exploratory paired outer-replicate pointwise 95% t; no multiplicity correction'))
        assert file_hash(archive)==val['archive_sha256']
        for values in state.values():
            for p,h in values.items():assert file_hash(ROOT/p)==h,p
        val.update(status='PASS',checks=checks,rate_rows=len(summary),transition_rows=len(tsum),margin_rows=len(msum),
            source_sha256=file_hash(Path(__file__)),elapsed_seconds=time.monotonic()-start,old_evidence_unchanged=True)
        put(out/'validation.json',val);print(json.dumps({k:v for k,v in val.items() if k!='preserved'},indent=2),flush=True)
    except Exception as e:
        val.update(status='FAILED',error=repr(e),elapsed_seconds=time.monotonic()-start);put(out/'validation.json',val);raise


if __name__=='__main__':run(sys.argv[1],sys.argv[2])
