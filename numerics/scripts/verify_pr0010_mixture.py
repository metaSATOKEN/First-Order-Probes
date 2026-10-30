# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Second-path PR-0010 audit: sorted cuts, scalar projections, add.at joints.

Shares authenticated ZIP reader and frozen design metadata, not assembly,
calibration, projection, or histogram arithmetic with the primary analysis.
"""
from pathlib import Path
import json,hashlib,time
import numpy as np
from analyze_pr0009_mechanisms import Archive

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'numerics/analysis/pr0010_9adfa7a8c6e0_mixture_v1'
def load(p):return json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def joints(left,right):
    left,right=np.broadcast_arrays(left,right);k,n=left.shape
    z=np.zeros((k,8,8),np.int64)
    np.add.at(z,(np.repeat(np.arange(k),n),left.ravel(),right.ravel()),1)
    return z
def main():
    start=time.monotonic();assert load(OUT/'validation.json')['status']=='PASS_SAVED_DATA_RECONSTRUCTION'
    recipe=load(OUT/'recipes.json');cs=recipe['unique'];groups=recipe['groups'];edges=recipe['edges']
    arc=Archive();templates={'GM':[0],'GP':[1],'P':[2],'GMGP31':[0,0,0,1],'GMGP11':[0,1],'GMGP13':[0,1,1,1],
      'GMP11':[0,2],'GPP11':[1,2],'GMGPP112':[0,1,2,2]}
    pidx=np.array([e['parent'] for e in edges]);cidx=np.array([e['child'] for e in edges])
    faults=['GM05','GP05','P02'];evals=['normal']+faults
    checked_cal=checked_batch=checked_e=checked_tables=0;raw_count=0;q_identity_max=0.
    for rep in range(50):
        f=f'shards/{rep//5:02d}/rep_{rep:02d}'
        with np.load(OUT/'recomputed'/f'rep_{rep:02d}.npz') as zz:d={k:zz[k] for k in zz.files}
        receipt=load(OUT/'recomputed'/f'rep_{rep:02d}.json');assert digest(OUT/'recomputed'/f'rep_{rep:02d}.npz')==receipt['output_sha256']
        for gi,g in enumerate(groups):
            base='B_M' if g.endswith('_M') else 'B_L';mod=arc.npz(f'{f}/models/{base}.npz');m=load_model=json.loads(str(mod['state']))['model']
            W=np.array(m['weights']);mu=np.array(m['mean']);ordered=mod['ordered'];old=arc.npz(f'{f}/old_cal/{base}.npz')
            def scalar(q):return np.column_stack([(q[:,0]-mu[0])*w[0]+(q[:,1]-mu[1])*w[1] for w in W])
            for ri,role in enumerate(['audit','update']):
                o=arc.npz(f'{f}/orders/{g}_{role}.npz');normal=arc.values(f,g,role+'_normal');pool=[arc.values(f,g+'__'+ff,role+'_fault')[:,[11,12]] for ff in faults]
                raw_count+=len(normal)+2400;normal=normal[o['normal_order']][:,[11,12]];pos=o['replacement_order']
                for ci,c in enumerate(cs):
                    q=normal.copy();name=c['parent'] or c['id'];counters=[0,0,0]
                    if name!='clean':
                        kind,n=name.split('_n');n=int(n);pat=templates[kind]
                        for j in range(n):
                            typ=pat[j%len(pat)];source=counters[typ];counters[typ]+=1
                            if c['keep'] is None or typ in c['keep']:q[pos[j]]=pool[typ][source]
                    s=scalar(q);features=np.column_stack([q,s])
                    assert np.allclose(features.mean(0),d['mean'][gi,ci,ri],rtol=2e-10,atol=2e-13)
                    assert np.allclose(features.std(0,ddof=1),d['sd'][gi,ci,ri],rtol=2e-10,atol=2e-13)
                    checked_batch+=1
                    if ri==1:
                        z=np.column_stack([s[:,0],-s[:,0],s[:,1]]);sort=np.sort(z,axis=0)
                        rank=np.maximum(np.searchsorted(ordered[:,0],abs(s[:,0]),side='left'),np.searchsorted(ordered[:,1],s[:,1],side='left'));sr=np.sort(rank)
                        for ai,den in enumerate([100,1000]):
                            idx=[16000-16000//(den*4)-1]*2+[16000-16000//(den*2)-1]
                            assert np.allclose(sort[idx,np.arange(3)],d['cut'][gi,ci,ai,0],rtol=2e-10,atol=2e-13)
                            assert sr[16000-16000//den-1]==d['rank_cut'][gi,ci,ai];checked_cal+=1
            for ei,ev in enumerate(evals):
                case=g if ei==0 else g+'__'+ev;role='null_eval' if ei==0 else 'fault_eval'
                raw=arc.values(f,case,role);raw_count+=len(raw);checked_e+=len(raw)
                from pr0009_spec import nominal_nu
                nu=nominal_nu();derived=np.column_stack([raw[:,8]-(1-nu)*raw[:,6]-nu*raw[:,7],raw[:,14]+raw[:,15]-2*raw[:,16]])
                err=float(np.max(abs(derived-raw[:,[11,12]])));q_identity_max=max(q_identity_max,err)
                assert np.allclose(derived,raw[:,[11,12]],rtol=2e-10,atol=2e-13)
                jf=np.zeros_like(d['joint_F'][gi,:,ei]);jc=np.zeros_like(jf)
                jp=np.zeros_like(d['joint_parent'][gi,:,ei]);jm=np.zeros_like(d['joint_methods'][gi,:,ei])
                for lo in range(0,len(raw),4096):
                    s=scalar(raw[lo:lo+4096][:,[11,12]]);sg,sp=s.T
                    rg=np.searchsorted(ordered[:,0],abs(sg),side='left');rp=np.searchsorted(ordered[:,1],sp,side='left')
                    for ai in range(2):
                        cut=d['cut'][gi,:,ai,0];pk=np.zeros((57,len(sg),2),np.int64)
                        pk[:,:,0]=(sg[None,:]>cut[:,0,None])+2*(-sg[None,:]>cut[:,1,None])+4*(sp[None,:]>cut[:,2,None])
                        cr=d['rank_cut'][gi,:,ai,None];bg=rg[None,:]>cr;bp=rp[None,:]>cr
                        pk[:,:,1]=(bg&(sg[None,:]>=0))+2*(bg&(sg[None,:]<0))+4*bp
                        of=old['cut'][ai,0];fp=(sg>of[0])+2*(-sg>of[1])+4*(sp>of[2])
                        rankcut=old['cut'][ai,5,0];bg0=rg>rankcut;bp0=rp>rankcut
                        fl=(bg0&(sg>=0))+2*(bg0&(sg<0))+4*bp0
                        for mi,pold in enumerate([fp,fl]):
                            pp=pk[:,:,mi];jf[:,ai,mi]+=joints(pold[None,:],pp);jc[:,ai,mi]+=joints(pp[0:1],pp)
                            jp[:,ai,mi]+=joints(pp[pidx],pp[cidx])
                        jm[:,ai]+=joints(pk[:,:,0],pk[:,:,1])
                for name,v in [('joint_F',jf),('joint_C',jc),('joint_parent',jp),('joint_methods',jm)]:
                    assert np.array_equal(v,d[name][gi,:,ei]),(rep,g,ev,name);checked_tables+=int(np.prod(v.shape[:-2]))
        print(f'INDEPENDENT_ARITHMETIC {rep+1}/50',flush=True)
    # Manifest consistency, no detector or manuscript mutation.
    snap=load(OUT/'start_snapshot.json')
    for p,h in snap['protected'].items():assert digest(ROOT/p)==h,p
    result=dict(status='PASS_SECOND_PATH',batches_rebuilt=checked_batch,calibration_alpha_pairs=checked_cal,
      E_records_read=checked_e,raw_records_read=raw_count,joint_tables_exact=checked_tables,
      checked_members=arc.checked,archives=arc.archives,independent_reviewer=False,
      method='scalar projection; explicit record-wise composition; sorted cuts; rank LF0; np.add.at joints',
      missing=0,excluded=0,scientific_generated=0,q_identity_measured_max=q_identity_max,
      metadata_note='Primary checks.q_identity_max was an unused zero-initialized counter, not a measured maximum; use this measured value. Primary identity assertions were executed.',
      elapsed_seconds=time.monotonic()-start)
    (OUT/'second_path_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('PASS_SECOND_PATH',checked_batch,checked_e,checked_tables,flush=True)

if __name__=='__main__':main()
