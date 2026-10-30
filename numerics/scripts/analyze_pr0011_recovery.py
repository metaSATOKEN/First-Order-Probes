# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""PR-0011 saved-data policy replay. No simulator, refit, or threshold tuning."""
from pathlib import Path
import hashlib, json, time, zipfile
import numpy as np
from scipy.stats import beta
from analyze_pr0009_mechanisms import Archive
import pr0009_spec as spec

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'numerics/analysis/pr0011_9adfa7a8c6e0_recovery_v2'
PREV=ROOT/'numerics/analysis/pr0010_9adfa7a8c6e0_mixture_v1'
OLD=ROOT/'numerics/analysis/pr0009_9adfa7a8c6e0_two_axis_v1'
CARD=ROOT/'audits/prereg/PR-0011_stop_recovery_saved_data.md'
DESIGN=ROOT/'numerics/design/POST_PR0010_STOP_AND_RECOVERY_OPTIONS_v1_JA.md'
G=list(spec.groups()); EV=['normal',*spec.FAULTS]; AL=[.01,.001]
CAL=['F','C0']; GATES=['G-old','G-M','G-MC']; POL=['P0','P1','P2']
LABEL=['SUPPORTED','EXCEEDED','UNRESOLVED','UNAVAILABLE']
ACTION=['U','F','C0','DEFER','INVALID']; STATUS=['ALARM','NO-ALARM','DEFER','INVALID']
PAIRS=[(1,0),(2,1),(2,0)]; ETA=.00625

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bh(b):return hashlib.sha256(b).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def cp(k,n=20000,eta=ETA,provenance=True):
    if not provenance or n<=0 or not np.isfinite(k) or not 0<=k<=n:
        return np.array([np.nan,np.nan])
    return np.array([0. if k==0 else beta.ppf(eta,k,n-k+1),1. if k==n else beta.ppf(1-eta,k+1,n-k)])
def label(bound,alpha):
    if not np.isfinite(bound).all():return 3
    return 0 if bound[1]<=1.25*alpha else 1 if bound[0]>1.25*alpha else 2
def choices(h,lf,lc,c_available=True):
    if h==2:return np.array([1,4,4],np.uint8) # preserves legacy P0 UNKNOWN->F
    if h not in (0,1) or lf not in range(4) or lc not in range(4):raise ValueError('state')
    if h==0:return np.array([0,0,0],np.uint8)
    return np.array([1,1 if lf==0 else 3,1 if lf==0 else 2 if c_available and lc==0 else 3],np.uint8)
def decide(q,model,cuts):
    s=(q-np.array(model['mean']))@np.array(model['weights']).T
    z=np.column_stack([s[:,0],-s[:,0],s[:,1]])
    return np.sum((z[:,None,None,:]>cuts[None])*[1,2,4],axis=-1,dtype=np.uint8)
def independent_decide(values,model,cuts):
    # Alternative coordinates from saved primitive moments, scalar projections.
    q0=values[:,8]-(1-spec.nominal_nu())*values[:,6]-spec.nominal_nu()*values[:,7]
    q1=values[:,14]+values[:,15]-2*values[:,16]
    mean=model['mean'];w=model['weights']
    s0=w[0][0]*(q0-mean[0])+w[0][1]*(q1-mean[1])
    s1=w[1][0]*(q0-mean[0])+w[1][1]*(q1-mean[1])
    out=np.zeros((len(values),2,2),np.uint8)
    for c in range(2):
        for a in range(2):
            out[:,c,a]=(s0>cuts[c,a,0]).astype(np.uint8)+2*(-s0>cuts[c,a,1])+4*(s1>cuts[c,a,2])
    error=float(np.max(abs(values[:,[11,12]]-np.column_stack([q0,q1]))))
    return out,error
def binary(j):
    return np.array([[j[1:,1:].sum(),j[1:,0].sum()],[j[0,1:].sum(),j[0,0]]],np.int64)
def base_hist(jf,jc):
    f=jf.sum(1);u=jf.sum(0);c=jc.sum(1)
    assert np.array_equal(u,jc.sum(0))
    return np.array([u,f,c])
def outcome(action,hist):
    n=int(hist[0].sum());v=np.zeros(4,np.int64)
    if action<3:v[:2]=[hist[action,1:].sum(),hist[action,0]]
    else:v[action-1]=n
    return v
def pair_table(left,right,jf,jc,jfc):
    hist=base_hist(jf,jc);n=int(jf.sum());out=np.zeros((4,4),np.int64)
    if left>=3:
        out[left-1]=outcome(right,hist);return out
    if right>=3:
        out[:,right-1]=outcome(left,hist);return out
    if left==right:
        out[0,0]=hist[left,1:].sum();out[1,1]=hist[left,0];return out
    pairs={(1,0):jf,(2,0):jc,(1,2):jfc}
    j=pairs[(left,right)] if (left,right) in pairs else pairs[(right,left)].T
    out[:2,:2]=binary(j)
    assert out.sum()==n
    return out
def select_joints(values,ids):
    out=np.take(values[:,:,:,:,0],ids,axis=1)
    assert out.shape==(6,len(ids),4,2,8,8)
    return out
def protected_start():
    d=load(ROOT/'audits/verification/2026-10-18_stop_recovery_design.json')
    c=load(ROOT/'audits/verification/2026-10-18_stop_recovery_design_closeout.json')
    keep={**d['protected'],**c['outputs'],**c['sources'],**c['frozen_sources']}
    for p in ['AGENTS.md','docs/THROUGHLINE.md','audits/fp_corpus.md','audits/prereg/README.md']:
        keep[p]=sha(ROOT/p)
    for p,h in keep.items():assert sha(ROOT/p)==h,p
    return dict(protected=keep,inputs={str(p.relative_to(ROOT)):sha(p) for p in [CARD,DESIGN,Path(__file__)]},
        ledger_bytes=(ROOT/'audits/ledger_2026-10.md').stat().st_size,ledger_sha256=sha(ROOT/'audits/ledger_2026-10.md'))

def run_rep(arc,r,ids):
    t0=time.monotonic();folder=f'shards/{r//5:02d}/rep_{r:02d}'
    with np.load(PREV/'recomputed'/f'rep_{r:02d}.npz',allow_pickle=False) as z:
        states=z['states'][:,ids];oldcuts=z['cut'][:,0,:,0]
    assert states.shape==(6,37,3) and np.isfinite(oldcuts).all()
    row=dict(v_counts=np.zeros((6,2,2),np.int64),v_bounds=np.zeros((6,2,2,2)),v_labels=np.zeros((6,2,2),np.uint8),
        old_labels=np.zeros((6,2),np.uint8),actions=np.zeros((6,37,3,2,3),np.uint8),states=states,
        counts=np.zeros((6,37,3,2,3,4,4),np.int64),paired=np.zeros((6,37,3,2,3,4,4,4),np.int64),
        shadows=np.zeros((6,37,3,2,3,4,3),np.int64),base=np.zeros((6,37,4,2,3,8),np.int64),
        e_labels=np.zeros((6,2,2),np.uint8),e_bounds=np.zeros((6,2,2,2)))
    checks=dict(V_records=0,record_cal_alpha_decisions=0,old_bit_matches=0,independent_bit_matches=0,
        q_identity_max=0.,role_checks=0,unknown=int(np.count_nonzero(states==2)),paired_tables=0)
    provenance=[];frozen=arc.js(folder+'/CALIBRATION_FROZEN.json')
    with np.load(OLD/'recomputed'/f'rep_{r:02d}.npz') as z:old_v=z['v_counts'][:,0,:,0,0]
    for g,group in enumerate(G):
        base=spec.baseline(group);mp=f'models/{base}.npz';fp=f'old_cal/{base}.npz';cc=f'calibration/{group}/candidate_0.npz'
        for p in [mp,fp,cc]:assert bh(arc.read(folder+'/'+p))==frozen[p]
        model=arc.npz(folder+'/'+mp);state=json.loads(str(model['state']));assert state['valid'] and not state['model']['disabled']
        fcal=arc.npz(folder+'/'+fp);ccal=arc.npz(folder+'/'+cc)
        cuts=np.array([fcal['cut'][:,0],ccal['cut'][:,0]])
        assert np.isfinite(cuts).all() and np.allclose(cuts[1],oldcuts[g],atol=2e-13,rtol=2e-10)
        for cal in [fcal,ccal]:assert json.loads(str(cal['metadata']))['valid']
        ap=arc.npz(f'{folder}/audit/{group}/candidate_0.npz');up=arc.npz(f'{folder}/update/{group}/candidate_0.npz')
        pa=json.loads(str(ap['provenance']));pu=json.loads(str(up['provenance']))
        assert pa['normal_task']==f'{r:02d}/{group}/audit_normal' and pu['normal_task']==f'{r:02d}/{group}/update_normal'
        assert pu['fault_task'] is None and pa['fault_task'] is None
        assert pu['input_sha256']==str(ccal['input_sha256']) and np.all(up['origin']==0)
        assert np.array_equal(np.sort(up['ids']),np.arange(15999))
        checks['role_checks']+=1
        vhash=[]
        for start in range(0,20000,4096):
            path=f'{folder}/chunks/{group}/validation_normal/{start:09d}.npz'
            a=arc.npz(path);v=a['values'];md=json.loads(str(a['metadata']))
            assert md['task']==dict(case=group,count=20000,id=f'{r:02d}/{group}/validation_normal',replicate=r,role='validation_normal')
            assert md['start']==start and md['count']==len(v)==min(4096,20000-start)
            assert v.shape==(len(v),17) and v.dtype==np.float64 and np.isfinite(v).all()
            assert md['values_sha256']==bh(v.tobytes());vhash.append(md['values_sha256'])
            bit=decide(v[:,[11,12]],state['model'],cuts);ind,error=independent_decide(v,state['model'],cuts)
            assert np.array_equal(bit,ind),'coordinate path bit mismatch'
            checks['q_identity_max']=max(checks['q_identity_max'],error)
            saved=arc.npz(f'{folder}/validation/{group}/{start:09d}.npz');smd=json.loads(str(saved['metadata']))
            nib=saved['packed'][:,0]&15
            assert smd['raw_sha256']==bh(v.tobytes()) and smd['packed_sha256']==bh(saved['packed'].tobytes())
            assert np.all(nib&8) and np.array_equal(bit[:,0],nib&7)
            row['v_counts'][g]+=np.count_nonzero(bit,axis=0)
            checks['V_records']+=len(v);checks['record_cal_alpha_decisions']+=len(v)*4
            checks['old_bit_matches']+=len(v)*2;checks['independent_bit_matches']+=len(v)*4
        assert np.array_equal(row['v_counts'][g,0],old_v[g])
        for c in range(2):
            for a,alpha in enumerate(AL):
                row['v_bounds'][g,c,a]=cp(int(row['v_counts'][g,c,a]))
                row['v_labels'][g,c,a]=label(row['v_bounds'][g,c,a],alpha)
        for a,alpha in enumerate(AL):
            row['old_labels'][g,a]=label(cp(int(row['v_counts'][g,0,a]),eta=.0125),alpha)
            for k in range(37):
                for gate in range(3):row['actions'][g,k,gate,a]=choices(int(states[g,k,gate]),int(row['v_labels'][g,0,a]),int(row['v_labels'][g,1,a]))
        provenance.append(dict(group=group,A=pa['normal_task'],U=pu['normal_task'],V=f'{r:02d}/{group}/validation_normal',
            E=[f'{r:02d}/{group}/null_eval']+[f'{r:02d}/{group}__{f}/fault_eval' for f in spec.FAULTS],
            C0_input_sha256=str(ccal['input_sha256']),V_chunks=vhash,model_sha256=frozen[mp],F_sha256=frozen[fp],C0_sha256=frozen[cc]))
    # Persist all V-based choices before loading E joint arrays.
    freeze=OUT/'selected'/f'rep_{r:02d}.npz'
    payload={k:row[k] for k in ['v_counts','v_bounds','v_labels','old_labels','actions','states']}
    if freeze.exists():
        with np.load(freeze) as z:
            for k,v in payload.items():assert np.array_equal(z[k],v,equal_nan=True)
    else:np.savez_compressed(freeze,**payload)
    with np.load(PREV/'recomputed'/f'rep_{r:02d}.npz') as z:
        jf=select_joints(z['joint_F'],ids);jc=select_joints(z['joint_C'],ids)
    assert not checks['unknown'],'Unexpected saved UNKNOWN: preserve evidence and review before replay'
    for g in range(6):
        for k in range(37):
            for e in range(4):
                n=20000 if e==0 else 5000
                for a in range(2):
                    f,u=jf[g,k,e,a],jc[g,k,e,a];fc=jf[g,0,e,a]
                    assert f.sum()==u.sum()==fc.sum()==n
                    hist=base_hist(f,u);row['base'][g,k,e,a]=hist
                    for gate in range(3):
                        act=row['actions'][g,k,gate,a]
                        for pol in range(3):
                            row['counts'][g,k,gate,a,pol,e]=outcome(int(act[pol]),hist)
                            if act[pol]==3:row['shadows'][g,k,gate,a,pol,e]=hist[:,1:].sum(1)
                        for pi,(left,right) in enumerate(PAIRS):
                            j=pair_table(int(act[left]),int(act[right]),f,u,fc)
                            row['paired'][g,k,gate,a,pi,e]=j;checks['paired_tables']+=1
                            assert np.array_equal(j.sum(1),row['counts'][g,k,gate,a,left,e])
                            assert np.array_equal(j.sum(0),row['counts'][g,k,gate,a,right,e])
        for c,baseindex in enumerate([1,2]):
            for a,alpha in enumerate(AL):
                row['e_bounds'][g,c,a]=cp(int(row['base'][g,0,0,a,baseindex,1:].sum()))
                row['e_labels'][g,c,a]=label(row['e_bounds'][g,c,a],alpha)
    target=OUT/'recomputed'/f'rep_{r:02d}.npz';np.savez_compressed(target,**row)
    receipt=dict(replicate=r,checks=checks,elapsed_seconds=time.monotonic()-t0,provenance=provenance,
        input_sha256=sha(PREV/'recomputed'/f'rep_{r:02d}.npz'),output_sha256=sha(target),
        selection_sha256=sha(freeze),code_sha256=sha(__file__),card_sha256=sha(CARD))
    dump(target.with_suffix('.json'),receipt)
    return receipt

def main():
    for sub in ['recomputed','selected','bundles']:(OUT/sub).mkdir(parents=True,exist_ok=True)
    snapfile=OUT/'start_snapshot.json'
    if snapfile.exists():
        snap=load(snapfile)
        for p,h in snap['inputs'].items():assert sha(ROOT/p)==h,p
    else:snap=protected_start();dump(snapfile,snap)
    if (OUT/'validation.json').exists():
        for r in range(50):
            p=OUT/'recomputed'/f'rep_{r:02d}.npz';x=load(p.with_suffix('.json'))
            assert sha(p)==x['output_sha256'] and sha(OUT/'selected'/p.name)==x['selection_sha256']
            assert sha(__file__)==x['code_sha256'] and sha(CARD)==x['card_sha256']
            assert sha(PREV/'recomputed'/p.name)==x['input_sha256']
        print('COMPLETE_VERIFIED_NOOP: 50 receipts; completed evidence unchanged');return
    recipes=load(PREV/'recipes.json');ids=[i for i,x in enumerate(recipes['unique']) if x['parent'] is None];assert len(ids)==37 and ids[0]==0
    dump(OUT/'schema.json',dict(groups=G,evaluations=EV,alphas=AL,gates=GATES,policies=POL,labels=LABEL,actions=ACTION,
        statuses=STATUS,pairs=PAIRS,candidates=[recipes['unique'][i] for i in ids],base_order=['U','F','C0'],eta=ETA))
    arc=Archive();receipts=[]
    for r in range(50):
        p=OUT/'recomputed'/f'rep_{r:02d}.json'
        if p.exists():
            x=load(p);assert sha(p.with_suffix('.npz'))==x['output_sha256'] and sha(__file__)==x['code_sha256']
            assert sha(CARD)==x['card_sha256'] and sha(PREV/'recomputed'/p.with_suffix('.npz').name)==x['input_sha256']
            assert sha(OUT/'selected'/p.with_suffix('.npz').name)==x['selection_sha256']
        else:x=run_rep(arc,r,ids)
        receipts.append(x);print('REP',r,'seconds',round(x['elapsed_seconds'],3),'PASS',flush=True)
        if sum(z['elapsed_seconds'] for z in receipts)>1800:raise TimeoutError('30 minute analysis budget; evidence retained')
        if r%5==4:
            zp=OUT/'bundles'/f'PR0011_shard_{r//5+1:02d}.zip'
            if not zp.exists():
                with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED) as z:
                    for j in range(r-4,r+1):
                        for f in [OUT/'recomputed'/f'rep_{j:02d}.npz',OUT/'recomputed'/f'rep_{j:02d}.json',OUT/'selected'/f'rep_{j:02d}.npz']:z.write(f,f.relative_to(OUT))
                assert zp.stat().st_size<128_000_000
    for p,h in snap['protected'].items():assert sha(ROOT/p)==h,p
    sums={k:sum(x['checks'][k] for x in receipts) for k in ['V_records','record_cal_alpha_decisions','old_bit_matches','independent_bit_matches','role_checks','paired_tables','unknown']}
    assert sums['V_records']==6000000 and sums['record_cal_alpha_decisions']==24000000
    dump(OUT/'validation.json',dict(status='PASS_SAVED_DATA_REPLAY',replicates=50,checks=sums,
        q_identity_max=max(x['checks']['q_identity_max'] for x in receipts),elapsed_rep_seconds=sum(x['elapsed_seconds'] for x in receipts),
        archives=arc.archives,checked_members=arc.checked,missing=0,excluded=0,scientific_generated=0,manuscript_edits=0,independent_reviewer=False))
    print('PASS_SAVED_DATA_REPLAY',sums,flush=True)
if __name__=='__main__':main()
