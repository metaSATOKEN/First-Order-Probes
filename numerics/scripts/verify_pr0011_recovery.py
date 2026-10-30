# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Independent integer replay and original packed E audit for PR-0011."""
import hashlib, json, time
from pathlib import Path
import numpy as np
from scipy.stats import binom
from analyze_pr0009_mechanisms import Archive

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'numerics/analysis/pr0011_9adfa7a8c6e0_recovery_v2'
PREV=ROOT/'numerics/analysis/pr0010_9adfa7a8c6e0_mixture_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def classify(k,a,eta=.00625):
    return np.where(binom.cdf(k,20000,1.25*a)<=eta,0,
                    np.where(binom.sf(k-1,20000,1.25*a)<eta,1,2)).astype('uint8')
def main():
    start=time.monotonic();arc=Archive();s=json.loads((OUT/'schema.json').read_text())
    assert json.loads((OUT/'validation.json').read_text())['status']=='PASS_SAVED_DATA_REPLAY'
    recipes=json.loads((PREV/'recipes.json').read_text())['unique']
    ids=[i for i,x in enumerate(recipes) if x['parent'] is None]
    checks=dict(raw_E_records=0,raw_F_C0_tables=0,policy_pairs=0,CP_labels=0,unknown=0)
    xx,yy=np.indices((8,8));one=np.arange(8)>0
    for r in range(50):
        with np.load(OUT/'recomputed'/f'rep_{r:02d}.npz') as z:d={k:z[k] for k in z.files}
        with np.load(PREV/'recomputed'/f'rep_{r:02d}.npz') as z:
            jf=np.take(z['joint_F'][:,:,:,:,0],ids,axis=1)
            jc=np.take(z['joint_C'][:,:,:,:,0],ids,axis=1)
        for a,alpha in enumerate(s['alphas']):
            assert np.array_equal(classify(d['v_counts'][:,:,a],alpha),d['v_labels'][:,:,a])
            assert np.array_equal(classify(d['v_counts'][:,0,a],alpha,.0125),d['old_labels'][:,a])
            en=d['base'][:,0,0,a,[1,2],1:].sum(-1).T
            # Explicit slice stack avoids NumPy advanced-index axis relocation.
            en=np.stack([d['base'][:,0,0,a,b,1:].sum(-1) for b in [1,2]],axis=1)
            assert np.array_equal(classify(en,alpha),d['e_labels'][:,:,a])
            checks['CP_labels']+=30
        actions=np.zeros_like(d['actions'])
        for g in range(6):
            for k in range(37):
                for gate in range(3):
                    h=int(d['states'][g,k,gate]);assert h in [0,1]
                    for a in range(2):
                        if h==0:act=[0,0,0]
                        elif d['v_labels'][g,0,a]==0:act=[1,1,1]
                        elif d['v_labels'][g,1,a]==0:act=[1,3,2]
                        else:act=[1,3,3]
                        actions[g,k,gate,a]=act
        assert np.array_equal(actions,d['actions'])
        for g,group in enumerate(s['groups']):
            folder=f'shards/{r//5:02d}/rep_{r:02d}'
            for e,ev in enumerate(s['evaluations']):
                n=20000 if e==0 else 5000;case=group if e==0 else group+'__'+ev
                raw=np.zeros((2,8,8),np.int64)
                for lo in range(0,n,4096):
                    z=arc.npz(f'{folder}/decisions/{case}/{lo:09d}.npz')
                    p=z['packed'];md=json.loads(str(z['metadata']))
                    assert md['start']==lo and md['count']==len(p) and md['task']['case']==case
                    assert md['task']['replicate']==r
                    assert md['task']['role']==('null_eval' if e==0 else 'fault_eval')
                    assert md['packed_sha256']==hashlib.sha256(p.tobytes()).hexdigest()
                    nib=p[:,:2,:]&15;assert np.all(nib&8)
                    for a in range(2):raw[a]+=np.bincount(((nib[:,0,a]&7)*8+(nib[:,1,a]&7)).astype(int),minlength=64).reshape(8,8)
                    checks['raw_E_records']+=len(p)
                assert np.array_equal(raw,jf[g,0,e]);checks['raw_F_C0_tables']+=2
                for a in range(2):
                    for k in range(37):
                        f,u=jf[g,k,e,a],jc[g,k,e,a]
                        hist=np.stack([f.sum(0),f.sum(1),u.sum(1)])
                        assert np.array_equal(hist,d['base'][g,k,e,a])
                        assert np.array_equal(f.sum(0),u.sum(0))
                        for gate in range(3):
                            acts=actions[g,k,gate,a]
                            for pol,act in enumerate(acts):
                                expected=np.zeros(4,np.int64)
                                if act<3:expected[:2]=[hist[act,1:].sum(),hist[act,0]]
                                else:expected[2]=n
                                assert np.array_equal(expected,d['counts'][g,k,gate,a,pol,e])
                                sh=hist[:,1:].sum(1) if act==3 else np.zeros(3,np.int64)
                                assert np.array_equal(sh,d['shadows'][g,k,gate,a,pol,e])
                            for pi,(lp,rp) in enumerate(s['pairs']):
                                l,rr=map(int,[acts[lp],acts[rp]])
                                if l<3 and rr<3 and l!=rr:
                                    maps={(1,0):f,(2,0):u,(1,2):raw[a]}
                                    j=maps[(l,rr)] if (l,rr) in maps else maps[(rr,l)].T
                                    table=np.bincount(((xx==0)*4+(yy==0)).ravel(),weights=j.ravel(),minlength=16).reshape(4,4).astype(np.int64)
                                elif l<3 and rr<3:
                                    table=np.diag([hist[l,1:].sum(),hist[l,0],0,0])
                                else:
                                    table=np.zeros((4,4),np.int64)
                                    if l>=3:table[2]=d['counts'][g,k,gate,a,rp,e]
                                    else:table[:,2]=d['counts'][g,k,gate,a,lp,e]
                                assert np.array_equal(table,d['paired'][g,k,gate,a,pi,e])
                                assert table.sum()==n
                                gain=table[0,1:].sum();loss=table[1:,0].sum()
                                assert gain-loss==d['counts'][g,k,gate,a,lp,e,0]-d['counts'][g,k,gate,a,rp,e,0]
                                if pi==0:assert gain==0 # P1 only withdraws P0 decisions
                                if pi==1:assert loss==0 # P2 only recovers P1 deferrals
                                checks['policy_pairs']+=1
        print('VERIFY',r,'PASS',flush=True)
    assert checks['raw_E_records']==10500000 and checks['policy_pairs']==799200
    snap=json.loads((OUT/'start_snapshot.json').read_text())
    for p,h in {**snap['protected'],**snap['inputs']}.items():assert sha(ROOT/p)==h,p
    result=dict(status='PASS_SECOND_PATH',checks=checks,elapsed_seconds=time.monotonic()-start,
        code_sha256=sha(__file__),raw_members=arc.checked,archives=arc.archives,
        scope='Original packed E F/C0, all stored mixed-U tables, CP via binomial tails, all policy choices and 4x4 pairs. No new raw mixed-U recalculation.',
        independent_reviewer=False)
    (OUT/'second_path_validation.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
