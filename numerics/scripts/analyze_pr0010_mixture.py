# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""PR-0010: fixed saved-data mixture stress analysis. No generator or fitting."""
from pathlib import Path
import argparse, hashlib, json, time, zipfile
import numpy as np
from analyze_pr0009_mechanisms import Archive
import pr0009_spec as spec
from analyze_pr0009_compression import ranks, project, safe

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'numerics/analysis/pr0010_9adfa7a8c6e0_mixture_v1'
DESIGN=ROOT/'audits/verification/2026-10-18_PR0009_mixture_design.json'
CARD=ROOT/'audits/prereg/PR-0010_mixture_dose_saved_data.md'
G=list(spec.groups()); EV=['normal',*spec.FAULTS]; AL=[.01,.001]; METHODS=['W0','LF0']
PATTERNS={'GM':[0],'GP':[1],'P':[2],'GMGP31':[0,0,0,1],
 'GMGP11':[0,1],'GMGP13':[0,1,1,1],'GMP11':[0,2],'GPP11':[1,2], 'GMGPP112':[0,1,2,2]}
OLDIDS=['clean','GM_n160','GM_n800','GP_n160','GP_n800','P_n160','P_n800']
OLD=ROOT/'numerics/analysis/pr0009_9adfa7a8c6e0_two_axis_v1'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(safe(x),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def layout(c):
    f=np.full(800,-1,np.int64);ids=np.full(800,-1,np.int64)
    name=c['parent'] or c['id']
    if name!='clean':
        kind,n=name.split('_n');n=int(n);f[:n]=np.resize(PATTERNS[kind],n)
        for k in range(3):ids[f==k]=np.arange(np.sum(f==k))
        if c['keep'] is not None:
            drop=~np.isin(f,c['keep']);f[drop]=-1;ids[drop]=-1
    assert [int(np.sum(f==k)) for k in range(3)]==c['counts']
    return f,ids

def recipes():
    d=load(DESIGN);unique=[];layouts=[];mapping={};keys={};aliases={}
    for c in d['candidates']:
        f,idx=layout(c);key=np.column_stack([f,idx]).tobytes()
        if key in keys:
            j=keys[key];aliases[c['id']]=unique[j]['id']
        else:
            j=len(unique);keys[key]=j;unique.append(c);layouts.append((f,idx))
        mapping[c['id']]=j
    assert len(unique)==57 and aliases==d['aliases']
    edges=[dict(parent=mapping[c['parent']],child=mapping[c['id']],id=c['id'])
           for c in d['candidates'] if c['parent']]
    return unique,layouts,mapping,edges

def assemble(base,faults,pos,lay):
    f,ids=lay;out=base.copy()
    for k in range(3):
        slot=np.flatnonzero(f==k);out[pos[slot]]=faults[k][ids[slot]]
    return out

def calibrate_scores(s,ordered):
    z=np.column_stack([s[:,0],-s[:,0],s[:,1]])
    rk=ranks(s,ordered);mx=rk.max(1);m=len(s)
    cut=np.zeros((2,2,3));rc=np.zeros(2,np.int64);ties=np.zeros((2,2,3),np.int64)
    for a,den in enumerate([100,1000]):
        for b,part in enumerate([2,2,4]):
            k=m+1-(m+1)*part//(8*den)
            cut[a,0,b]=np.partition(z[:,b],k-1)[k-1]
            ties[a,0,b]=np.sum(z[:,b]==cut[a,0,b])
        k=m+1-(m+1)//den;rc[a]=np.partition(mx,k-1)[k-1]
        ties[a,1,:]=np.sum(mx==rc[a])
        cut[a,1]=np.inf if rc[a]==len(ordered) else [ordered[rc[a],0],ordered[rc[a],0],ordered[rc[a],1]]
    top=np.array([(rk[:,0]==9999).sum(),(rk[:,1]==9999).sum(),(mx==9999).sum()])
    assert np.array_equal(rc==9999,top[2]>=np.array([160,16]))
    return cut,rc,top,ties

def gate_score(q,s,oldcut,window):
    mu=q.mean(0);sd=q.std(0,ddof=1);T=np.divide(np.sqrt(len(q))*mu,sd,out=np.full(2,np.nan),where=sd>0)
    z=np.column_stack([s[:,0],-s[:,0],s[:,1]])
    sc=np.array([np.any(z>oldcut,axis=1).mean(),abs(T[0]),max(abs(T[0]),T[1])])
    if not np.isfinite(T).all():sc[2]=np.nan
    st=np.where(np.isfinite(sc),sc>spec.GATE_CUTS[window],2).astype(np.uint8)
    return sc,st,T

def joint_many(left,right):
    """K x n integer patterns -> K x 8 x 8 joint tables."""
    left,right=np.broadcast_arrays(left,right);k,n=left.shape
    key=(left.astype(np.int64)*8+right)+64*np.arange(k)[:,None]
    return np.bincount(key.ravel(),minlength=k*64).reshape(k,8,8)

def patterns(s,cuts):
    z=np.column_stack([s[:,0],-s[:,0],s[:,1]])
    # K x N x alpha x method. Chunked; never retains trial bits across E chunks.
    out=np.empty((len(cuts),len(s),2,2),np.uint8)
    for a in range(2):
        for m in range(2):out[:,:,a,m]=np.sum((z[None,:,:]>cuts[:,None,a,m,:])*[1,2,4],axis=2,dtype=np.uint8)
    return out

def run_rep(arc,rep,cs,lays,mapping,edges):
    start=time.monotonic();folder=f'shards/{rep//5:02d}/rep_{rep:02d}';K=len(cs)
    row=dict(mean=np.zeros((6,K,2,4)),sd=np.zeros((6,K,2,4)),T=np.zeros((6,K,2,2)),
      bf=np.zeros((6,K,2,3,2)),varparts=np.zeros((6,K,2,2,2)),Tparts=np.zeros((6,K,2,2,2)),
      scores=np.zeros((6,K,3)),states=np.zeros((6,K,3),np.uint8),
      cut=np.zeros((6,K,2,2,3)),rank_cut=np.zeros((6,K,2),np.int64),
      top=np.zeros((6,K,3),np.int64),ties=np.zeros((6,K,2,2,3),np.int64),
      joint_F=np.zeros((6,K,4,2,2,8,8),np.int64),joint_C=np.zeros((6,K,4,2,2,8,8),np.int64),
      joint_methods=np.zeros((6,K,4,2,8,8),np.int64),
      joint_parent=np.zeros((6,len(edges),4,2,2,8,8),np.int64))
    oldindices=[mapping[x] for x in OLDIDS];parent=np.array([e['parent'] for e in edges]);child=np.array([e['child'] for e in edges])
    checks=dict(anchor_batches=0,anchor_cut_comparisons=0,anchor_record_method_alpha=0,role_orders=0,
                q_identity_max=0.,mean_identity_max=0.,variance_identity_max=0.,T_identity_max=0.,raw_records=0)
    sources={};old=np.load(OLD/'recomputed'/f'rep_{rep:02d}.npz')
    for gi,group in enumerate(G):
        win=int(group.endswith('_L'));base=spec.baseline(group)
        model=arc.npz(f'{folder}/models/{base}.npz');state=json.loads(str(model['state']));assert state['valid']
        mod=state['model'];assert not mod['disabled'];ordered=model['ordered']
        assert ordered.shape==(9999,2) and np.all(np.diff(ordered,axis=0)>=0)
        oc=arc.npz(f'{folder}/old_cal/{base}.npz');fc=np.empty((2,2,3));fc[:,0]=oc['cut'][:,0]
        for a in range(2):
            cr=int(oc['cut'][a,5,0]);fc[a,1]=np.inf if cr>=9999 else [ordered[cr,0],ordered[cr,0],ordered[cr,1]]
        rolepaths=[]
        for ri,role in enumerate(['audit','update']):
            vals=arc.values(folder,group,role+'_normal');checks['raw_records']+=len(vals)
            o=arc.npz(f'{folder}/orders/{group}_{role}.npz');order=o['normal_order'];pos=o['replacement_order']
            for perm in [order,pos]:assert np.array_equal(np.sort(perm),np.arange(15999))
            checks['role_orders']+=1;rolepaths.append(f'{folder}/chunks/{group}/{role}_normal/')
            q0=vals[order][:,[11,12]];m=len(q0);mu0=q0.mean(0);sd0=q0.std(0,ddof=1)
            faults=[]
            for f in spec.FAULTS:
                v=arc.values(folder,group+'__'+f,role+'_fault');checks['raw_records']+=len(v);faults.append(v[:,[11,12]])
            for ci,lay in enumerate(lays):
                q=assemble(q0,faults,pos,lay);s=project(q,mod);v=np.column_stack([q,s]);mu=q.mean(0);sd=q.std(0,ddof=1)
                assert len(q)==15999 and np.isfinite(v).all() and (sd>0).all()
                row['mean'][gi,ci,ri]=v.mean(0);row['sd'][gi,ci,ri]=v.std(0,ddof=1)
                sc,st,T=gate_score(q,s,oc['cut'][0,0],win);row['T'][gi,ci,ri]=T
                f,ids=lay;within=np.zeros(2);between=np.zeros(2);groups=[]
                keep=np.ones(m,bool);keep[pos[np.flatnonzero(f>=0)]]=False;groups.append(q0[keep])
                for k in range(3):
                    slot=np.flatnonzero(f==k)
                    if len(slot):
                        added=faults[k][ids[slot]];groups.append(added)
                        row['bf'][gi,ci,ri,k]=(added-q0[pos[slot]]).sum(0)/m
                for part in groups:
                    pm=part.mean(0);within+=np.square(part-pm).sum(0)/(m-1);between+=len(part)*np.square(pm-mu)/(m-1)
                row['varparts'][gi,ci,ri]=[within,between]
                assert np.allclose(within+between,sd**2,rtol=2e-10,atol=1e-16)
                err=np.max(np.abs(row['bf'][gi,ci,ri].sum(0)-(mu-mu0)));checks['mean_identity_max']=max(checks['mean_identity_max'],float(err))
                assert np.allclose(row['bf'][gi,ci,ri].sum(0),mu-mu0,rtol=2e-10,atol=2e-13)
                checks['variance_identity_max']=max(checks['variance_identity_max'],float(np.max(abs(within+between-sd**2))))
                mp=np.sqrt(m)/2*(mu-mu0)*(1/sd0+1/sd);sp=np.sqrt(m)/2*(mu+mu0)*(1/sd-1/sd0)
                row['Tparts'][gi,ci,ri]=[mp,sp];err=np.max(abs(T-np.sqrt(m)*mu0/sd0-mp-sp))
                checks['T_identity_max']=max(checks['T_identity_max'],float(err));assert err<1e-10
                if ri==0:row['scores'][gi,ci]=sc;row['states'][gi,ci]=st
                else:
                    cut,rc,top,ties=calibrate_scores(s,ordered)
                    row['cut'][gi,ci]=cut;row['rank_cut'][gi,ci]=rc;row['top'][gi,ci]=top;row['ties'][gi,ci]=ties
                if ci in oldindices:
                    oi=oldindices.index(ci)
                    if ri==0:
                        saved=arc.npz(f'{folder}/audit/{group}/candidate_{oi}.npz')
                        assert np.allclose(sc,saved['scores'],rtol=2e-10,atol=2e-13)
                        assert np.array_equal(st,saved['states']) and np.array_equal(st,old['gate_states'][gi,oi]);checks['anchor_batches']+=1
                    else:
                        saved=arc.npz(f'{folder}/calibration/{group}/candidate_{oi}.npz')
                        assert np.allclose(cut[:,0],saved['cut'][:,0],rtol=2e-10,atol=2e-13)
                        assert np.array_equal(rc,saved['cut'][:,5,0]);checks['anchor_cut_comparisons']+=1
        assert rolepaths[0]!=rolepaths[1]
        # Freeze A states and U cuts before reading this group's E scientific values.
        sources[group]={'A':rolepaths[0],'U':rolepaths[1],
          'states_sha256':hashlib.sha256(row['states'][gi].tobytes()).hexdigest(),
          'cuts_sha256':hashlib.sha256(row['cut'][gi].tobytes()).hexdigest()}
        frozen=OUT/'frozen_candidates'/f'rep_{rep:02d}_{group}.npz'
        payload={k:row[k][gi] for k in ['scores','states','cut','rank_cut','top','ties']}
        if frozen.exists():
            with np.load(frozen) as zz:
                for k,v in payload.items():assert np.array_equal(zz[k],v,equal_nan=True)
        else:np.savez_compressed(frozen,**payload)
        sources[group]['frozen_before_E_sha256']=sha(frozen)
        for ei,ev in enumerate(EV):
            case=group if ei==0 else group+'__'+ev;role='null_eval' if ei==0 else 'fault_eval'
            vals=arc.values(folder,case,role);checks['raw_records']+=len(vals);n=len(vals)
            q=vals[:,[11,12]];nu=spec.nominal_nu()
            assert np.allclose(q[:,0],vals[:,8]-(1-nu)*vals[:,6]-nu*vals[:,7],rtol=2e-10,atol=2e-13)
            assert np.allclose(q[:,1],vals[:,14]+vals[:,15]-2*vals[:,16],rtol=2e-10,atol=2e-13)
            for lo in range(0,n,4096):
                block=vals[lo:lo+4096];s=project(block[:,[11,12]],mod)
                pp=patterns(s,row['cut'][gi]);fp=patterns(s,fc[None])[0];cp=pp[0]
                # Saved seven candidates + F are exact integer/bit anchors.
                saved=arc.npz(f'{folder}/decisions/{case}/{lo:09d}.npz');bits=saved['packed'];md=json.loads(str(saved['metadata']))
                assert md['start']==lo and md['raw_sha256']==hashlib.sha256(block.tobytes()).hexdigest()
                assert md['packed_sha256']==hashlib.sha256(bits.tobytes()).hexdigest()
                for ai in range(2):
                    for mi,shift in enumerate([0,20]):
                        nib=(bits[:,:,ai]>>shift)&15;assert np.all(nib&8)
                        assert np.array_equal(fp[:,ai,mi],nib[:,0]&7)
                        assert np.array_equal(pp[oldindices,:,ai,mi].T,nib[:,1:]&7)
                        checks['anchor_record_method_alpha']+=len(block)*8
                        row['joint_F'][gi,:,ei,ai,mi]+=joint_many(fp[None,:,ai,mi],pp[:,:,ai,mi])
                        row['joint_C'][gi,:,ei,ai,mi]+=joint_many(cp[None,:,ai,mi],pp[:,:,ai,mi])
                        row['joint_parent'][gi,:,ei,ai,mi]+=joint_many(pp[parent,:,ai,mi],pp[child,:,ai,mi])
                    row['joint_methods'][gi,:,ei,ai]+=joint_many(pp[:,:,ai,0],pp[:,:,ai,1])
                # Independently evaluate LF0 by ranks, not equivalent continuous cut.
                rk=ranks(s,ordered)
                for ai in range(2):
                    bg=rk[None,:,0]>row['rank_cut'][gi,:,None,ai];bp=rk[None,:,1]>row['rank_cut'][gi,:,None,ai]
                    rp=((bg&(s[None,:,0]>=0))+2*(bg&(s[None,:,0]<0))+4*bp).astype(np.uint8)
                    assert np.array_equal(rp,pp[:,:,ai,1])
            for key in ['joint_F','joint_C','joint_parent','joint_methods']:
                assert np.all(row[key][gi,:,ei].sum(axis=(-1,-2))==n)
            for ci,oi in zip(oldindices,range(7)):
                hist=row['joint_F'][gi,ci,ei].sum(-2)
                assert np.array_equal(hist,old['counts'][gi,ei,oi+1][:,[0,5]])
    old.close()
    return row,dict(replicate=rep,checks=checks,sources=sources,elapsed_seconds=time.monotonic()-start)

def prepare():
    OUT.mkdir(exist_ok=True);(OUT/'recomputed').mkdir(exist_ok=True);(OUT/'bundles').mkdir(exist_ok=True)
    (OUT/'frozen_candidates').mkdir(exist_ok=True)
    prior=load(ROOT/'audits/verification/2026-10-18_PR0009_mixture_design_closeout.json')
    protected=load(DESIGN)['protected'].copy()
    for key in ['outputs','frozen_sources']:protected.update(prior[key])
    protected[str(Path('audits/verification/2026-10-18_PR0009_mixture_design_closeout.json'))]=sha(ROOT/'audits/verification/2026-10-18_PR0009_mixture_design_closeout.json')
    for p,h in protected.items():assert sha(ROOT/p)==h,p
    source_files=[CARD,Path(__file__),ROOT/'numerics/design/PR0009_MIXTURE_DOSE_DESIGN_v1_JA.md']
    inputs={str(p.relative_to(ROOT)):sha(p) for p in source_files}
    p=OUT/'start_snapshot.json'
    if p.exists():
        snap=load(p);assert snap['inputs']==inputs,'Frozen implementation changed; use a new version'
    else:
        ledger=(ROOT/'audits/ledger_2026-10.md').read_bytes()
        dump(p,dict(inputs=inputs,protected=protected,ledger_bytes=len(ledger),ledger_sha256=hashlib.sha256(ledger).hexdigest(),
             kind='post-hoc saved-data analysis; D-A approved',new_simulation=False))
    cs,lays,mapping,edges=recipes()
    manifest=dict(unique=cs,mapping=mapping,edges=edges,groups=G,evaluations=EV,methods=METHODS,alpha=AL,
                  shapes='joint_*: group,candidate/edge,evaluation,alpha,method,pattern-left,pattern-right')
    if not (OUT/'recipes.json').exists():dump(OUT/'recipes.json',manifest)
    return inputs,cs,lays,mapping,edges

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stop',type=int,default=50);args=ap.parse_args()
    inputs,cs,lays,mapping,edges=prepare();arc=Archive();start=time.monotonic();receipts=[]
    for rep in range(args.stop):
        path=OUT/'recomputed'/f'rep_{rep:02d}.npz';receipt=path.with_suffix('.json')
        if receipt.exists():
            rr=load(receipt);assert rr['inputs']==inputs and sha(path)==rr['output_sha256'];receipts.append(rr);continue
        assert not path.exists(),'Incomplete output: retain and inspect before recovery'
        checked_before=set(arc.checked)
        row,rr=run_rep(arc,rep,cs,lays,mapping,edges)
        np.savez_compressed(path,**row)
        rr.update(inputs=inputs,output_sha256=sha(path),read_members={k:v for k,v in arc.checked.items() if k not in checked_before},archives=arc.archives)
        dump(receipt,rr);receipts.append(rr)
        print(f'COMPLETE {rep+1}/50: {rr["elapsed_seconds"]:.2f}s; {path.stat().st_size/1e6:.2f} MB',flush=True)
        if rep==0:print('FIRST_REP technical estimate: '+str(round(50*rr['elapsed_seconds']/60,2))+' minutes excluding QA/report; conditions unchanged',flush=True)
        if (rep+1)%5==0:
            zp=OUT/'bundles'/f'PR0010_shard_{rep//5+1:02d}.zip'
            if not zp.exists():
                with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED) as z:
                    for k in range(rep-4,rep+1):
                        for ext in ['npz','json']:
                            src=OUT/'recomputed'/f'rep_{k:02d}.{ext}';z.write(src,src.name)
                    z.write(OUT/'recipes.json','recipes.json');z.write(CARD,CARD.name)
                assert zp.stat().st_size<1_000_000_000
    if args.stop==50:
        members={};checks={}
        for rr in receipts:
            members.update(rr['read_members'])
            for k,v in rr['checks'].items():checks[k]=(max(checks.get(k,0),v) if k.endswith('_max') else checks.get(k,0)+v)
        snap=load(OUT/'start_snapshot.json')
        for p,h in snap['protected'].items():assert sha(ROOT/p)==h,p
        dump(OUT/'validation.json',dict(status='PASS_SAVED_DATA_RECONSTRUCTION',inputs=inputs,archives=arc.archives,
          checks=checks,checked_members=members,replicates=50,unique_candidates=57,main_candidates=37,
          auxiliary_labels=24,alias_count=4,planned_AU_batches=34200,unknown=0,missing=0,excluded=0,
          science_generated=0,manuscript_edits=0,elapsed_rep_seconds=sum(r['elapsed_seconds'] for r in receipts),
          wall_this_invocation=time.monotonic()-start,independent_reviewer=False))
        print('PASS_SAVED_DATA_RECONSTRUCTION',checks,flush=True)

if __name__=='__main__':main()
