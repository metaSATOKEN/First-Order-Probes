# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Local technical tests only: deterministic statistic fixtures, no scientific RNG/GPU run."""
import ast
import base64
import io
import json
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import zlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'numerics/scripts'))
import pr0006_spec as s
import pr0006_methods as m
import pr0006_io as d
import pr0006_runner as r
import pr0006_launcher as l


def fixture(n,offset=0):
    """Artificial sufficient-statistic records, not generated physical trials."""
    t=np.arange(n,dtype=np.float64)+offset;a=np.zeros((n,17),np.float64)
    _,_,_,lam,nu=s.core.coefficients(s.conditions()['B_M']['params'])
    for k in (0,3,6):
        a[:,k]=1+.08*np.sin(t*.173);a[:,k+1]=.8+.06*np.cos(t*.31)
        a[:,k+2]=lam*a[:,k]+nu*a[:,k+1]+.01*np.sin(t*.37)
    a[:,9]=a[:,2]-lam*a[:,0]-nu*a[:,1];a[:,10]=a[:,5]-lam*a[:,3]-nu*a[:,4]
    a[:,11]=a[:,8]-lam*a[:,6]-nu*a[:,7]
    a[:,14]=.4+.01*np.cos(t*.19);a[:,15]=.5+.01*np.sin(t*.23);a[:,16]=.42
    a[:,12]=a[:,14]+a[:,15]-2*a[:,16]
    return a


class PlanTests(unittest.TestCase):
    def test_dimensions_and_counts(self):
        self.assertEqual(len(s.conditions()),44);self.assertEqual(len(s.contexts()),76)
        self.assertEqual(sum(t['count'] for t in s.tasks(0)),549988)
        self.assertEqual(25*549988,13749700);self.assertEqual(len(s.primary()),25)
        self.assertEqual(sum(t['count'] for t in s.tasks(0) if t['role'] not in ('ref','cal'))*25,9250000)
        self.assertEqual(sum(s.COUNTS['null' if s.conditions()[c]['kind']=='normal' else s.conditions()[c]['kind']] for c,p in s.contexts())*25,16250000)
    def test_rng_addresses_unique_without_generation(self):
        cases=s.conditions()
        addresses=[(s.MASTER,cases[t['case']]['id'],s.ROLES[t['role']],rep,b)
            for rep in range(25) for t in s.tasks(rep) for b in range((t['count']+63)//64)]
        self.assertEqual(len(addresses),len(set(addresses)));self.assertNotEqual(s.MASTER,s.PREFLIGHT_MASTER)
        self.assertNotIn(s.MASTER,(2026101704,2026101705))
    def test_generator_rechunking_with_deterministic_fake_rng(self):
        # Exercise construction and addresses, replacing RNG: no PR-0006 draws.
        seen=[]
        class Fake:
            def __init__(self,seed):self.offset=int(sum(seed));self.counter=0
            def normal(self,size):
                self.counter+=1
                return np.broadcast_to((np.arange(size[0])+self.offset+self.counter)[:,None]*1e-7,size).copy()
        def seed(x):seen.append(tuple(x));return x
        with (patch.object(s.np.random,'SeedSequence',side_effect=seed),patch.object(s.np.random,'PCG64',side_effect=lambda x:x),
              patch.object(s.np.random,'Generator',side_effect=Fake)):
            a=s.make_inputs('B_M__P02','preflight',2,62,5)
            b=s.make_inputs('B_M__P02','preflight',2,62,2);c=s.make_inputs('B_M__P02','preflight',2,64,3)
        for k in a:np.testing.assert_array_equal(a[k],np.concatenate((b[k],c[k])))
        self.assertTrue(all(x[0]==s.PREFLIGHT_MASTER for x in seen));self.assertIn('private_source',a)
    def test_integer_alarm_budgets(self):
        for den in (100,1000):
            for weights in s.BUDGETS.values():
                tails=[16000-m.order_index(15999,den,w) for w in weights]
                self.assertEqual(sum(tails),16000//den)
                self.assertEqual(tails,[16000*w//(8*den) for w in weights])
    def test_card_freeze_and_core(self):
        f=d.read(ROOT/'audits/verification/2026-10-17_PR-0006_design_freeze.json')
        self.assertEqual(f['card_sha256'],s.CARD_SHA)
        self.assertEqual(d.digest(ROOT/'audits/prereg/PR-0006_lfdc_alarm_budget_confirmation.md'),s.CARD_SHA)
        self.assertEqual(d.digest(ROOT/'numerics/scripts/probe_backend_check_v1.py'),s.CORE_SHA)


class MethodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref=fixture(9999);cls.cal=fixture(15999,11000);cls.state,cls.ordered=m.fit(cls.ref)
        cls.meta,cls.cut,cls.indices=m.calibrate(cls.state,cls.ordered,cls.cal)
    def test_fit_is_frozen_projection(self):
        old,ordered=m.old.fit_reference(self.ref,s.nominal_nu())
        self.assertEqual(self.state['model'],old);np.testing.assert_array_equal(self.ordered,ordered['LF'])
    def test_direct_sort_oracle(self):
        v=self.cal[:,[11,12]];model=self.state['model'];z=(v-np.array(model['mean']))@np.array(model['weights']).T
        z=np.column_stack((z[:,0],-z[:,0],z[:,1]))
        for ai,den in enumerate((100,1000)):
            for wi,weights in enumerate(s.BUDGETS.values()):
                for b,w in enumerate(weights):self.assertEqual(self.cut[ai,wi,b],sorted(z[:,b])[16000-16000*w//(8*den)-1])
    def test_lf0_matches_old_main_rule(self):
        state=self.state['model'];_,ordered=m.old.fit_reference(self.ref,s.nominal_nu())
        oldcut=m.old.calibrate(state,ordered,self.cal)
        oldflags,_,_=m.old.decide(state,ordered,oldcut,fixture(80,30000))
        packed=m.decide(self.state,self.ordered,self.meta,self.cut,fixture(80,30000))
        patterns,available=m.decode(packed)
        np.testing.assert_array_equal(oldflags[:,4],patterns[:,0,5]>0);self.assertTrue(available.all())
    def test_independent_decision_oracle(self):
        a=fixture(80,30000);sg,sp=((a[:,[11,12]]-self.state['model']['mean'])@np.array(self.state['model']['weights']).T).T
        patterns,available=m.decode(m.decide(self.state,self.ordered,self.meta,self.cut,a))
        for ai in range(2):
            for wi in range(5):
                for k in range(len(a)):
                    want=sum(1<<b for b,x in enumerate((sg[k],-sg[k],sp[k])) if x>self.cut[ai,wi,b])
                    self.assertEqual(int(patterns[k,ai,wi]),want)
    def test_ties_are_strict(self):
        state=dict(valid=True,model=dict(mean=[0.,0.],weights=[[1.,0.],[0.,1.]],disabled=[]))
        a=np.zeros((15999,17),np.float64);ordered=np.zeros((9999,2))
        meta,cut,_=m.calibrate(state,ordered,a);patterns,valid=m.decode(m.decide(state,ordered,meta,cut,a[:5]))
        self.assertTrue(valid.all());self.assertFalse(patterns.any())
        self.assertTrue(meta['lf0_structurally_inactive']==[False,False])
    def test_upper_rank_saturation(self):
        state=dict(valid=True,model=dict(mean=[0.,0.],weights=[[1.,0.],[0.,1.]],disabled=[]))
        a=np.zeros((15999,17));a[:,11:13]=2.;ordered=np.ones((9999,2))
        meta,cut,_=m.calibrate(state,ordered,a)
        self.assertEqual(meta['lf0_structurally_inactive'],[True,True])
        p,_=m.decode(m.decide(state,ordered,meta,cut,a[:4]*3));self.assertFalse(p[:,:,5].any())
    def test_insufficient_calibration(self):
        meta,cut,idx=m.calibrate(self.state,self.ordered,self.cal[:2]);self.assertTrue(np.isinf(cut[:,:5]).all())
        self.assertEqual(meta['lf0_structurally_inactive'],[True,True])
    def test_nonfinite_ref_cal_eval_unavailable(self):
        bad=self.ref.copy();bad[0,0]=np.nan;state,_=m.fit(bad);self.assertFalse(state['valid'])
        a=self.cal.copy();a[0,0]=np.inf;meta,cut,_=m.calibrate(self.state,self.ordered,a);self.assertFalse(meta['valid'])
        _,valid=m.decode(m.decide(self.state,self.ordered,meta,cut,fixture(4)));self.assertFalse(valid.any())
        a=fixture(4);a[1,0]=np.nan;_,valid=m.decode(m.decide(self.state,self.ordered,self.meta,self.cut,a))
        self.assertFalse(valid[1].any());self.assertTrue(valid[[0,2,3]].all())
    def test_nonpositive_variance_and_disabled(self):
        state,_=m.fit(np.ones((20,17),np.float64));self.assertFalse(state['valid'])
        a=fixture(9999);a[:,7:9]=0.;a[:,15:17]=0.
        state,ordered=m.fit(a);self.assertTrue(state['valid']);self.assertEqual(state['model']['disabled'],[0,1])
        meta,cut,_=m.calibrate(state,ordered,a)
        self.assertTrue(np.isinf(cut[:,:5]).all());self.assertEqual(meta['disabled_tails'],[0,1,2])
    def test_negative_statistics_retained(self):
        a=fixture(5);a[0,11:13]=[-10.,-10.];scores,ok=m.signed(self.state,a)
        self.assertTrue(ok.all());self.assertFalse(np.array_equal(scores[0],scores[1]))
    def test_packed_pairs_all_patterns(self):
        packed=np.zeros((8,2),np.uint32)
        for mi in range(6):packed|=(((np.arange(8,dtype=np.uint32)+mi)%8)|8)[:,None]<<(4*mi)
        c,a,p,n=m.count(packed);self.assertTrue((c==1).all());self.assertTrue((a==8).all());self.assertTrue((n==8).all())
        for pi,(left,right) in enumerate(s.PAIRS):np.testing.assert_array_equal(p[:,pi,1]-p[:,pi,2],c[:,left,1:].sum(-1)-c[:,right,1:].sum(-1))
    def test_invalid_bits(self):
        for word in (1,1<<24):
            with self.assertRaises(ValueError):m.decode(np.array([[word,0]],np.uint32))
    def test_outer_families_and_no_zero_sd_claim(self):
        results=[]
        for rep in range(25):
            c=np.zeros((76,2,6,8),np.int64);a=np.zeros((76,2,6),np.int64);p=np.zeros((76,2,15,4),np.int64);n=np.zeros((76,2,15),np.int64)
            for ki,(case,_) in enumerate(s.contexts()):
                kind=s.conditions()[case]['kind'];total=s.COUNTS['null' if kind=='normal' else kind]
                c[ki,:,:,0]=total-rep;c[ki,:,:,1]=rep;a[ki]=total;p[ki,:,:,0]=rep;p[ki,:,:,3]=total-rep;n[ki]=total
            results.append(dict(replicate=rep,counts=c,available=a,pairs=p,pair_n=n,inactive=np.zeros((76,2),np.int64)))
        report=m.aggregate(results);self.assertEqual(len(report['primary25']),25);self.assertEqual(len(report['fpr216']),216)
        self.assertTrue(all(x['classification']=='unresolved' for x in report['primary25']))
        self.assertTrue(all(x['interval']['lower'] is None for x in report['primary25']))
        self.assertEqual(len(report['rates'][0]['per_replicate']),25)
        results[0]['available'][s.contexts().index(('B_L__GM05','B')),0,0]-=1
        report=m.aggregate(results);self.assertEqual(report['primary25'][0]['interval']['inference'],'unavailable records')


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.base=Path(self.temp.name);self.root=self.base/'run'
        self.sources={'test.py':'raise RuntimeError("MUST NOT EXECUTE RESTORED SOURCE")\n'}
        d.new_run(self.root,'testcell',self.sources)
    def tearDown(self):self.temp.cleanup()
    def fake_rep(self,rep):
        f=d.rep_dir(self.root,rep);d.atomic(f/'tiny.bin',f'fixture {rep}'.encode())
        p=self.root/'carry'/f'rep_{rep:02d}.npz';d.atomic(p,b'opaque mock statistics; not scientific evidence')
        meta=dict(replicate=rep,records=549988,statistics_sha256=d.digest(p),files={'tiny.bin':d.digest(f/'tiny.bin')})
        d.write(p.with_suffix('.json'),meta);state=d.read(self.root/'state.json')
        state['completed'][f'{rep:02d}']=dict(metadata_sha256=d.digest(p.with_suffix('.json')));d.write(self.root/'state.json',state)
    def test_roundtrip_without_source_execution(self):
        z=d.snapshot(self.root,self.base/'zip','PARTIAL',[])[0];new=self.base/'restored'
        self.assertEqual(d.restore(z,new,'testcell',self.sources),[]);d.check_run(new,'testcell',self.sources)
        with self.assertRaises(FileExistsError):d.restore(z,new,'testcell',self.sources)
    def test_noncumulative_carry_roundtrip(self):
        for rep in range(5):self.fake_rep(rep)
        first=d.snapshot(self.root,self.base/'zip','SHARD_01_COMPLETE',range(5))[0]
        for rep in range(5,10):self.fake_rep(rep)
        second=d.snapshot(self.root,self.base/'zip','SHARD_02_COMPLETE',range(5,10))[0]
        idx=d.inspect_zip(second,'testcell');self.assertFalse(any(n.startswith('shards/00/') for n in idx['entries']))
        self.assertIn('carry/rep_00.json',idx['entries']);new=self.base/'restored'
        self.assertEqual(d.restore(second,new,'testcell',self.sources),list(range(5)))
        self.assertEqual(d.verify_carry(new),list(range(5)));self.assertTrue(first.exists())
    def test_carry_tamper_and_missing_receipt(self):
        self.fake_rep(0);d.verify_carry(self.root);shutil.rmtree(d.rep_dir(self.root,0))
        with self.assertRaises(ValueError):d.verify_carry(self.root)
    def test_cross_shard_rejected(self):
        with self.assertRaises(ValueError):d.snapshot(self.root,self.base/'zip','PARTIAL',[4,5])
    def test_prepack_and_actual_size_gates(self):
        with patch.object(d,'PREPACK_LIMIT',1):
            with self.assertRaises(OverflowError):d.snapshot(self.root,self.base/'zip','PARTIAL',[])
        with patch.object(d,'ZIP_LIMIT',20):
            with self.assertRaises(OverflowError):d.snapshot(self.root,self.base/'zip','PARTIAL',[])
        self.assertFalse(list((self.base/'zip').glob('*.zip')))
    def test_fixed_fallback_last_part_restores(self):
        for rep in range(5):self.fake_rep(rep)
        original=d.snapshot_one
        def gate(root,dest,status,reps):
            if len(reps)>1:raise OverflowError('mock size gate')
            return original(root,dest,status,reps)
        with patch.object(d,'snapshot_one',side_effect=gate):paths=d.snapshot(self.root,self.base/'zip','SHARD',range(5))
        self.assertEqual(len(paths),5);self.assertEqual(d.restore(paths[-1],self.base/'new','testcell',self.sources),list(range(4)))
    def test_resnapshot_never_claims_remote_raw_coverage(self):
        for rep in range(5):self.fake_rep(rep)
        original=d.snapshot_one
        def gate(root,dest,status,reps):
            if len(reps)>1:raise OverflowError('mock size gate')
            return original(root,dest,status,reps)
        with patch.object(d,'snapshot_one',side_effect=gate):parts=d.snapshot(self.root,self.base/'zip','SHARD',range(5))
        restored=self.base/'restored';d.restore(parts[-1],restored,'testcell',self.sources)
        new=d.snapshot(restored,self.base/'newzip','PARTIAL',range(5))[0]
        self.assertEqual(d.inspect_zip(new,'testcell')['covered_replicates'],[4])
        self.assertEqual(d.restore(new,self.base/'again','testcell',self.sources),list(range(4)))
    def test_zip_checksum_wrong_cell_and_capsule(self):
        z=d.snapshot(self.root,self.base/'zip','PARTIAL',[])[0]
        with self.assertRaises(ValueError):d.inspect_zip(z,'wrongcell')
        with self.assertRaises(ValueError):d.restore(z,self.base/'bad','testcell',{'test.py':'changed'})
        target=self.base/'corrupt.zip'
        with zipfile.ZipFile(z) as src,zipfile.ZipFile(target,'w') as dst:
            for n in src.namelist():dst.writestr(n,b'altered' if n=='state.json' else src.read(n))
        with self.assertRaises(ValueError):d.inspect_zip(target,'testcell')
    def test_path_and_symlink_rejected(self):
        for name in ('../evil','/tmp/evil','shards/../../evil'):
            with self.assertRaises(ValueError):d.allowed(name)
        z=self.base/'symlink.zip';i=zipfile.ZipInfo('sources/a');i.create_system=3;i.external_attr=(stat.S_IFLNK|0o777)<<16
        with zipfile.ZipFile(z,'w') as f:f.writestr(i,'/tmp/evil')
        with self.assertRaises(ValueError):d.inspect_zip(z,'testcell')
    def test_existing_runtime_never_uploads(self):
        root=self.base/'runs/current_testcell';root.parent.mkdir();shutil.move(self.root,root)
        actual,mode=l.resolve_run(self.base,'testcell',self.sources,'UPLOAD',upload=lambda:self.fail('must not prompt'))
        self.assertEqual(actual,root);self.assertEqual(mode,'existing')
    def test_auto_latest_uploaded_zip_and_lineage_collision(self):
        z=d.snapshot(self.root,self.base/'zip','PARTIAL',[])[0]
        root,mode=l.resolve_run(self.base/'freshbase','testcell',self.sources,content=z.parent)
        self.assertEqual(mode,'restored');self.assertTrue(root.exists())
        second=self.base/'run2';d.new_run(second,'testcell',self.sources);z2=d.snapshot(second,z.parent,'PARTIAL',[])[0]
        with self.assertRaises(ValueError):d.choose_archive([z,z2],'testcell')
    def test_active_budget_survives_restart(self):
        p=self.base/'time.json';d.write(p,dict(seconds=20));clock=d.ActiveClock(p,10)
        with self.assertRaises(TimeoutError):clock.check()
        self.assertGreaterEqual(d.read(p)['seconds'],20)


class WorkerTests(unittest.TestCase):
    def test_npz_immutable_and_no_pickle(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'a.npz';r.save_npz(p,a=np.array([1.,np.inf,np.nan]));r.save_npz(p,a=np.array([1.,np.inf,np.nan]))
            with self.assertRaises(ValueError):r.save_npz(p,a=np.array([2.,np.inf,np.nan]))
            r.save_npz(Path(td)/'bad.npz',a=np.array([{}],object))
            with self.assertRaises(ValueError):r.open_npz(Path(td)/'bad.npz')
    def test_chunk_checksum(self):
        with tempfile.TemporaryDirectory() as td:
            t=dict(id='00/B_M/ref',replicate=0,case='B_M',role='ref',count=5);a=fixture(5)
            meta=dict(task=t,start=0,specification_sha256='hash',values_sha256=d.sha(a.tobytes()),unavailable_rows=0)
            r.save_npz(r.chunk_path(td,t,0),values=a,metadata=np.array(r.canonical(meta)))
            np.testing.assert_array_equal(r.read_chunk(td,t,0,'hash'),a)
            with self.assertRaises(ValueError):r.read_chunk(td,t,0,'wrong')
    def test_mock_worker_phase_order_resume_and_immutable_calibration(self):
        # Reduced fixture counts ONLY under monkeypatch; physical RNG is forbidden.
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'run';d.new_run(root,'test',{'a.py':'# fixture'})
            original=s.tasks;small=[dict(t,count=8) for t in original(0)];calls=[]
            def generator(root,t,start,h,gpu):
                calls.append(t['role']);a=fixture(min(s.CHUNK,t['count']-start),offset=s.conditions()[t['case']]['id']*31)
                meta=dict(task=t,start=start,specification_sha256=h,values_sha256=d.sha(a.tobytes()),unavailable_rows=0)
                r.save_npz(r.chunk_path(root,t,start),values=a,metadata=np.array(r.canonical(meta)))
            worker=r.Worker(root,Path(td)/'zip',None,d.ActiveClock(root/'active_time.json',1000))
            with (patch.object(s,'tasks',return_value=small),
                  patch.object(s,'make_inputs',side_effect=AssertionError('No science RNG in local tests')),
                  patch.object(r,'generate_chunk',side_effect=generator)):
                worker.replicate(0);first=list(calls);worker.replicate(0)
            self.assertEqual(calls,first);self.assertEqual(calls[:2],['ref','ref']);self.assertEqual(calls[2:12],['cal']*10)
            self.assertTrue(all(x in ('null','fault','control') for x in calls[12:]))
            meta=d.read(root/'carry/rep_00.json');self.assertTrue(meta['calibration_unchanged']);self.assertEqual(meta['records'],448)
            with r.open_npz(root/'carry/rep_00.npz') as z:self.assertEqual(z['counts'].shape,(76,2,6,8))
    def test_local_main_guard(self):
        with patch.object(r.sys,'platform','darwin'):
            with self.assertRaises(RuntimeError):r.main('/tmp/nonexistent','/tmp/out','test')


class BuildTests(unittest.TestCase):
    def test_onecell_payload_and_manifest(self):
        path=ROOT/'numerics/colab/pr0006_lfdc_budget_b25_one_cell.py';body=path.read_text();tree=ast.parse(body)
        constants={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('_CELL_ID','_PAYLOAD','_PAYLOAD_SHA256')}
        raw=zlib.decompress(base64.b64decode(constants['_PAYLOAD']));self.assertEqual(d.sha(raw),constants['_PAYLOAD_SHA256'])
        manifest=d.read(ROOT/'numerics/colab/pr0006_lfdc_budget_b25_build.json');self.assertEqual(manifest['cell_id'],constants['_CELL_ID'])
        import build_pr0006_colab as builder
        aliases={p:n for n,p in builder.PATHS.items()}
        for p,h in manifest['sources'].items():
            self.assertEqual(d.digest(ROOT/p),h);self.assertEqual(json.loads(raw)['sources'][aliases[p]],(ROOT/p).read_text())
        nb=d.read(path.with_suffix('.ipynb'));self.assertEqual(len(nb['cells']),1);self.assertEqual(''.join(nb['cells'][0]['source']),body)
        self.assertNotIn('drive.mount',body)


if __name__=='__main__':unittest.main(verbosity=2)
