# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Technical artificial-statistic fixtures only. Never science-master physical draws."""
import ast,base64,io,json,sys,tempfile,time,types,unittest,zlib
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.stats import binom,binomtest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'numerics/scripts'))
import pr0012_spec as s
import pr0012_methods as m
import pr0012_runner as r
import pr0012_io as d
import pr0012_launcher as l
import pr0012_download as dl
import pr0006_methods as oldm
import test_pr0009 as prior
from test_pr0006 import fixture
prior.d,prior.l,prior.dl=d,l,dl

class TransportTests(prior.TransportTests):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.base=Path(self.temp.name);self.root=self.base/'run'
        self.sources={'never_execute.py':'raise RuntimeError("DO NOT EXECUTE ARCHIVE")','PR-0012.md':'fixture','PR0012-development.json':'fixture'}
        d.new_run(self.root,'testcell',self.sources)
    def fake_rep(self,rep):
        folder=d.rep_dir(self.root,rep);d.atomic(folder/'tiny.bin',('fixture '+str(rep)).encode())
        data=self.root/'carry'/f'rep_{rep:02d}.npz';d.atomic(data,b'opaque fixture')
        d.write(data.with_suffix('.json'),dict(replicate=rep,records=479984,statistics_sha256=d.digest(data),files={'tiny.bin':d.digest(folder/'tiny.bin')}))
        state=d.read(self.root/'state.json');state['completed'][f'{rep:02d}']=dict(metadata_sha256=d.digest(data.with_suffix('.json')));d.write(self.root/'state.json',state)
    def test_tenth_shard(self):prior.FiftyShardTests.test_tenth_shard_and_receipt_only_prefix(self)
    def test_fresh_no_picker(self):
        base=self.base/'fresh';base.mkdir()
        with patch.object(l.shutil,'disk_usage',return_value=types.SimpleNamespace(free=100*1024**3)),patch.object(l,'upload_streaming',side_effect=AssertionError('No picker')):
            root,mode=l.resolve_run(base,'testcell',self.sources,content=self.base/'empty')
        self.assertEqual(mode,'fresh');self.assertEqual(d.check_run(root,'testcell',self.sources)['experiment'],s.EXPERIMENT)
    def test_complete_noop_no_compute_or_pack(self):
        base=self.base/'base';base.mkdir();root=base/'runs'/'current_testcell';d.new_run(root,'testcell',self.sources)
        self.root=root
        for rep in range(50):self.fake_rep(rep)
        state=d.read(root/'state.json');state['status']='COMPLETE';d.write(root/'state.json',state)
        d.write(root/'catalog.json',dict(archives=[dict(status='COMPLETE_INDEX',name='missing.zip')]))
        before={p:d.digest(p) for p in d.evidence(root)}
        original=Path.is_dir
        with (patch.object(l.sys,'platform','linux'),patch.object(l.importlib,'import_module',return_value=types.SimpleNamespace()),
              patch.object(Path,'is_dir',lambda p:True if str(p)=='/content' else original(p)),
              patch.object(l,'resolve_run',return_value=(root,'existing')),patch.object(l,'ensure_environment',side_effect=AssertionError('No bootstrap')),
              patch.object(d,'snapshot',side_effect=AssertionError('No packing')),
              patch.object(Path,'mkdir',lambda *a,**kw:None),patch.object(l,'Delivery',return_value=types.SimpleNamespace())):
            # lock needs a real parent; redirect the /content base to the fixture.
            from contextlib import nullcontext
            with patch.object(d,'lock',return_value=nullcontext()):out=l.run(self.sources,'testcell',self.base,auto_download=False)
        self.assertEqual(out['status'],'COMPLETE_NOOP')
        self.assertEqual(before,{p:d.digest(p) for p in d.evidence(root)})

class BrowserTests(prior.BrowserAndBundleTests):
    def test_js_serial_release_and_pause_mock(self):
        with patch.object(dl,'QUEUE_JS',dl.QUEUE_JS+'\nglobalThis.pr0009Queue=globalThis.pr0012Queue;'):
            super().test_js_serial_release_and_pause_mock()
    def test_bundle_sources_not_executed_and_exact(self):
        import build_pr0012_colab as builder
        py=ROOT/'numerics/colab/pr0012_mixture_recovery_b50_one_cell.py';v={}
        for n in ast.parse(py.read_text()).body:
            if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('_PAYLOAD','_PAYLOAD_SHA256'):
                v[n.targets[0].id]=ast.literal_eval(n.value)
        raw=zlib.decompress(base64.b64decode(v['_PAYLOAD']));self.assertEqual(d.sha(raw),v['_PAYLOAD_SHA256'])
        sources=json.loads(raw)['sources']
        for n,p in builder.PATHS.items():self.assertEqual(sources[n],(ROOT/p).read_text())
        nb=d.read(py.with_suffix('.ipynb'));self.assertEqual(len(nb['cells']),1)
        self.assertEqual(''.join(nb['cells'][0]['source']),py.read_text());self.assertFalse(nb['cells'][0]['outputs'])

class PlanTests(unittest.TestCase):
    def test_frozen_sources_and_plan(self):
        f=d.read(ROOT/'audits/verification/2026-10-18_PR-0012_design_freeze.json')
        self.assertEqual(d.digest(ROOT/f['card']['path']),s.CARD_SHA)
        for p,h in f['inherited_sources'].items():self.assertEqual(d.digest(ROOT/p),h)
        self.assertEqual(len(s.conditions()),17);self.assertEqual(len(s.primary()),13)
        self.assertEqual(sum(t['count'] for t in s.tasks(0)),479984)
        self.assertEqual(479984*50,23999200)
        self.assertEqual((479984*136+4*8*2*35000+4*3*2*35000+4*2*2*20000)*5,343389120)
        addresses=[(s.MASTER,s.conditions()[t['case']]['id'],s.ROLES[t['role']],rep,b) for rep in range(50) for t in s.tasks(rep) for b in range((t['count']+63)//64)]
        self.assertEqual(len(addresses),len(set(addresses)))
        self.assertNotEqual(s.ROLES['recovery'],s.ROLES['update_normal'])
        import pr0009_spec
        np.testing.assert_array_equal(s.GATE_CUTS,pr0009_spec.GATE_CUTS[:,1:])
    def test_incomplete_complete_flag_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            d.write(Path(td)/'state.json',dict(completed={},status='COMPLETE'))
            d.write(Path(td)/'catalog.json',dict(archives=[]))
            with self.assertRaisesRegex(ValueError,'all 50'):d.verify_carry(td)
    def test_recipe_pure_addition(self):
        normal=fixture(15999);faults={f:fixture(n,100+10*i) for i,(f,n) in enumerate(zip(s.FAULTS,(800,800,400)))}
        order=s.order('B_L','audit_order',0,15999);pos=s.order('B_L','audit_replace',0,15999)
        out={c:r.mixture(normal,faults,order,pos,c) for c in s.CANDIDATES}
        for c,counts in zip(s.CANDIDATES,((0,0,0),(800,0,0),(0,800,0),(400,400,0),(200,200,0),(200,200,400))):
            self.assertEqual(tuple(int((out[c][2]==i).sum()) for i in (1,2,3)),counts)
        a,b=out['BAL400'],out['TRI800'];gain=(a[2]>0)
        np.testing.assert_array_equal(a[0][gain],b[0][gain]);np.testing.assert_array_equal(a[1][gain],b[1][gain])
        same=b[2]!=3;np.testing.assert_array_equal(a[0][same],b[0][same])
        self.assertFalse(np.array_equal(order,s.order('B_L','update_order',0,15999)))
    def test_rng_address_and_rechunk_fake_only(self):
        import pr0012_input as inp
        seen=[]
        class Fake:
            def __init__(self,seed):seen.append(seed);self.value=sum(seed);self.step=0
            def normal(self,size):
                self.step+=1;return np.broadcast_to((np.arange(size[0])+self.value+self.step)[:,None]*1e-7,size).copy()
        with patch.object(inp.np.random,'SeedSequence',side_effect=lambda x:x),patch.object(inp.np.random,'PCG64',side_effect=lambda x:x),patch.object(inp.np.random,'Generator',side_effect=Fake):
            a=s.make_inputs('K2_M__P02','preflight',0,62,5)
            b=s.make_inputs('K2_M__P02','preflight',0,62,2);c=s.make_inputs('K2_M__P02','preflight',0,64,3)
        for k in a:np.testing.assert_array_equal(a[k],np.concatenate((b[k],c[k])))
        self.assertTrue(all(x[0]==s.PREFLIGHT_MASTER for x in seen))
    def test_local_science_disabled(self):
        with patch.object(r.sys,'platform','darwin'):
            with self.assertRaises(RuntimeError):r.main('/tmp/unused','/tmp/unused','none')
            with self.assertRaises(RuntimeError):r.generate_chunk(None,None,None,None,None)

class MethodTests(unittest.TestCase):
    def test_fixed_readouts_match_old_and_halfplane(self):
        state,ordered=m.fit(fixture(9999));cal=fixture(15999,104);values=fixture(500,10101)
        meta,cut,idx=m.calibrate(state,ordered,cal);om,oc,oi=oldm.calibrate(state,ordered,cal)
        np.testing.assert_array_equal(cut,oc[:,[0,5]])
        np.testing.assert_array_equal(idx,oi[:,[0,5]])
        prior_bits=oldm.decide(state,ordered,om,oc,values)
        for method,oldidx in ((0,0),(1,5)):
            expected=((prior_bits>>(4*oldidx))&15).astype(np.uint8)
            np.testing.assert_array_equal(m.decide(state,ordered,meta,cut,values,method),expected)
        bits=np.stack([m.decide(state,ordered,meta,cut,values,1)]*3,axis=1)
        audit=r.readout_audit((state,ordered),[(meta,cut)]*3,values,bits)
        self.assertEqual(audit['rank_halfplane_mismatches'],0);self.assertEqual(audit['rank'],2)
        self.assertLess(audit['max_reconstruction_error'],1e-12)
    def test_invalid_calibration_no_silent_exclusion(self):
        a=fixture(99);state,ordered=m.fit(a);a[3,11]=np.nan
        meta,cut,_=m.calibrate(state,ordered,a);self.assertFalse(meta['valid'])
        self.assertTrue((m.decide(state,ordered,meta,cut,fixture(3))==0).all())
        states,_=m.gates(a,0);np.testing.assert_array_equal(states,[2,2])
    def test_gate_sidedness_strict_zero_sd(self):
        a=fixture(50);a[:,11]=np.sin(np.arange(50));a[:,12]=-100+np.cos(np.arange(50))
        _,desc=m.gates(a,0);self.assertEqual(desc['scores'][1],abs(desc['signed_T'][0]))
        with patch.object(s,'GATE_CUTS',np.array([desc['scores'],desc['scores']])):
            np.testing.assert_array_equal(m.gates(a,0)[0],[0,0])
        a[:,11]=0;np.testing.assert_array_equal(m.gates(a,0)[0],[2,2])
    def test_cp_boundaries_and_recovery_threshold(self):
        for alpha,ks,ke in ((.01,211,291),(.001,12,39)):
            self.assertEqual(m.evidence(ks,20000,20000,alpha)['label'],'SUPPORTED')
            self.assertEqual(m.evidence(ks+1,20000,20000,alpha)['label'],'UNRESOLVED')
            self.assertEqual(m.evidence(ke-1,20000,20000,alpha)['label'],'UNRESOLVED')
            self.assertEqual(m.evidence(ke,20000,20000,alpha)['label'],'EXCEEDED')
            for k in (ks,ks+1,ke-1,ke):
                lo,hi=m.cp(k,20000,.00625)
                ref=binomtest(k,20000).proportion_ci(confidence_level=.9875)
                self.assertAlmostEqual(lo,ref.low,places=9);self.assertAlmostEqual(hi,ref.high,places=9)
        self.assertLessEqual(m.cp(35,50,.05/26)[0],.5);self.assertGreater(m.cp(36,50,.05/26)[0],.5)
        self.assertEqual(m.evidence(0,19999,20000,.01)['label'],'UNAVAILABLE')
    def test_all_policy_cases_and_unknown(self):
        for f in ('SUPPORTED','EXCEEDED','UNRESOLVED','UNAVAILABLE'):
            for c in ('SUPPORTED','EXCEEDED','UNRESOLVED','UNAVAILABLE'):
                np.testing.assert_array_equal(m.select(0,3,f,c),[4,4,4])
                np.testing.assert_array_equal(m.select(2,3,f,c),[-2,-2,-2])
                expected=[0,0,0] if f=='SUPPORTED' else [0,-1,7 if c=='SUPPORTED' else -1]
                np.testing.assert_array_equal(m.select(1,3,f,c),expected)
    def test_streaming_and_scalar_state_table(self):
        n=43;w=(8+np.arange(n*8*2).reshape(n,8,2)%8).astype(np.uint8);w[::7,3]=0;lbits=w[:,[0,1,7]]
        a=m.empty();a['selection'][0,:,0]=[0,-1,7];a['selection'][0,:,1]=[6,-2,5]
        b={k:v.copy() for k,v in a.items()};m.accumulate(a,0,0,w,lbits)
        for lo,hi in ((0,9),(9,25),(25,n)):m.accumulate(b,0,0,w[lo:hi],lbits[lo:hi])
        for k in a:np.testing.assert_array_equal(a[k],b[k])
        m.validate(a,False)
        for ai in range(2):
            table=np.zeros((4,4),np.int64)
            for i in range(n):
                states=[]
                for idx in a['selection'][0,0,ai]:
                    states.append(2 if idx==-1 else 3 if idx==-2 or w[i,idx,ai]==0 else 0 if w[i,idx,ai]&7 else 1)
                table[states[2],states[0]]+=1
            np.testing.assert_array_equal(table,a['pairs'][0,0,0,ai,1])
    def test_primary13_zero_sd_and_invalid(self):
        rows=[]
        for rep in range(50):
            a=m.empty();a['gates'][:]=0;a['selection'][:]=0
            for e,n in enumerate((20000,5000,5000,5000)):
                a['wcounts'][:,e,...,0]=n;a['lcounts'][:,e,...,0]=n;a['lf_pairs'][:,e,...,3]=n
                a['policy'][:,:,e,...,1]=n;a['pairs'][:,:,e,...,1,1]=n
            a['v_counts'][...,1]=20000;rows.append(a)
        result=m.aggregate(rows);self.assertEqual(len(result['primary13']),13)
        self.assertTrue(all(p['classification']=='UNRESOLVED' for p in result['primary13'] if p['id']!='R01'))
        self.assertEqual(result['primary13'][8]['classification'],'OPPOSITE')
        self.assertLess(m.rate_interval(np.zeros(50))['lower'],0)
        self.assertEqual(m.binary_interval([2]*50,[0]*50)['status'],'TECHNICAL_UNESTABLISHED')
        r.canonical(r.safe(result))
    def test_primary13_nonzero_scalar_reference(self):
        rows=[]
        for rep in range(50):
            a=m.empty();a['gates'][:]=0;a['selection'][:]=0
            a['gates'][0,1:3,0]=1;a['gates'][0,3,1]=1;a['gates'][2,5,1]=1
            a['selection'][1,5,0]=[0,-1,7 if rep<40 else -1]
            for e,n in enumerate((20000,5000,5000,5000)):
                a['wcounts'][:,e,...,0]=n;a['lcounts'][:,e,...,0]=n;a['lf_pairs'][:,e,...,3]=n
                a['policy'][:,:,e,...,1]=n;a['pairs'][:,:,e,...,1,1]=n
                for c,k in ((5,100+rep),(6,90)):
                    a['wcounts'][2,e,c,0,0]=n-k;a['wcounts'][2,e,c,0,1]=k
                # Independent scalar record-state construction for the recovery cell.
                p0=np.where(np.arange(n)<250+rep,0,1);p1=np.full(n,2)
                p2=np.where(np.arange(n)<200,0,1) if rep<40 else np.full(n,2)
                states=(p0,p1,p2)
                for pi,x in enumerate(states):a['policy'][1,5,e,0,pi]=np.bincount(x,minlength=4)
                for j,(left,right) in enumerate(((1,0),(2,0),(2,1))):
                    a['pairs'][1,5,e,0,j]=np.bincount(4*states[left]+states[right],minlength=16).reshape(4,4)
            a['v_counts'][...,1]=20000;rows.append(a)
        out=m.aggregate(rows);p=out['primary13'];self.assertEqual([x['id'] for x in p],[f'M{i:02}' for i in range(1,9)]+[f'R{i:02}' for i in range(1,6)])
        self.assertEqual([x['interval']['mean'] for x in p[:4]],[-1.,-1.,1.,1.])
        for e,n in enumerate((20000,5000,5000,5000)):
            self.assertAlmostEqual(p[4+e]['interval']['mean'],np.mean(-(10+np.arange(50))/n))
            delta=(np.where(np.arange(50)<40,200,0)-(250+np.arange(50)))/n
            self.assertAlmostEqual(p[9+e]['interval']['mean'],delta.mean())
        self.assertEqual(p[8]['interval']['k'],40);self.assertTrue(all(x['classification']=='SUPPORTED' for x in p))
        self.assertEqual(len(out['policy_results']),576);self.assertEqual(len(out['calibration_results']),352)

class WorkerTests(unittest.TestCase):
    def test_full_plan_artificial_atomic_resume_and_role_boundary(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);sources={'PR-0012.md':'fixture','PR0012-development.json':'fixture'};generated=[]
            class Clock:
                def check(self):pass
                def commit(self):pass
            def fake(root,t,start,sh,gpu):
                folder=d.rep_dir(root,t['replicate'])
                if t['role']=='validation_normal':self.assertTrue((folder/'CALIBRATION_FROZEN.json').exists())
                if t['role'].endswith('_eval'):self.assertTrue((folder/'SELECTION_BEFORE_E.json').exists())
                generated.append((str(root),t['id'],start))
                a=fixture(min(s.CHUNK,t['count']-start),start+100*s.ROLES[t['role']]+1000*s.conditions()[t['case']]['id'])
                meta=dict(task=t,start=start,count=len(a),specification_sha256=sh,values_sha256=d.sha(a.tobytes()),unavailable_rows=0)
                r.save_npz(r.chunk_path(root,t,start),values=a,metadata=np.array(r.canonical(meta)))
            original=r.save_npz;stopped=[False]
            def crash(path,**arrays):
                original(path,**arrays)
                if 'validation' in Path(path).parts and 'bits' in arrays and not stopped[0]:
                    stopped[0]=True;raise RuntimeError('Injected after V commit')
            with patch.object(r,'generate_chunk',side_effect=fake),patch.object(s,'make_inputs',side_effect=AssertionError('No physical generation')),patch('builtins.print'):
                interrupted=base/'interrupted';fresh=base/'fresh'
                for root in (interrupted,fresh):d.new_run(root,'fixture',sources)
                worker=r.Worker(interrupted,base/'archives',None,Clock(),'technical')
                with patch.object(r,'save_npz',side_effect=crash):
                    with self.assertRaisesRegex(RuntimeError,'Injected'):worker.replicate(0)
                before={p:d.digest(p) for p in d.evidence(d.rep_dir(interrupted,0))}
                worker.replicate(0);r.Worker(fresh,base/'archives',None,Clock(),'technical').replicate(0)
                for p,h in before.items():self.assertEqual(d.digest(p),h)
                self.assertEqual(len(generated),len(set(generated)))
                with r.open_npz(interrupted/'carry/rep_00.npz') as a,r.open_npz(fresh/'carry/rep_00.npz') as b:
                    for k in a.files:np.testing.assert_array_equal(a[k],b[k])
                self.assertEqual(d.verify_carry(interrupted),[])
                folder=d.rep_dir(interrupted,0)
                for g in s.GROUPS:
                    with r.open_npz(folder/'calibration'/g/'C.npz') as a,r.open_npz(folder/'calibration'/g/'clean.npz') as b:
                        self.assertNotEqual(str(a['input_sha256']),str(b['input_sha256']))
                self.assertFalse(d.read(folder/'SELECTION_BEFORE_E.json')['E_used'])

if __name__=='__main__':
    started=time.monotonic();suite=unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]);stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite);print(stream.getvalue())
    report=dict(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
      seconds=time.monotonic()-started,scientific_records_generated=0,artificial_statistics_used=True,
      actual_gpu_tested=False,actual_colab_browser_tested=False,python=sys.version,numpy=np.__version__)
    if len(sys.argv)>1:
        out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False);d.atomic(out/'test.log',stream.getvalue().encode());d.write(out/'results.json',report)
    print(json.dumps(report));sys.exit(0 if result.wasSuccessful() else 1)
