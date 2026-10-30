# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Local technical tests ONLY. No physical RNG, production GPU or scientific run."""
import ast
import base64
import io
import json
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch
import zipfile
import zlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'numerics/scripts'))
import pr0009_spec as s
import pr0009_methods as m
import pr0009_io as d
import pr0009_runner as r
from scipy.stats import binomtest
import pr0009_launcher as l
import pr0009_download as dl
from test_pr0006 import fixture,MethodTests as InheritedMethodTests


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.base=Path(self.temp.name);self.root=self.base/'run'
        self.sources={'never_execute.py':'raise RuntimeError("NOT EXECUTABLE DATA")\n','PR-0009.md':'fixture','PR0009-development.json':'fixture A01'}
        d.new_run(self.root,'testcell',self.sources)
    def tearDown(self):self.temp.cleanup()
    def fake_rep(self,rep):
        folder=d.rep_dir(self.root,rep);d.atomic(folder/'tiny.bin',('fixture '+str(rep)).encode())
        data=self.root/'carry'/f'rep_{rep:02d}.npz';d.atomic(data,b'opaque technical fixture')
        d.write(data.with_suffix('.json'),dict(replicate=rep,records=620784,statistics_sha256=d.digest(data),files={'tiny.bin':d.digest(folder/'tiny.bin')}))
        state=d.read(self.root/'state.json');state['completed'][f'{rep:02d}']=dict(metadata_sha256=d.digest(data.with_suffix('.json')));d.write(self.root/'state.json',state)
    def test_roundtrip_source_not_executed(self):
        z=d.snapshot(self.root,self.base/'zip','PARTIAL',[])[0]
        self.assertEqual(d.restore_many([z],self.base/'new','testcell',self.sources),[])
        with self.assertRaises(FileExistsError):d.restore_many([z],self.base/'new','testcell',self.sources)
    def test_noncumulative_five_and_receipt_only(self):
        for rep in range(5):self.fake_rep(rep)
        first=d.snapshot(self.root,self.base/'zip','SHARD_01_COMPLETE',range(5))[0]
        for rep in range(5,10):self.fake_rep(rep)
        second=d.snapshot(self.root,self.base/'zip','SHARD_02_COMPLETE',range(5,10))[0]
        idx=d.inspect_zip(second,'testcell');self.assertFalse(any(n.startswith('shards/00/') for n in idx['entries']))
        self.assertEqual(d.restore_many([second],self.base/'new','testcell',self.sources),list(range(5)))
        self.assertTrue(first.exists());self.assertEqual(d.verify_carry(self.base/'new'),list(range(5)))
    def test_delta_requires_dependencies_and_dedup(self):
        folder=d.rep_dir(self.root,0);d.atomic(folder/'a.bin',b'first committed chunk')
        first=d.snapshot(self.root,self.base/'zip','PARTIAL',[0])[0]
        d.atomic(folder/'b.bin',b'second committed chunk');second=d.snapshot(self.root,self.base/'zip','PARTIAL',[0])[0]
        self.assertNotIn('shards/00/rep_00/a.bin',d.inspect_zip(second,'testcell')['entries'])
        with self.assertRaisesRegex(ValueError,'Missing ongoing delta'):d.restore_many([second],self.base/'bad','testcell',self.sources)
        self.assertFalse((self.base/'bad').exists())
        d.restore_many([first,second,first],self.base/'good','testcell',self.sources)
        self.assertEqual(len(d.evidence(d.rep_dir(self.base/'good',0))),2)
    def test_final_shard_overlap_no_double_count(self):
        d.atomic(d.rep_dir(self.root,0)/'a.bin',b'same raw')
        a=d.snapshot(self.root,self.base/'zip','PARTIAL',[0])[0]
        b=d.snapshot(self.root,self.base/'zip','SHARD_01_COMPLETE',[0])[0]
        d.restore_many([a,b],self.base/'new','testcell',self.sources)
        self.assertEqual(len(d.evidence(d.rep_dir(self.base/'new',0))),1)
    def test_split_at_rep_and_committed_file(self):
        for i in range(2):
            for j in range(4):d.atomic(d.rep_dir(self.root,i)/f'{j}.bin',bytes([j])*100)
        original=d.snapshot_one
        def gate(root,dest,status,reps,paths):
            raw=[p for p in paths if p.is_relative_to(Path(root)/'shards')]
            if len(raw)>2:raise OverflowError('Mock size gate')
            return original(root,dest,status,reps,paths)
        with patch.object(d,'snapshot_one',side_effect=gate):zs=d.snapshot(self.root,self.base/'zip','PARTIAL',[0,1])
        self.assertEqual(len(zs),4);self.assertTrue(all(z.stat().st_size<d.ZIP_LIMIT for z in zs))
        d.restore_many(zs,self.base/'new','testcell',self.sources)
        self.assertEqual(len(d.evidence(self.base/'new'/'shards')),8)
    def test_actual_size_and_prepack_guards(self):
        with patch.object(d,'PREPACK_LIMIT',1):
            with self.assertRaises(OverflowError):d.snapshot(self.root,self.base/'zip','PARTIAL',[])
        with patch.object(d,'ZIP_LIMIT',50):
            with self.assertRaises(OverflowError):d.snapshot(self.root,self.base/'zip','PARTIAL',[])
        self.assertFalse(list((self.base/'zip').glob('*.zip')))
    def test_cross_shard_unknown_and_zip_slip(self):
        with self.assertRaises(ValueError):d.snapshot(self.root,self.base/'zip','PARTIAL',[4,5])
        z=d.snapshot(self.root,self.base/'zip','PARTIAL',[])[0]
        with zipfile.ZipFile(z,'a') as f:f.writestr('../escape',b'x')
        with self.assertRaises(ValueError):d.inspect_zip(z,'testcell')
    def test_duplicate_and_symlink_rejected(self):
        for mode in ('duplicate','symlink'):
            z=d.snapshot(self.root,self.base/'zip','PARTIAL',[])[0]
            with zipfile.ZipFile(z,'a') as f:
                if mode=='duplicate':f.writestr('state.json',b'{}')
                else:
                    info=zipfile.ZipInfo('sessions/link');info.external_attr=(stat.S_IFLNK|0o777)<<16;f.writestr(info,b'/tmp')
            with self.assertRaises(ValueError):d.inspect_zip(z,'testcell')
    def test_missing_receipt_and_tamper_rejected(self):
        self.fake_rep(0);d.verify_carry(self.root);d.atomic(d.rep_dir(self.root,0)/'tiny.bin',b'changed')
        with self.assertRaises(ValueError):d.verify_carry(self.root)
        shutil.rmtree(d.rep_dir(self.root,0))
        with self.assertRaises(ValueError):d.verify_carry(self.root)
    def test_live_resume_never_prompts(self):
        base=self.base/'base';base.mkdir();root=base/'runs'/'current_testcell';d.new_run(root,'testcell',self.sources)
        with patch.object(l,'upload_streaming',side_effect=AssertionError('must not prompt')):
            actual,mode=l.resolve_run(base,'testcell',self.sources,'UPLOAD')
        self.assertEqual((actual,mode),(root,'existing'))
    def test_mixed_run_restore_refused(self):
        a=d.snapshot(self.root,self.base/'zip','PARTIAL',[])[0]
        other=self.base/'other';d.new_run(other,'testcell',self.sources);b=d.snapshot(other,self.base/'zip','PARTIAL',[])[0]
        with self.assertRaises(ValueError):d.restore_many([a,b],self.base/'new','testcell',self.sources)


class BrowserAndBundleTests(unittest.TestCase):
    def test_download_acknowledgement_and_no_disk_claim(self):
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td);path=folder/'small.zip';path.write_bytes(b'technical bytes')
            class Manager:
                def __init__(self):self.targets={}
                def register_target(self,key,target):self.targets[key]=target
                def unregister_target(self,key,target):del self.targets[key]
            class Comm:
                def on_msg(self,callback):self.callback=callback
                def send(self,data,metadata,buffers):self.sent=buffers
                def close(self):self.closed=True
            manager=Manager();ipy=types.ModuleType('IPython')
            ipy.get_ipython=lambda:types.SimpleNamespace(kernel=types.SimpleNamespace(comm_manager=manager))
            display=types.ModuleType('IPython.display');display.display=lambda x:None;display.Javascript=lambda x:x
            with patch.dict(sys.modules,{'IPython':ipy,'IPython.display':display}):
                delivery=dl.Delivery(folder/'receipts');delivery.enqueue(path)
                target=next(iter(manager.targets.values()));comm=Comm();target(comm,None)
                comm.callback({'content':{'data':{'action':'next'}}})
                self.assertEqual(comm.sent,[b'technical bytes'])
                comm.callback({'content':{'data':dict(action='complete',received_bytes=len(b'technical bytes'),
                    save_requested=True,resources_released=True)}})
                saved=delivery.status()['items'][0]
                self.assertTrue(saved['browser_received']);self.assertTrue(saved['save_requested'])
                self.assertFalse(saved['user_disk_confirmed']);self.assertFalse(manager.targets)
                delivery.enqueue(path);self.assertFalse(manager.targets)

    def test_upload_streams_fragments_and_keeps_multiple_archives(self):
        with tempfile.TemporaryDirectory() as td:
            frames=[{'action':'started'},{'action':'append','file':'a.zip','data':base64.b64encode(b'abc').decode()},
                    {'action':'append','file':'a.zip','data':base64.b64encode(b'def').decode()},
                    {'action':'append','file':'b.zip','data':base64.b64encode(b'123').decode()},{'action':'complete'}]
            colab=types.ModuleType('google.colab');colab.output=types.SimpleNamespace(eval_js=lambda js:frames.pop(0))
            display=types.ModuleType('IPython.display');display.display=lambda x:None;display.HTML=lambda x:x
            with (patch.dict(sys.modules,{'google.colab':colab,'IPython.display':display}),
                  patch.object(dl.pkgutil,'get_data',return_value=b'// trusted mock widget')):
                paths=dl.upload_streaming(td)
            self.assertEqual([p.read_bytes() for p in paths],[b'abcdef',b'123']);self.assertEqual(len(frames),0)

    def test_js_serial_release_and_pause_mock(self):
        harness=r'''
const realTimeout=setTimeout; global.setTimeout=(f,ms)=>realTimeout(f,ms===10000?0:ms);
let openCount=0,maxOpen=0,revoked=0,complete=0,cancel=0,deny=false;
global.document={body:{appendChild(){}},createElement(){return {appendChild(){},click(){}}}};
global.URL={createObjectURL(){return 'mock';},revokeObjectURL(){revoked++;}};
global.google={colab:{kernel:{accessAllowed:true,comms:{async open(id){
  if(deny && id==='bad')throw new Error('mock blocked');
  openCount++;maxOpen=Math.max(openCount,maxOpen);
  let sent=false;
  return {send(data){if(data.action==='complete'){complete++;openCount--;}if(data.action==='cancel'){cancel++;openCount--; }},
    messages:{[Symbol.asyncIterator](){return {async next(){if(sent)return {done:true};sent=true;return {done:false,value:{buffers:[new Uint8Array(4)]}};}}}}};
}}}}};
eval(INPUT);
const q=globalThis.pr0009Queue;
q.enqueue({id:'one',key:'one',name:'one',size:4});q.enqueue({id:'two',key:'two',name:'two',size:4});
async function wait(){for(let i=0;i<100;i++){if(!q.running)break;await new Promise(r=>realTimeout(r,2));}}
(async()=>{await wait();if(maxOpen!==1||revoked!==2||complete!==2||q.active!==0)throw new Error('serial/cleanup mismatch');
deny=true;q.enqueue({id:'bad',key:'bad',name:'bad',size:4});q.enqueue({id:'later',key:'later',name:'later',size:4});
await wait();if(!q.paused||q.items.length!==1||q.running)throw new Error('pause mismatch');
console.log(JSON.stringify({maxActive:q.maxActive,revoked,complete,paused:q.paused,pending:q.items.length}));})().catch(e=>{console.error(e);process.exitCode=1;});
'''
        code='const INPUT='+json.dumps(dl.QUEUE_JS)+';\n'+harness
        proc=subprocess.run(['node','-e',code],capture_output=True,text=True,timeout=15)
        self.assertEqual(proc.returncode,0,proc.stderr);self.assertEqual(json.loads(proc.stdout)['maxActive'],1)

    def test_bundle_sources_not_executed_and_exact(self):
        py=ROOT/'numerics/colab/pr0009_two_axis_a50_one_cell.py';tree=ast.parse(py.read_text());values={}
        for n in tree.body:
            if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('_CELL_ID','_PAYLOAD','_PAYLOAD_SHA256'):
                values[n.targets[0].id]=ast.literal_eval(n.value)
        raw=zlib.decompress(base64.b64decode(values['_PAYLOAD']));self.assertEqual(d.sha(raw),values['_PAYLOAD_SHA256'])
        import build_pr0009_colab as builder
        sources=json.loads(raw)['sources']
        for n,p in builder.PATHS.items():self.assertEqual(sources[n],(ROOT/p).read_text())
        nb=d.read(py.with_suffix('.ipynb'));self.assertEqual(len(nb['cells']),1)
        self.assertEqual(''.join(nb['cells'][0]['source']),py.read_text());self.assertEqual(nb['cells'][0]['outputs'],[])


class PlanTests(unittest.TestCase):
    def test_card_development_and_inherited_sources(self):
        freeze=d.read(ROOT/'audits/verification/2026-10-18_PR-0009_design_freeze.json')
        self.assertEqual(d.digest(ROOT/freeze['card']['path']),s.CARD_SHA)
        for p,h in freeze['inherited_sources'].items():self.assertEqual(d.digest(ROOT/p),h)
        dev=d.read(ROOT/'audits/verification/2026-10-18_PR-0009_development.json')
        self.assertEqual(d.digest(ROOT/dev['source']),dev['source_sha256'])
        with np.load(ROOT/dev['source'],allow_pickle=False) as z:clean=z['scores'][:,:,0,:]
        np.testing.assert_array_equal(clean,np.array(dev['clean_scores']))
        np.testing.assert_array_equal(s.GATE_CUTS,np.stack([clean[:,[0,2]].max((0,1)),clean[:,[1,3,4,5]].max((0,1))]))

    def test_budget_and_ids_no_physical_draws(self):
        tasks=s.tasks(0);self.assertEqual(len(tasks),100);self.assertEqual(sum(x['count'] for x in tasks),620784)
        self.assertEqual(sum(x['count'] for x in tasks)*50,31039200)
        self.assertEqual(len(s.primary()),18);self.assertEqual(s.CAP_MULTIPLIER,1.25)
        addresses=[(s.MASTER,s.conditions()[t['case']]['id'],s.ROLES[t['role']],rep,block)
            for rep in range(50) for t in s.tasks(rep) for block in range((t['count']+63)//64)]
        self.assertEqual(len(addresses),len(set(addresses)))
        self.assertEqual(s.ROLES['validation_replace'],15)
        self.assertEqual((620784*136+8*6*35000*8+6*23000*8)*5,494853120)
        self.assertEqual(len({t['id'] for t in tasks}),len(tasks))

    def test_order_roles_disjoint_and_rechunk_fake_rng(self):
        np.testing.assert_array_equal(np.sort(s.order('B_M','audit_order',0,15999)),np.arange(15999))
        self.assertFalse(np.array_equal(s.order('B_M','audit_replace',0,15999),s.order('B_M','update_replace',0,15999)))
        import pr0009_input as inp
        class Fake:
            def __init__(self,seed):self.value=sum(seed);self.offset=0
            def normal(self,size):
                self.offset+=1;return np.broadcast_to((np.arange(size[0])+self.value+self.offset)[:,None]*1e-7,size).copy()
        with (patch.object(inp.np.random,'SeedSequence',side_effect=lambda x:x),patch.object(inp.np.random,'PCG64',side_effect=lambda x:x),
              patch.object(inp.np.random,'Generator',side_effect=Fake)):
            a=s.make_inputs('K2_M__P02','preflight',0,62,5)
            b=s.make_inputs('K2_M__P02','preflight',0,62,2);c=s.make_inputs('K2_M__P02','preflight',0,64,3)
        for k in a:np.testing.assert_array_equal(a[k],np.concatenate([b[k],c[k]]))

    def test_local_main_disabled(self):
        with patch.object(r.sys,'platform','darwin'):
            with self.assertRaisesRegex(RuntimeError,'Colab'):r.main('/tmp/unused','/tmp/unused','unused')

class FiftyShardTests(unittest.TestCase):
    setUp=TransportTests.setUp
    tearDown=TransportTests.tearDown
    fake_rep=TransportTests.fake_rep
    # Reuse fixture helpers without registering all transport tests twice.
    def test_tenth_shard_and_receipt_only_prefix(self):
        for rep in range(50):
            self.fake_rep(rep)
            if (rep+1)%5==0:
                last=d.snapshot(self.root,self.base/'zip',f'SHARD_{rep//5+1:02d}_COMPLETE',range(rep-4,rep+1))[0]
        idx=d.inspect_zip(last,'testcell')
        self.assertEqual(idx['covered_replicates'],[45,46,47,48,49])
        self.assertFalse(any(n.startswith('shards/00/') for n in idx['entries']))
        remote=d.restore_many([last],self.base/'restored50','testcell',self.sources)
        self.assertEqual(remote,list(range(45)))
        self.assertEqual(len(d.read(self.base/'restored50/state.json')['completed']),50)

class TwoAxisTests(unittest.TestCase):
    def test_exact_cp_boundaries_and_unavailable(self):
        for alpha,ks,ke in [(.01,214,287),(.001,14,38)]:
            for k in [ks,ks+1,ke-1,ke]:
                lo,hi=m.cp(k,20000)
                reflo=binomtest(k,20000,alternative='greater').proportion_ci(confidence_level=.9875).low
                refhi=binomtest(k,20000,alternative='less').proportion_ci(confidence_level=.9875).high
                self.assertAlmostEqual(lo,reflo,places=10);self.assertAlmostEqual(hi,refhi,places=10)
            self.assertEqual(m.evidence(ks,20000,20000,alpha)['label'],'FPR-SUPPORTED')
            self.assertEqual(m.evidence(ks+1,20000,20000,alpha)['label'],'FPR-UNRESOLVED')
            self.assertEqual(m.evidence(ke-1,20000,20000,alpha)['label'],'FPR-UNRESOLVED')
            self.assertEqual(m.evidence(ke,20000,20000,alpha)['label'],'FPR-EXCEEDED')
        self.assertEqual(m.evidence(0,19999,20000,.01)['label'],'UNAVAILABLE')
        self.assertEqual(m.evidence(0,20000,20000,.01,False)['label'],'UNAVAILABLE')
        self.assertEqual(m.cp(0,20000)[0],0);self.assertEqual(m.cp(20000,20000)[1],1)

    def test_gate_strict_sided_unknown(self):
        np.testing.assert_array_equal(m.gate(s.GATE_CUTS[0],0),[0,0,0])
        np.testing.assert_array_equal(m.gate(s.GATE_CUTS[0]+.1,0),[1,1,1])
        np.testing.assert_array_equal(m.gate([np.nan,np.inf,0],0),[2,2,0])
        a=fixture(50);a[:,11]=np.sin(np.arange(50));a[:,12]=-100+np.cos(np.arange(50))
        scores,info=m.batch_scores(a,np.zeros(50,bool),np.ones(50,bool))
        self.assertLess(info['signed_T'][1],-100)
        self.assertAlmostEqual(scores[2],abs(info['signed_T'][0]))
        a[:,11]=0;scores,_=m.batch_scores(a,np.zeros(50,bool),np.ones(50,bool))
        self.assertEqual(m.gate(scores,0)[1],2);self.assertEqual(m.gate(scores,0)[2],2)

    def test_packed_selection_streaming_and_missing(self):
        n=29;bits=np.zeros((n,8,2),np.uint32)
        for j in range(8):
            for ai in range(2):
                for mi in range(6):
                    nib=np.where((np.arange(n)+j+mi)%11==0,0,8|((np.arange(n)+j+ai)%8)).astype(np.uint32)
                    bits[:,j,ai]|=nib<<(4*mi)
        a=m.empty();a['gate_states'][0]=np.arange(21).reshape(7,3)%3;a['same_states'][0]=np.arange(7)%3
        b={k:v.copy() for k,v in a.items()};m.accumulate(a,0,0,bits)
        for z in (bits[:7],bits[7:20],bits[20:]):m.accumulate(b,0,0,z)
        for k in a:np.testing.assert_array_equal(a[k],b[k])
        for ci in range(7):
            sel=m.selections(ci,a['gate_states'][0,ci],a['same_states'][0,ci])
            for gi,j in enumerate(sel[2:]):
                for ri,o in enumerate([ci+1,0,1]):
                    pa,va=m.decode(bits[:,j]);pb,vb=m.decode(bits[:,o]);valid=va&vb
                    reference=np.stack([np.sum(valid&x,axis=0) for x in [(pa>0)&(pb>0),(pa>0)&(pb==0),(pa==0)&(pb>0),(pa==0)&(pb==0)]],axis=-1)
                    np.testing.assert_array_equal(a['paired'][0,ci,0,gi,ri],reference)
        m.validate(a)

    def test_aggregate_shape_missing_and_zero_sd(self):
        rows=[]
        for i in range(50):
            a=m.empty();a['gate_states'][:]=0;a['same_states'][:]=0
            for e,n in enumerate([20000,5000,5000,5000]):
                a['counts'][:,e,:,...,0]=n;a['available'][:,e]=n;a['paired'][:,:,e,...,3]=n
            a['v_counts'][...,1]=20000;a['v_counts'][...,0]=200
            rows.append(a)
        out=m.aggregate(rows)
        self.assertEqual(len(out['primary18']),18);self.assertEqual(len(out['policy_results']),1008)
        self.assertTrue(all(p['interval']['lower'] is None for p in out['primary18']))
        self.assertEqual(len(out['gate_results']),42);self.assertEqual(len(out['validation_evidence']),3000)
        self.assertEqual(len(out['clean_gate_summary']),18)
        clean_mc=next(x for x in out['clean_gate_summary'] if x['gate']=='G-MC')
        self.assertAlmostEqual(clean_mc['upper'],1-(.05/6)**(1/50))
        r.canonical(r.json_safe(out))

class WorkerTests(unittest.TestCase):
    def test_atomic_crash_resume_roles_and_v_no_intervention(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);sources={'PR-0009.md':'fixture','PR0009-development.json':'fixture'}
            counts={k:(39 if k=='ref' else 31 if k.endswith('normal') or k=='old_cal' else 4 if k.endswith('fault') else 30 if k=='null_eval' else 15) for k in s.COUNTS}
            cand=(('clean',0),('GM05',2),('GM05',4),('GP05',2),('GP05',4),('P02',2),('P02',4))
            frozen_primary=s.primary()
            generated=[]
            class Clock:
                def check(self):pass
                def commit(self):pass
            def fake(root,t,start,sh,gpu):
                if t['role'].startswith('validation') or t['role'].endswith('_eval'):
                    self.assertTrue((d.rep_dir(root,t['replicate'])/'CALIBRATION_FROZEN.json').exists())
                generated.append((str(root),t['id'],start));a=fixture(min(s.CHUNK,t['count']-start),start+100*s.ROLES[t['role']]+1000*s.conditions()[t['case']]['id'])
                meta=dict(task=t,start=start,count=len(a),specification_sha256=sh,values_sha256=d.sha(a.tobytes()),unavailable_rows=0)
                r.save_npz(r.chunk_path(root,t,start),values=a,metadata=np.array(r.canonical(meta)))
            original=r.save_npz;stopped=[False]
            def crashing(path,**arrays):
                original(path,**arrays)
                if 'validation' in Path(path).parts and 'packed' in arrays and not stopped[0]:
                    stopped[0]=True;raise RuntimeError('Injected after V atomic commit')
            with (patch.object(s,'COUNTS',counts),patch.object(s,'CANDIDATES',cand),patch.object(s,'CHUNK',16),
                  patch.object(s,'primary',return_value=frozen_primary),
                  patch.object(r,'generate_chunk',side_effect=fake),patch.object(s,'make_inputs',side_effect=AssertionError('No physical generation')),
                  patch.object(m,'evidence',side_effect=AssertionError('V evidence must not intervene in worker E decisions')),
                  patch('builtins.print')):
                interrupted=base/'interrupted';fresh=base/'fresh'
                for root in (interrupted,fresh):d.new_run(root,'fixture',sources)
                worker=r.Worker(interrupted,base/'archives',None,Clock())
                with patch.object(r,'save_npz',side_effect=crashing):
                    with self.assertRaisesRegex(RuntimeError,'Injected'):worker.replicate(0)
                before={p:d.digest(p) for p in d.evidence(d.rep_dir(interrupted,0))}
                worker.replicate(0);r.Worker(fresh,base/'archives',None,Clock()).replicate(0)
                for p,h in before.items():self.assertEqual(d.digest(p),h)
                self.assertEqual(len(generated),len(set(generated)))
                with r.open_npz(interrupted/'carry/rep_00.npz') as a,r.open_npz(fresh/'carry/rep_00.npz') as b:
                    for k in a.files:np.testing.assert_array_equal(a[k],b[k])
                folder=d.rep_dir(interrupted,0)
                for role in ('audit','update'):
                    with r.open_npz(folder/role/'B_M'/'candidate_2.npz') as z:
                        p=json.loads(str(z['provenance']));self.assertIn(role,p['normal_task']);self.assertIn(role,p['fault_task'])
                        self.assertEqual(int(z['origin'].sum()),4)
                self.assertEqual(d.read(interrupted/'carry/rep_00.json')['records'],sum(t['count'] for t in s.tasks(0)))

if __name__=='__main__':
    started=time.monotonic();suite=unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    output=stream.getvalue();print(output)
    report=dict(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        seconds=time.monotonic()-started,scientific_records_generated=0,scope='Synthetic sufficient-statistic fixtures / fake RNG / browser mocks only',
        actual_gpu_tested=False,actual_colab_browser_tested=False,python=sys.version,numpy=np.__version__)
    if len(sys.argv)>1:
        destination=Path(sys.argv[1]);destination.mkdir(parents=True,exist_ok=False)
        d.atomic(destination/'test.log',output.encode());d.write(destination/'results.json',report)
    print(json.dumps(report));sys.exit(0 if result.wasSuccessful() else 1)
