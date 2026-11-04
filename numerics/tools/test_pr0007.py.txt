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
import pr0007_spec as s
import pr0007_methods as m
import pr0007_io as d
import pr0007_runner as r
import pr0007_launcher as l
import pr0007_download as dl
from test_pr0006 import fixture,MethodTests as InheritedMethodTests


class PlanTests(unittest.TestCase):
    def test_frozen_cards(self):
        self.assertEqual(d.digest(ROOT/'audits/prereg/PR-0007_lfdc_calibration_contamination.md'),s.CARD_SHA)
        self.assertEqual(d.digest(ROOT/'audits/prereg/PR-0007_A01_five_replicates_per_zip.md'),s.A01_SHA)
        f=d.read(ROOT/'audits/verification/2026-10-18_PR-0007_design_freeze.json')
        self.assertEqual(f['card']['sha256'],s.CARD_SHA)
        for path,expected in f['inherited_sources'].items():self.assertEqual(d.digest(ROOT/path),expected)
        self.assertEqual(d.digest(ROOT/'numerics/scripts/probe_backend_check_v1.py'),s.inherited.CORE_SHA)

    def test_counts_and_storage(self):
        self.assertEqual((len(s.conditions()),len(s.tasks(0))), (24,50))
        self.assertEqual(sum(t['count'] for t in s.tasks(0)),436392)
        self.assertEqual((len(s.calibrations()),len({x['fit_key'] for x in s.calibrations()}),len(s.contexts())),(66,54,264))
        self.assertEqual(len(s.contrasts()),312)
        self.assertEqual(sum(t['count'] for t in s.tasks(0) if t['role'].endswith('_eval'))*25,5250000)
        records=sum(s.COUNTS['null_eval' if s.conditions()[c['case']]['kind']=='normal' else 'fault_eval'] for c in s.contexts())
        self.assertEqual(records*25*12,693000000);self.assertEqual(records*8,18480000)
        self.assertEqual((436392*17*8+records*8)*5,389146560)
        self.assertEqual([list(range(i,i+5)) for i in range(0,25,5)],[[0,1,2,3,4],[5,6,7,8,9],[10,11,12,13,14],[15,16,17,18,19],[20,21,22,23,24]])

    def test_addresses_without_physical_draws(self):
        a=[(s.MASTER,s.conditions()[t['case']]['id'],s.ROLES[t['role']],rep,b)
           for rep in range(25) for t in s.tasks(rep) for b in range((t['count']+63)//64)]
        self.assertEqual(len(a),len(set(a)));self.assertNotEqual(s.MASTER,s.PREFLIGHT_MASTER)
        self.assertNotEqual(s.MASTER,s.inherited.MASTER)
        for g in s.groups():
            gid=s.conditions()[g]['id']
            for j,f in enumerate(s.FAULTS):self.assertEqual(s.conditions()[g+'__'+f]['id'],6+3*gid+j)

    def test_generator_draw_order_rechunk_fake_rng(self):
        calls=[]
        class Fake:
            def __init__(self,seed):self.offset=sum(seed);self.counter=0
            def normal(self,size):
                self.counter+=1
                return np.broadcast_to((np.arange(size[0])+self.offset+self.counter)[:,None]*1e-7,size).copy()
        def seed(x):calls.append(tuple(x));return x
        with (patch.object(s.np.random,'SeedSequence',side_effect=seed),patch.object(s.np.random,'PCG64',side_effect=lambda x:x),
              patch.object(s.np.random,'Generator',side_effect=Fake)):
            a=s.make_inputs('K2_M__P02','preflight',2,62,5)
            b=s.make_inputs('K2_M__P02','preflight',2,62,2);c=s.make_inputs('K2_M__P02','preflight',2,64,3)
        for k in a:np.testing.assert_array_equal(a[k],np.concatenate((b[k],c[k])))
        self.assertTrue(all(a[0]==s.PREFLIGHT_MASTER for a in calls))
        self.assertEqual(list(a),['initial','source','private_initial','private_source','ni','nj','ci','cj'])

    def test_permutations_fixed_nested_independent_of_scores(self):
        # Permutations are technical design objects, not generated scientific records.
        a,b=s.orders('B_M',0);aa,bb=s.orders('B_M',0)
        np.testing.assert_array_equal(a,aa);np.testing.assert_array_equal(b,bb)
        np.testing.assert_array_equal(np.sort(a),np.arange(31999));np.testing.assert_array_equal(np.sort(b),np.arange(15999))
        self.assertTrue(set(b[:160])<=set(b[:800]));self.assertFalse(np.array_equal(a,s.orders('B_M',1)[0]))

    def test_exact_rank_budgets_all_sizes(self):
        for size in s.SIZES:
            for den in s.ALPHA_DENOMINATORS:
                for w in s.BUDGETS.values():
                    tails=[size+1-m.order_index(size,den,e) for e in w]
                    self.assertEqual(tails,[(size+1)*e//(8*den) for e in w])
                    self.assertEqual(sum(tails),(size+1)//den)

    def test_calibration_contexts_and_primary(self):
        cs=s.contexts();lookup={(c['case'],c['id']):c for c in cs}
        self.assertEqual(len(lookup),264)
        for c in s.calibrations():
            if c['policy']=='F':self.assertEqual(c['fit_key'],s.clean_id(s.baseline(c['group']),c['m']))
        self.assertEqual(len(s.primary()),8)
        self.assertEqual(sum('__' not in c['case'] for c in cs)*12,792)
        self.assertEqual(sum(c['policy']=='C-mix' and '__' in c['case'] for c in cs)*12,1296)

    def test_local_scientific_main_rejected(self):
        with self.assertRaises(RuntimeError):r.main('/tmp/not-a-run','/tmp/not-archives','no')
        with self.assertRaises(RuntimeError):l.run({},'no','/tmp')


class NewMethodTests(unittest.TestCase):
    def test_paired_oracle_and_unavailable(self):
        a=np.zeros((9,2),np.uint32);b=a.copy()
        for mi in range(6):
            a|=np.where(np.arange(9)==8,0,8+np.arange(9)%8).astype(np.uint32)[:,None]<<(mi*4)
            b|=np.where(np.arange(9)==7,0,8+(np.arange(9)+1)%8).astype(np.uint32)[:,None]<<(mi*4)
        cells,n=m.paired(a,b);self.assertTrue((n==7).all());self.assertTrue((cells.sum(-1)==n).all())
        self.assertEqual(cells[0,0].tolist(),[6,0,1,0])

    def test_all_size_thresholds_against_sort(self):
        state,ordered=m.fit(fixture(9999));values=fixture(31999,10000)
        for size in s.SIZES:
            meta,cut,idx=m.calibrate(state,ordered,values[:size]);scores,_=m.signed(state,values[:size])
            z=np.column_stack((scores[:,0],-scores[:,0],scores[:,1]))
            for ai,den in enumerate(s.ALPHA_DENOMINATORS):
                for wi,w in enumerate(s.BUDGETS.values()):
                    for b in range(3):self.assertEqual(cut[ai,wi,b],np.sort(z[:,b])[m.order_index(size,den,w[b])-1])

    @staticmethod
    def results():
        rows=[]
        for rep in range(25):
            a=m.empty();a['inactive'][:]=0
            for c in s.contexts():
                k=c['index'];n=s.COUNTS['null_eval' if '__' not in c['case'] else 'fault_eval']
                a['counts'][k,:,:,0]=n-rep;a['counts'][k,:,:,1]=rep;a['available'][k]=n
                a['pairs'][k,:,:,0]=rep;a['pairs'][k,:,:,3]=n-rep;a['pair_n'][k]=n
            for c in s.contrasts():
                k=c['index'];n=s.COUNTS['null_eval' if '__' not in c['case'] else 'fault_eval']
                a['contrasts'][k,:,:,0]=rep;a['contrasts'][k,:,:,3]=n-rep;a['contrast_n'][k]=n
            r.validate_result(a);rows.append(dict(replicate=rep,**a))
        return rows

    def test_family_sizes_zero_sd_and_missing_not_passed(self):
        rows=self.results();out=m.aggregate(rows)
        self.assertEqual((len(out['primary8']),len(out['fpr792']),len(out['cross1296'])),(8,792,1296))
        self.assertTrue(all(q['interval']['lower'] is None for q in out['primary8']))
        self.assertTrue(all(q['classification']=='unresolved' for q in out['primary8']))
        ctx=next(c for c in s.contexts() if c['case']=='B_L' and c['id']==s.clean_id('B_L',15999))
        rows[0]['available'][ctx['index'],0,0]-=1
        c=next(q for q in s.contrasts() if q['case']=='B_L__GM05' and s.contexts()[q['left']]['id']==s.mix_id('B_L','GM05',160))
        rows[0]['contrast_n'][c['index'],0,0]-=1
        out=m.aggregate(rows)
        f=next(q for q in out['fpr792'] if q['context']['index']==ctx['index'] and q['method']=='W0' and q['alpha']==.01)
        self.assertIsNone(f['interval']['lower']);self.assertFalse(f['excess_flag'])
        self.assertIsNone(out['primary8'][0]['interval']['upper'])
        self.assertEqual(len(out['rates'][0]['rate']['values']),25)


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.base=Path(self.temp.name);self.root=self.base/'run'
        self.sources={'never_execute.py':'raise RuntimeError("NOT EXECUTABLE DATA")\n','PR-0007.md':'fixture','PR-0007-A01.md':'fixture A01'}
        d.new_run(self.root,'testcell',self.sources)
    def tearDown(self):self.temp.cleanup()
    def fake_rep(self,rep):
        folder=d.rep_dir(self.root,rep);d.atomic(folder/'tiny.bin',('fixture '+str(rep)).encode())
        data=self.root/'carry'/f'rep_{rep:02d}.npz';d.atomic(data,b'opaque technical fixture')
        d.write(data.with_suffix('.json'),dict(replicate=rep,records=436392,statistics_sha256=d.digest(data),files={'tiny.bin':d.digest(folder/'tiny.bin')}))
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


class WorkerTests(unittest.TestCase):
    def test_streaming_crash_resume_and_calibration_provenance(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);sources={'PR-0007.md':'fixture','PR-0007-A01.md':'fixture'}
            counts=dict(ref=39,normal_cal=31,fault_cal=4,null_eval=30,fault_eval=15)
            class Clock:
                def check(self):pass
                def commit(self):pass
            generated=[]
            def fake(root,t,start,spechash,gpu):
                generated.append((str(root),t['id'],start));n=min(s.CHUNK,t['count']-start)
                a=fixture(n,start+1000*s.conditions()[t['case']]['id']+100*s.ROLES[t['role']])
                meta=dict(task=t,start=start,count=n,specification_sha256=spechash,values_sha256=d.sha(a.tobytes()),unavailable_rows=0)
                r.save_npz(r.chunk_path(root,t,start),values=a,metadata=np.array(r.canonical(meta)))
            original=r.save_npz;stopped=[False]
            def crashing(path,**arrays):
                original(path,**arrays)
                if 'decisions' in Path(path).parts and not stopped[0]:stopped[0]=True;raise RuntimeError('Injected after atomic decision commit')
            with (patch.object(s,'COUNTS',counts),patch.object(s,'SIZES',(7,15,31)),patch.object(s,'MIX_COUNTS',(2,4)),
                  patch.object(s,'CHUNK',16),patch.object(r,'generate_chunk',side_effect=fake),
                  patch.object(s,'make_inputs',side_effect=AssertionError('No scientific generation permitted')),
                  patch('builtins.print')):
                interrupted=base/'interrupted';fresh=base/'fresh'
                for root in (interrupted,fresh):d.new_run(root,'fixture',sources)
                worker=r.Worker(interrupted,base/'archives',None,Clock())
                with patch.object(r,'save_npz',side_effect=crashing):
                    with self.assertRaisesRegex(RuntimeError,'Injected'):worker.replicate(0)
                self.assertTrue((d.rep_dir(interrupted,0)/'CALIBRATION_FROZEN.json').exists())
                before={p:d.digest(p) for p in d.evidence(d.rep_dir(interrupted,0))}
                worker.replicate(0);r.Worker(fresh,base/'archives',None,Clock()).replicate(0)
                for p,h in before.items():self.assertEqual(d.digest(p),h)
                self.assertEqual(len(generated),len(set(generated)))
                with r.open_npz(interrupted/'carry/rep_00.npz') as a,r.open_npz(fresh/'carry/rep_00.npz') as b:
                    for k in a.files:np.testing.assert_array_equal(a[k],b[k])
                folder=d.rep_dir(interrupted,0)
                with r.open_npz(folder/'orders/B_M.npz') as z:order=z['normal_order'];positions=z['replacement_order']
                for fault in s.FAULTS:
                    for n in s.MIX_COUNTS:
                        with r.open_npz(folder/'calibration'/(s.mix_id('B_M',fault,n)+'.npz')) as z:
                            meta=json.loads(str(z['provenance']));origin=z['origin_role'];ids=z['record_ids']
                            self.assertEqual(meta['replaced_positions'],positions[:n].tolist())
                            self.assertEqual(meta['removed_normal_ids'],order[:15][positions[:n]].tolist())
                            np.testing.assert_array_equal(ids[positions[:n]],np.arange(n));self.assertEqual(int(origin.sum()),n)
                self.assertEqual(d.read(interrupted/'carry/rep_00.json')['records'],sum(t['count'] for t in s.tasks(0)))


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
const q=globalThis.pr0007Queue;
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
        py=ROOT/'numerics/colab/pr0007_lfdc_calibration_a25_one_cell.py';tree=ast.parse(py.read_text());values={}
        for n in tree.body:
            if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('_CELL_ID','_PAYLOAD','_PAYLOAD_SHA256'):
                values[n.targets[0].id]=ast.literal_eval(n.value)
        raw=zlib.decompress(base64.b64decode(values['_PAYLOAD']));self.assertEqual(d.sha(raw),values['_PAYLOAD_SHA256'])
        import build_pr0007_colab as builder
        sources=json.loads(raw)['sources']
        for n,p in builder.PATHS.items():self.assertEqual(sources[n],(ROOT/p).read_text())
        nb=d.read(py.with_suffix('.ipynb'));self.assertEqual(len(nb['cells']),1)
        self.assertEqual(''.join(nb['cells'][0]['source']),py.read_text());self.assertEqual(nb['cells'][0]['outputs'],[])


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
