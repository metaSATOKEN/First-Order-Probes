# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Small local unit/transport tests only. No CUDA or performance experiment."""
import ast
import base64
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'numerics/scripts'))
sys.path.insert(0,str(ROOT/'numerics/tools'))
import numpy as np
import probe_backend_check_v1 as core
import probe_check_io as disk
import probe_check_launcher as launcher
import build_probe_backend_check as build


def tiny(name='B_M'):
    return dict(core.conditions()[name],dt=.01,tau_i=.02,tau_j=.01,T=.08)


def fixed_inputs(p, count=3):
    n,w,g = core.dimensions(p)
    def wave(length, shift):
        return np.sin(np.arange(count*length,dtype=np.float64).reshape(count,length)*.21+shift)
    z = dict(initial=wave(3,.1)*.2,source=wave(n+2*w+2*g,.2),
             ni=wave(n+2*w,.3)*.03,nj=wave(n+2*w,.5)*.04,
             ci=wave(n+w,.7)*.03,cj=wave(n+w,.9)*.04)
    if p['fault']=='private':
        z.update(private_initial=wave(3,1.1)*.2,private_source=wave(n+2*w,1.3))
    return z


class Numerics(unittest.TestCase):
    def test_plan_counts(self):
        self.assertEqual(len(core.conditions()),22)
        self.assertEqual(len(core.tasks()),94)
        self.assertEqual(len([x for x in core.tasks() if x['phase']=='time']),72)
        self.assertEqual(len(core.COLUMNS),17)

    def test_nonzero_initial_filter_oracle(self):
        x = np.array([[0.,1.,0.,-1.],[2.,0.,3.,0.]])
        initial = np.array([.7,-.2])
        a,b = .97,.03
        np.testing.assert_allclose(core.cpu_filter(x,a,b,initial),core.loop_filter(x,a,b,initial),rtol=0,atol=2e-15)
        np.testing.assert_allclose(core.cpu_filter(x,a,b,initial)[:,0],b*x[:,0]+a*initial,rtol=0,atol=1e-15)

    def test_all_faults_fixed_input_oracle(self):
        for name in core.conditions():
            with self.subTest(name=name):
                p = tiny(name)
                z = fixed_inputs(p)
                a,ta = core.calculate(p,z,traces=True)
                b,tb = core.calculate(p,z,filt=core.loop_filter,traces=True)
                self.assertEqual(a.shape,(3,17))
                self.assertTrue(core.compare(a,b)['passed'])
                self.assertTrue(core.identity(a)['passed'])
                for key in ta:
                    self.assertTrue(core.compare(ta[key],tb[key])['passed'])

    def test_stationary_discrete_identity(self):
        for p in core.conditions().values():
            cov,_ = core.stationary(p['kappa'],p['tau_i'],p['tau_j'],p['dt'])
            _,_,_,lam,nu = core.coefficients(p)
            self.assertAlmostEqual(cov[1,2],lam*cov[1,1]+nu*cov[2,2],places=13)

    def test_rng_batch_independence(self):
        p = tiny('PRIVATE_M')
        whole = core.make_inputs(p,'FIXTURE',1,0,71)
        a = core.make_inputs(p,'FIXTURE',1,0,17)
        b = core.make_inputs(p,'FIXTURE',1,17,54)
        for key in whole:
            np.testing.assert_array_equal(whole[key],np.concatenate([a[key],b[key]]))

    def test_rng_namespaces(self):
        p = tiny()
        a = core.make_inputs(p,'A',0,0,2)['source']
        self.assertFalse(np.array_equal(a,core.make_inputs(p,'B',0,0,2)['source']))
        self.assertFalse(np.array_equal(a,core.make_inputs(p,'A',1,0,2)['source']))

    def test_float64_and_nonfinite_rejected(self):
        a = np.zeros((3,17))
        self.assertFalse(core.compare(a,a.astype(np.float32))['passed'])
        b = a.copy(); b[0,0] = np.nan
        self.assertFalse(core.compare(a,b)['passed'])
        self.assertFalse(core.compare(a,a[:2])['passed'])

    def test_tolerance_columns(self):
        a = np.zeros((3,17)); b = a.copy()
        b[0,5] = 3e-12
        result = core.compare(a,b)
        self.assertFalse(result['passed'])
        self.assertEqual(result['exceed_by_column'][5],1)
        self.assertAlmostEqual(result['max_scaled'],1.5)

    def test_reversal_preserves_center_moments(self):
        p = tiny('REVERSE')
        out,_ = core.calculate(p,fixed_inputs(p))
        self.assertLess(np.max(out[:,13]),1e-14)

    def test_gain_is_signal_only(self):
        p = tiny('GAIN_PLUS'); p['strength'] = .05
        z = fixed_inputs(p)
        _,t = core.calculate(p,z,traces=True)
        normal = dict(p,fault='none',strength=0.)
        _,b = core.calculate(normal,z,traces=True)
        np.testing.assert_allclose(t['yj']-z['nj'],1.05*(b['yj']-z['nj']),atol=1e-15,rtol=0)

    def test_signed_corrections_retained(self):
        p = tiny(); z = fixed_inputs(p)
        z['ci'] *= 1000; z['cj'] *= 1000
        out,_ = core.calculate(p,z)
        self.assertTrue((out[:,6]<0).all())
        self.assertTrue((out[:,12]<0).any())

    def test_gpu_source_precision_contract(self):
        self.assertIn('const double*',core.CUDA_SOURCE)
        self.assertIn('(long long)t * B + j',core.CUDA_SOURCE)
        self.assertNotIn('__half',core.CUDA_SOURCE)
        source = (ROOT/'numerics/scripts/probe_backend_check_v1.py').read_text()
        self.assertIn('--fmad=false',source)
        self.assertNotIn('--use_fast_math',source)


class Transport(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.root = self.base/'run'
        self.sources = {'dummy.py':'raise RuntimeError("NEVER EXECUTE RESTORED SOURCE")\n'}
        disk.new_run(self.root,'TEST',self.sources)

    def tearDown(self):
        self.temp.cleanup()

    def test_archive_roundtrip_no_execution(self):
        disk.write(self.root/'state.json',dict(ready=True))
        archive = disk.snapshot(self.root,self.base/'zip','CHECKPOINT')
        restored = self.base/'restored'
        disk.restore(archive,restored,'TEST')
        self.assertEqual(disk.read(restored/'state.json'),{'ready':True})
        disk.check_run(restored,'TEST',self.sources)
        self.assertEqual((restored/'sources/dummy.py').read_text(),self.sources['dummy.py'])

    def test_reject_wrong_version(self):
        archive = disk.snapshot(self.root,self.base/'zip','CHECKPOINT')
        with self.assertRaises(ValueError):
            disk.inspect_zip(archive,'WRONG')

    def test_corrupt_payload_rejected(self):
        archive = disk.snapshot(self.root,self.base/'zip','CHECKPOINT')
        bad = self.base/'bad.zip'
        with zipfile.ZipFile(archive) as a,zipfile.ZipFile(bad,'w') as b:
            for item in a.infolist():
                data = a.read(item.filename)
                b.writestr(item,data+b'changed' if item.filename.endswith('dummy.py') else data)
        with self.assertRaises(ValueError):
            disk.inspect_zip(bad,'TEST')

    def test_zip_traversal_and_duplicate_rejected(self):
        import warnings
        for names in (('../escape','manifest.json'),('manifest.json','manifest.json')):
            bad = self.base/'bad.zip'
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',UserWarning)
                with zipfile.ZipFile(bad,'w') as archive:
                    for name in names: archive.writestr(name,'{}')
            with self.assertRaises(ValueError):
                disk.inspect_zip(bad,'TEST')
        self.assertFalse((self.base.parent/'escape').exists())

    def test_unsafe_paths(self):
        for name in ('../outside','a/../../b','/absolute','a\\b','a//b','.'):
            with self.subTest(name=name),self.assertRaises(ValueError):
                disk.safe_name(name)

    def test_existing_never_overwritten(self):
        archive = disk.snapshot(self.root,self.base/'zip','CHECKPOINT')
        with self.assertRaises(FileExistsError):
            disk.restore(archive,self.root,'TEST')
        with self.assertRaises(ValueError):
            disk.check_run(self.root,'TEST',{'dummy.py':'different'})

    def test_latest_single_lineage(self):
        a = disk.snapshot(self.root,self.base/'zip','CHECKPOINT')
        b = disk.snapshot(self.root,self.base/'zip','CHECKPOINT')
        self.assertEqual(disk.choose_archive([a,b],'TEST'),b)
        other = self.base/'other'
        disk.new_run(other,'TEST',self.sources)
        c = disk.snapshot(other,self.base/'zip','CHECKPOINT')
        with self.assertRaises(ValueError):
            disk.choose_archive([a,c],'TEST')

    def test_concurrent_lock_rejected(self):
        with disk.lock(self.base/'run.lock'):
            with self.assertRaises(RuntimeError):
                with disk.lock(self.base/'run.lock'):
                    pass

    def test_launcher_fresh_then_automatic_resume(self):
        base = self.base/'new'
        root,mode = launcher.resolve_run(base,'TEST',self.sources)
        self.assertEqual(mode,'fresh')
        same,mode = launcher.resolve_run(base,'TEST',self.sources)
        self.assertEqual(mode,'existing')
        self.assertEqual(root,same)

    def test_launcher_explicit_restore_then_automatic(self):
        archive = disk.snapshot(self.root,self.base/'zip','CHECKPOINT')
        base = self.base/'new'
        root,mode = launcher.resolve_run(base,'TEST',self.sources,str(archive))
        self.assertEqual(mode,'restored')
        self.assertEqual(launcher.resolve_run(base,'TEST',self.sources)[1],'existing')
        with self.assertRaises(ValueError):
            launcher.resolve_run(base,'TEST',self.sources,str(archive))

    def test_launcher_uploaded_file_autodetection(self):
        base = self.base/'new'
        archive = disk.snapshot(self.root,base,'CHECKPOINT')
        root,mode = launcher.resolve_run(base,'TEST',self.sources)
        self.assertEqual(mode,'restored')
        self.assertEqual(disk.read(root/'manifest.json')['run_id'],disk.inspect_zip(archive,'TEST')['run_id'])

    def test_environment_keeps_cuda_removes_injection(self):
        with mock.patch.dict('os.environ',{'LD_LIBRARY_PATH':'cuda-loader','LD_PRELOAD':'driver',
                                         'PYTHONPATH':'bad','PYTHONHOME':'bad'}):
            env = launcher.child_environment()
        self.assertEqual(env['LD_LIBRARY_PATH'],'cuda-loader')
        self.assertEqual(env['LD_PRELOAD'],'driver')
        self.assertNotIn('PYTHONPATH',env)
        self.assertNotIn('PYTHONHOME',env)

    def test_committed_task_resume_and_corruption(self):
        task = dict(id='fixed-test',phase='check',case='B_M',repeat=100,batch=3,count=3)
        folder = self.root/'tasks'/task['id']/'attempt'
        data,_ = core.calculate(tiny(),fixed_inputs(tiny()))
        core.save_values(folder/'cpu.npz',data)
        core.save_values(folder/'gpu.npz',data)
        row = dict(task=task,attempt='attempt',files={n:disk.digest(folder/n) for n in ('cpu.npz','gpu.npz')},
                   comparison=core.verify_pair(folder))
        disk.write(folder.parent/'done.json',row)
        self.assertEqual(core.load_done(self.root,task),row)
        self.assertIsNone(core.load_done(self.root,dict(task,id='not-done')))
        core.save_values(folder/'gpu.npz',data+1)
        with self.assertRaises(ValueError):
            core.load_done(self.root,task)

    def test_npz_header_bomb_rejected_before_allocation(self):
        import struct
        header = (repr(dict(descr='<f8',fortran_order=False,shape=(999999999,17)))+'\n').encode()
        payload = b'\x93NUMPY\x01\x00'+struct.pack('<H',len(header))+header+b'12345678'
        path = self.root/'bad.npz'
        with zipfile.ZipFile(path,'w') as archive:
            archive.writestr('values.npy',payload)
        with self.assertRaises(ValueError):
            core.open_values(path)

    def test_subprocess_checkpoint_delivery_plumbing(self):
        seen = []
        code = "print('CHECKPOINT_ZIP=/tmp/mock-checkpoint.zip'); print('finished')"
        rc = launcher.stream_process([sys.executable,'-c',code],5,self.base/'child.log',seen.append)
        self.assertEqual(rc,0)
        self.assertEqual(seen,[Path('/tmp/mock-checkpoint.zip')])
        self.assertIn('finished',(self.base/'child.log').read_text())

    def test_local_worker_guard(self):
        with mock.patch.object(core.sys,'platform','darwin'):
            with self.assertRaises(RuntimeError):
                core.main(self.root)

    def test_summary_environment_separation_and_numerical_failure(self):
        data,_ = core.calculate(tiny(),fixed_inputs(tiny()))
        plan = []
        for repeat,env in ((0,'ENV_A'),(1,'ENV_B')):
            for batch in (64,512):
                task = dict(id=f'time-test-r{repeat}-b{batch}',phase='time',case='B_M',repeat=repeat,batch=batch,count=3)
                plan.append(task)
                folder = self.root/'tasks'/task['id']/'attempt'
                for backend in ('cpu','gpu'): core.save_values(folder/(backend+'.npz'),data)
                row = dict(task=task,attempt='attempt',environment_id=env,
                    timing=dict(cpu=dict(compute_seconds=2.,end_to_end_seconds=4.),
                                gpu=dict(compute_seconds=1.,end_to_end_seconds=3.)),
                    files={n:disk.digest(folder/n) for n in ('cpu.npz','gpu.npz')},comparison=core.verify_pair(folder))
                disk.write(folder.parent/'done.json',row)
        with mock.patch.object(core,'tasks',return_value=plan):
            result = core.summarize(self.root)
            self.assertEqual(result['status'],'PILOT_COMPLETE')
            self.assertEqual(len(result['timing_by_environment']),4)
            self.assertEqual(len(result['batch_comparisons']),4)
            for value in result['timing_by_environment'].values():
                self.assertFalse(value['full_three_rounds'])
                self.assertEqual(value['end_to_end_seconds']['median_ratio'],.75)
                self.assertEqual(value['end_to_end_vs_cpu64']['median_ratio'],.75)
            folder = self.root/'tasks'/plan[0]['id']/'attempt'
            bad = data.copy(); bad[0,0] += .1
            core.save_values(folder/'gpu.npz',bad)
            row = disk.read(folder.parent/'done.json')
            row['files']['gpu.npz'] = disk.digest(folder/'gpu.npz')
            row['comparison'] = core.verify_pair(folder)
            disk.write(folder.parent/'done.json',row)
            self.assertEqual(core.summarize(self.root)['status'],'NUMERICAL_FAIL')
            (folder.parent/'done.json').unlink()
            self.assertEqual(core.summarize(self.root)['status'],'PARTIAL')

    def test_small_backend_assembly(self):
        p = tiny(); z = fixed_inputs(p,7)
        task = dict(id='fixed',phase='time',case='B_M',repeat=0,batch=3,count=7)
        def inputs(params,case,repeat,start,count):
            return {k:v[start:start+count].copy() for k,v in z.items()}
        with mock.patch.object(core,'conditions',return_value={'B_M':p}),mock.patch.object(core,'make_inputs',side_effect=inputs):
            timing = core.backend_task(task,None,'cpu',self.root)
        with np.load(self.root/'cpu.npz') as values:
            self.assertTrue(core.compare(core.calculate(p,z)[0],values['values'])['passed'])
        self.assertGreaterEqual(timing['end_to_end_seconds'],timing['compute_seconds'])
        self.assertEqual(timing['h2d_seconds'],0)


class Packaging(unittest.TestCase):
    def test_bundle_matches_sources_and_notebook(self):
        path = ROOT/'numerics/colab/probe_cpu_gpu_check_one_cell.py'
        tree = ast.parse(path.read_text())
        literals = {}
        for node in tree.body:
            if isinstance(node,ast.Assign) and isinstance(node.value,ast.Constant):
                for target in node.targets:
                    if isinstance(target,ast.Name): literals[target.id] = node.value.value
        raw = zlib.decompress(base64.b64decode(literals['_PAYLOAD']))
        self.assertEqual(disk.sha(raw),literals['_PAYLOAD_SHA256'])
        sources = json.loads(raw)['sources']
        for name,relative in build.PATHS.items():
            self.assertEqual(sources[name],(ROOT/relative).read_text())
        self.assertEqual(literals['_CELL_ID'],disk.sha(raw+build.PREAMBLE.encode()+build.BOOTSTRAP.encode())[:16])
        notebook = json.loads(path.with_suffix('.ipynb').read_text())
        self.assertEqual(len(notebook['cells']),1)
        self.assertEqual(''.join(notebook['cells'][0]['source']),path.read_text())


if __name__=='__main__':
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    print(stream.getvalue())
    target = ROOT/'output/runs/pr0003_implementation_validation'
    target.mkdir(parents=True,exist_ok=True)
    disk.atomic(target/'unit_tests.txt',stream.getvalue().encode())
    disk.write(target/'unit_tests.json',dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        passed=result.wasSuccessful(),gpu_executed=False,scientific_performance_run=False,
        scope='small local unit tests, fixed-input numerical oracle, tiny seed plumbing, ZIP and package verification',
        tested_sources={p:disk.digest(ROOT/p) for p in (
            'numerics/scripts/probe_check_io.py','numerics/scripts/probe_check_launcher.py',
            'numerics/scripts/probe_backend_check_v1.py','numerics/tools/build_probe_backend_check.py',
            'numerics/tests/test_probe_backend_check.py')},
        python=sys.version,numpy=np.__version__,scipy=core.scipy.__version__))
    sys.exit(0 if result.wasSuccessful() else 1)
