# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Small deterministic/component tests; never calls the Colab main experiment."""
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'numerics/scripts'))
import pr0004_spec as s
import pr0004_methods as m
import pr0004_io as d
import pr0004_runner as run
import pr0004_launcher as launch
import probe_backend_check_v1 as core
import fusion_confirmation_pr0002_v1 as old


def fixture(n=400,offset=0):
    # Artificial observation summaries, not simulated probe trajectories.
    t=np.arange(n,dtype=float)+offset
    a=np.zeros((n,17)); a[:,7]=1+.01*np.sin(t*.7); a[:,8]=.8+.02*np.cos(t*.6)
    a[:,11]=.02*np.sin(t*.33)+.005*np.cos(t*.19)
    a[:,12]=.03*np.cos(t*.27)+.3*a[:,11]
    a[:,14]=.7; a[:,15]=.65+.03*np.sin(t*.24); a[:,16]=.55+.02*np.cos(t*.31)
    return a


class DesignTests(unittest.TestCase):
    def test_counts(self):
        c=s.conditions(); tasks=s.tasks()
        self.assertEqual(len(c),50); self.assertEqual(len(tasks),3700)
        self.assertEqual(sum(t['count'] for t in tasks),33499400)
        self.assertEqual(sum((t['count']+4095)//4096 for t in tasks),10400)
        self.assertEqual(sum(v['kind']=='fault' for v in c.values()),37)
        self.assertEqual(len({t['id'] for t in tasks}),3700)
    def test_card_and_core(self):
        self.assertEqual(d.digest(ROOT/'audits/prereg/PR-0004_linear_fusion_source_shift_gpu.md'),s.CARD_SHA)
        self.assertEqual(d.digest(ROOT/'numerics/scripts/probe_backend_check_v1.py'),s.CORE_SHA)
    def test_seed_roles_and_batch_invariance(self):
        # 7 technical records only, cut across a logical block boundary.
        a=s.make_inputs('B_S','ref',3,62,7)
        b=s.make_inputs('B_S','ref',3,62,2); c=s.make_inputs('B_S','ref',3,64,5)
        for k in a: np.testing.assert_array_equal(a[k],np.concatenate((b[k],c[k])))
        other=s.make_inputs('B_S','cal',3,62,7)
        self.assertFalse(np.array_equal(a['source'],other['source']))
        self.assertEqual(s.MASTER,2026101710)
    def test_stationary_and_fixed_noise(self):
        # Shared fixed RNG draws through mocked Generator expose noise scaling.
        class ConstantRNG:
            def normal(self,size): return np.ones(size)
        with mock.patch.object(np.random,'Generator',return_value=ConstantRNG()):
            base=s.make_inputs('B_M','ref',0,0,1)
            for case in ('K2_M','K200_M','V05_M','V15_M'):
                z=s.make_inputs(case,'ref',0,0,1)
                np.testing.assert_array_equal(z['ni'],base['ni']); np.testing.assert_array_equal(z['nj'],base['nj'])
    def test_nominal_only(self):
        self.assertEqual(set(m.fit_reference.__code__.co_varnames[:m.fit_reference.__code__.co_argcount]),{'reference','nu'})
        for c in s.conditions().values():
            p=c['params']; self.assertAlmostEqual(s.nominal_nu(p['dt'],p['tau_i'],p['tau_j']),core.coefficients(p)[4])
    def test_small_cpu_recurrence(self):
        p=s.conditions()['B_S']['params']; z=s.make_inputs('B_S','preflight',1000,0,2)
        a,_=core.calculate(p,z); b,_=core.calculate(p,z,filt=core.loop_filter)
        np.testing.assert_allclose(a,b,rtol=2e-12,atol=2e-12)
        run.verify_values(a,'B_S')
    def test_preflight_namespace(self):
        x=s.make_inputs('B_S','preflight',1000,0,1); y=s.make_inputs('B_S','ref',0,0,1)
        self.assertFalse(np.array_equal(x['source'],y['source']))
    def test_local_main_guard(self):
        with mock.patch.object(sys,'platform','darwin'):
            with self.assertRaises(RuntimeError): run.main(Path('/tmp/not-created-pr0004'))
            with self.assertRaises(RuntimeError): launch.run({},'x','/tmp')


class MethodTests(unittest.TestCase):
    def setUp(self):
        self.ref=fixture(); self.cal=fixture(1000,1000); self.eval=fixture(300,3000)
        self.model,self.ordered=m.fit_reference(self.ref,.22)
        self.threshold=m.calibrate(self.model,self.ordered,self.cal)
    def test_legacy_four_exact(self):
        oldref=old.scores(self.ref); oldcal=old.scores(self.cal)
        cut,_=old.fit(oldref,oldcal)
        np.testing.assert_array_equal(cut,np.asarray(self.threshold['cutoffs'])[:4])
        expected,_=old.decide(old.ranks(oldref,old.scores(self.eval)),cut)
        actual,_,_=m.decide(self.model,self.ordered,self.threshold,self.eval)
        np.testing.assert_array_equal(expected,actual[:,:4])
    def test_covariance_weights(self):
        cov=np.asarray(self.model['regularized_covariance'])
        for d0,w in zip(self.model['directions'],self.model['weights']):
            expected=np.linalg.solve(cov,d0); expected/=np.sqrt(np.dot(d0,expected))
            np.testing.assert_allclose(w,expected,rtol=1e-12)
    def test_direction_formula(self):
        a=self.ref.mean(0); dg=np.array(self.model['directions'][0]); dp=np.array(self.model['directions'][1])
        for delta in (-.05,.05):
            direct=np.array([delta*a[8]-.22*((1+delta)**2-1)*a[7],
                            -2*delta*a[16]+((1+delta)**2-1)*a[15]])
            np.testing.assert_allclose(delta*dg+delta**2*dp,direct,atol=1e-15)
    def test_gain_sign_invariance(self):
        other=copy.deepcopy(self.model); other['weights'][0]=(-np.array(other['weights'][0])).tolist()
        np.testing.assert_array_equal(m.component_ranks(other,self.ordered,self.eval)['LF'],
                                      m.component_ranks(self.model,self.ordered,self.eval)['LF'])
    def test_axis_projection_degenerates_to_J(self):
        model=copy.deepcopy(self.model); model.update(mean=[0.,0.],weights=[[2.,0.],[0.,3.]])
        ordered=dict(self.ordered); ordered['LF']=np.sort(m.projections(model,m.qvalues(self.ref)),axis=0)
        ranks=m.component_ranks(model,ordered,self.eval)
        np.testing.assert_array_equal(ranks['J'],ranks['LF'])
    def test_zero_and_weak_direction(self):
        ref=self.ref.copy(); ref[:,7:9]=0;ref[:,15:17]=0
        model,ordered=m.fit_reference(ref,.22)
        self.assertEqual(model['disabled'],[0,1])
        np.testing.assert_array_equal(m.component_ranks(model,ordered,self.eval)['LF'],0)
        ref[:,7]=1e-200
        model,_=m.fit_reference(ref,.22)
        self.assertNotIn(0,model['disabled']); self.assertNotIn(1,model['disabled'])
        self.assertGreater(model['direction_norms'][0],0)
    def test_invalid_reference_not_replaced(self):
        for ref in (np.zeros((10,17)),self.ref.astype('float32')):
            with self.assertRaises(ValueError):m.fit_reference(ref,.2)
        bad=self.ref.copy(); bad[0,12]=np.nan
        with self.assertRaises(ValueError):m.fit_reference(bad,.2)
    def test_threshold_order(self):
        self.assertEqual(m.cutoff_order(10000,100),9901)
        self.assertEqual(m.cutoff_order(10000,200),9951)
    def test_rank_ties_and_saturation(self):
        saturated=self.cal.copy(); saturated[:,11]=1e4;saturated[:,12]=1e4
        threshold=m.calibrate(self.model,self.ordered,saturated)
        f,_,r=m.decide(self.model,self.ordered,threshold,saturated)
        self.assertFalse(f.any()); self.assertTrue((r['J']==len(self.ref)).all())
        tied=self.ref[[0]].repeat(10,axis=0)
        r=m.component_ranks(self.model,self.ordered,tied)['J']
        for j in range(2):
            value=abs(tied[0,11]) if j==0 else tied[0,12]
            self.assertEqual(r[0,j],np.sum(self.ordered['J'][:,j]<value))
    def test_negative_scores_preserved(self):
        q=m.qvalues(self.ref); self.assertTrue((q[:,1]<0).any())
        self.assertAlmostEqual(self.model['mean'][1],self.ref[:,12].mean())
    def test_paired_conservation(self):
        a=np.array([1,0,1,0],bool);b=np.array([0,1,1,0],bool)
        self.assertEqual(m.overlap(a,b),dict(n=4,both=1,gained=1,lost=1,neither=1,difference=0.))
    def test_zero_sd_and_multiplicity(self):
        self.assertIsNone(m.outer_interval([1]*50,12)['lower'])
        narrow=m.outer_interval(np.linspace(0,1,50),1); wide=m.outer_interval(np.linspace(0,1,50),140)
        self.assertLess(wide['lower'],narrow['lower'])


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.base=Path(self.temp.name)
        self.root=self.base/'run';self.sources={'demo.py':'raise RuntimeError("never execute")\n'}
        d.new_run(self.root,'cell',self.sources)
    def tearDown(self):self.temp.cleanup()
    def test_zip_roundtrip_and_never_execute(self):
        d.write(self.root/'data.json',dict(x=7)); z=d.snapshot(self.root,self.base,'PARTIAL')
        index=d.inspect_zip(z,'cell'); self.assertEqual(index['status'],'PARTIAL')
        d.restore(z,self.base/'restored','cell');d.check_run(self.base/'restored','cell',self.sources)
        self.assertEqual(d.read(self.base/'restored/data.json'),dict(x=7))
        with self.assertRaises(FileExistsError):d.restore(z,self.root,'cell')
    def test_corruption_and_wrong_capsule(self):
        z=d.snapshot(self.root,self.base,'PARTIAL')
        with self.assertRaises(ValueError):d.inspect_zip(z,'wrong')
        with zipfile.ZipFile(z,'a') as out:out.writestr('unknown.txt','bad')
        with self.assertRaises(ValueError):d.inspect_zip(z,'cell')
    def test_traversal_rejected(self):
        for name in ('../evil','/absolute','a/../b','a\\b','a//b'):
            with self.assertRaises(ValueError):d.safe_name(name)
    def test_no_snapshot_overwrite(self):
        a=d.snapshot(self.root,self.base,'PARTIAL'); b=d.snapshot(self.root,self.base,'PARTIAL')
        self.assertNotEqual(a,b); self.assertTrue(a.exists())
    def test_same_runtime_auto_resume(self):
        base=self.base/'base';root=base/'runs/current_cell';d.new_run(root,'cell',self.sources)
        result,mode=launch.resolve_run(base,'cell',self.sources)
        self.assertEqual(result,root);self.assertEqual(mode,'existing')
        with self.assertRaises(ValueError):launch.resolve_run(base,'cell',self.sources,'old.zip')
    def test_explicit_zip_recovery(self):
        z=d.snapshot(self.root,self.base,'PARTIAL');base=self.base/'new';base.mkdir()
        root,mode=launch.resolve_run(base,'cell',self.sources,str(z));self.assertEqual(mode,'restored')
        self.assertEqual(launch.resolve_run(base,'cell',self.sources)[1],'existing')
    def test_new_run_no_dialog(self):
        base=self.base/'fresh';base.mkdir()
        with mock.patch.object(shutil := launch.shutil,'disk_usage',return_value=mock.Mock(free=100*d.GIB)):
            _,mode=launch.resolve_run(base,'cell',self.sources)
        self.assertEqual(mode,'fresh')
    def test_budget_carries_across_restart(self):
        path=self.root/'clock.json';d.write(path,dict(seconds=100))
        clock=d.ActiveClock(path,90)
        with self.assertRaises(TimeoutError):clock.check()
        self.assertGreaterEqual(d.read(path)['seconds'],100)
    def test_npz_no_pickle_and_bad_header(self):
        p=self.root/'x.npz';run.save_npz(p,a=np.arange(10,dtype=float))
        with run.open_npz(p) as z:np.testing.assert_array_equal(z['a'],np.arange(10))
        with self.assertRaises(FileExistsError):run.save_npz(p,a=np.zeros(1))
        bad=self.root/'bad.npz';run.save_npz(bad,a=np.array([{}],dtype=object))
        with self.assertRaises(ValueError):run.open_npz(bad)
    def test_chunk_identity_and_incomplete(self):
        task=dict(id='00/B_S/ref',replicate=0,case='B_S',role='ref',count=2)
        a,_=core.calculate(s.conditions()['B_S']['params'],s.make_inputs('B_S','preflight',1000,0,2))
        p=run.chunk_path(self.root,task,0)
        run.save_npz(p,values=a,metadata=np.array(run.canonical(dict(task=task,start=0,count=2,
            specification_sha256='abc',values_sha256=d.sha(a.tobytes())))))
        read,_=run.read_chunk(self.root,task,0,'abc');np.testing.assert_array_equal(a,read)
        with self.assertRaises(ValueError):run.read_chunk(self.root,task,0,'bad')
        self.assertEqual(run.inventory(self.root,[task],'abc')['records'],2)
    def test_chunk_save_resume_fixed_input(self):
        task=dict(id='00/B_S/ref',replicate=0,case='B_S',role='ref',count=3)
        def cpu_adapter(gpu,p,z,traces=False):
            a,tr=core.calculate(p,z,traces=traces)
            return a,{k:v[:3] for k,v in tr.items()},dict(h2d_seconds=0.,compute_seconds=0.,gpu_event_seconds=0.,d2h_seconds=0.)
        with mock.patch.object(run,'gpu_values',side_effect=cpu_adapter),mock.patch.object(run,'memory',return_value={}):
            run.generate_chunk(self.root,task,0,'hash',None)
        a,_=run.read_chunk(self.root,task,0,'hash')
        expected,_=core.calculate(s.conditions()['B_S']['params'],s.make_inputs('B_S','ref',0,0,3))
        np.testing.assert_array_equal(a,expected)
        before=d.digest(run.chunk_path(self.root,task,0))
        z=d.snapshot(self.root,self.base,'PARTIAL');d.restore(z,self.base/'resumed','cell')
        b,_=run.read_chunk(self.base/'resumed',task,0,'hash')
        np.testing.assert_array_equal(a,b);self.assertEqual(before,d.digest(run.chunk_path(self.base/'resumed',task,0)))
        self.assertTrue(run.chunk_path(self.root,task,0).with_suffix('.timing.json').exists())
    def test_zip_size_limits_without_gigabyte_allocation(self):
        with mock.patch.object(d,'EXPANDED_LIMIT',1):
            with self.assertRaises(RuntimeError):d.snapshot(self.root,self.base,'PARTIAL')
        z=d.snapshot(self.root,self.base,'PARTIAL')
        with mock.patch.object(d,'ZIP_LIMIT',1):
            with self.assertRaises(ValueError):d.inspect_zip(z,'cell')
    def test_zip64_configuration(self):
        # Actual >4GiB disk/transfer is not tested locally; force ZIP64 headers on tiny data.
        with mock.patch.object(zipfile,'ZIP64_LIMIT',64):
            z=d.snapshot(self.root,self.base,'PARTIAL')
            d.inspect_zip(z,'cell')
            d.restore(z,self.base/'zip64','cell')
        d.check_run(self.base/'zip64','cell',self.sources)


class AnalysisTests(unittest.TestCase):
    def test_all_condition_analysis_and_recovery(self):
        # Artificial summaries exercise every path; no time-series simulation.
        def fake_load(root,task,h):
            return fixture(300 if task['role']=='ref' else 400,1000*list(s.ROLES).index(task['role']))
        with tempfile.TemporaryDirectory() as td,mock.patch.object(run,'load_task',side_effect=fake_load):
            first=run.evaluate_rep(td,0,'fixture')
            second=run.evaluate_rep(td,0,'fixture')
            self.assertEqual(first,second)
            self.assertEqual(len(first['rates']),114*6)
            for p in first['patterns']:
                self.assertEqual(sum(p['main32']),p['n']);self.assertEqual(sum(p['auxiliary64']),p['n'])
            # F/C have identical normal-reference models, independently calibrated thresholds.
            modeldir=next((Path(td)/'analysis/repeats/00').glob('attempt_*'))/'models'
            base=d.read(modeldir/'B_M/B_calibration.json')
            self.assertEqual(base,d.read(modeldir/'K2_M/F_calibration.json'))
            synthetic=[dict(first,replicate=i) for i in range(50)]
            summary=run.aggregate(synthetic)
            self.assertEqual(len(summary['primary12']),12);self.assertEqual(len(summary['fpr140']),140)
            self.assertEqual(len(summary['fpr_changes_from_baseline']),8*3*6)
            with self.assertRaises(ValueError):run.aggregate(synthetic[:-1])


if __name__=='__main__': unittest.main(verbosity=2)
