# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Local implementation tests; not the planned scientific main run."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import types
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import covariance_consistency_colab_v1 as s


class MathematicsTests(unittest.TestCase):
    def test_discrete_covariance_identity(self):
        for k in (2.,20.,200.):
            for r in (2.,4.,10.):
                for dt in (.001,.002,.004):
                    P = s.digital_model(k,.01*r,.01,dt)[1]
                    lam,nu = s.discrete_weights(.01*r,.01,dt)
                    self.assertAlmostEqual(P[1,2], lam*P[1,1]+nu*P[2,2], places=12)

    def test_product_threshold(self):
        self.assertAlmostEqual(s.product_threshold(.01),3.60421320225167,places=9)

    def test_analog_covariance_and_stationary_transition(self):
        for k in (2.,20.,200.):
            F,P,L,Q = s.analog_model(k,.04,.01,.002)
            np.testing.assert_allclose(F@P@F.T + Q@Q.T, P, atol=2e-14)
            np.testing.assert_allclose(L@L.T, P, atol=2e-14)
            self.assertAlmostEqual(.05*P[1,2]-.04*P[1,1]-.01*P[2,2],0,places=14)

    def test_analog_generator_sample_variance(self):
        p = dict(kappa=20., tau_i=.04, tau_j=.01, dt=.002, T=.032)
        data = s.simulate_analog(p,np.random.default_rng(1),20000)
        self.assertLess(abs(data['endpoint'].std()/s.endpoint_scale(p)-1),.035)
        self.assertGreater(np.std(data['raw']-data['endpoint']),0)
        self.assertGreater(np.std(data['centered']-data['raw']),0)

    def test_reversal_moments(self):
        p = dict(kappa=20.,tau_i=.04,tau_j=.01,dt=.002,T=.512,
                 eta_readout=.1,topology='downstream',fault='time_reverse',strength=1.)
        result = s.simulate_digital(p,np.random.default_rng(44),20)
        self.assertLess(result['reversal_moment_error'].max(),1e-14)

    def test_empirical_p_conservative_ties(self):
        np.testing.assert_allclose(s.empirical_p(np.arange(99.),np.array([200.,98.,0.])),[.01,.02,1.])
        np.testing.assert_allclose(s.empirical_p(np.ones(99),np.ones(3)),np.ones(3))

    def test_source_shift_is_not_recalibrated(self):
        cfg=s.scientific_config('main',(3,))
        shifts=[c for c in s.build_design(cfg) if c['kind']=='null_shift']
        self.assertEqual(len(shifts),2)
        for c in shifts:
            self.assertEqual(c['reference_params']['kappa'],20.)
            self.assertNotEqual(c['params']['kappa'],20.)
            self.assertEqual(c['params']['noise_kappa'],20.)


class PersistenceTests(unittest.TestCase):
    def tiny_config(self):
        cfg=s.scientific_config('smoke',(1,2,3),seed=321,batch_size=13)
        cfg['repetitions']=dict(reference=19,null=23,alternative=17,endpoint=29)
        return cfg

    def quiet_run(self,*args,**kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return s.run_suite(*args,auto_download=False,max_minutes=0,**kwargs)

    def test_interruption_resume_zip_and_equivalence(self):
        cfg=self.tiny_config()
        with tempfile.TemporaryDirectory() as tmp:
            one=self.quiet_run(cfg,Path(tmp)/'one')
            paused=self.quiet_run(cfg,Path(tmp)/'two',max_new_batches=3)
            self.assertEqual(paused['status'],'PARTIAL')
            resumed=self.quiet_run(cfg,Path(tmp)/'two')
            self.assertEqual(resumed['status'],'COMPLETE')
            self.assertEqual(resumed['reused_batches'],3)
            tasks=s.tasks_for(s.build_design(cfg),cfg)
            for task in tasks:
                a=s.load_task(one['root'],task,cfg)
                b=s.load_task(resumed['root'],task,cfg)
                self.assertEqual(set(a),set(b))
                for key in a:
                    np.testing.assert_array_equal(a[key],b[key])
            restored=self.quiet_run(cfg,Path(tmp)/'three',restore_zip=paused['archive'])
            self.assertEqual(restored['status'],'COMPLETE')
            self.assertEqual(restored['reused_batches'],3)
            with zipfile.ZipFile(restored['archive']) as z:
                for name in ('config.json','design.json','run_status.json','README.md',
                             'LICENSE','LICENSE-CC-BY-4.0.txt','runner_source.py','tables/rejection_rates.csv'):
                    self.assertIn(name,z.namelist())
                self.assertEqual(json.loads(z.read('run_status.json'))['status'],'COMPLETE')
            again=self.quiet_run(cfg,Path(tmp)/'two')
            self.assertEqual(again['new_batches'],0)

    def test_tampered_checkpoint_is_rejected(self):
        cfg=self.tiny_config()
        with tempfile.TemporaryDirectory() as tmp:
            partial=self.quiet_run(cfg,tmp,max_new_batches=1)
            path=next(Path(partial['root']).glob('chunks/*/*.npz'))
            path.write_bytes(path.read_bytes()+b'corrupt')
            with self.assertRaises(RuntimeError):
                self.quiet_run(cfg,tmp)
            self.assertEqual(json.loads((Path(partial['root'])/'run_status.json').read_text())['status'],'FAILED')

    def test_orphan_batch_is_recomputed(self):
        cfg=self.tiny_config()
        with tempfile.TemporaryDirectory() as tmp:
            partial=self.quiet_run(cfg,tmp,max_new_batches=1)
            next(Path(partial['root']).glob('chunks/*/*.json')).unlink()
            done=self.quiet_run(cfg,tmp,max_new_batches=1)
            self.assertEqual(done['new_batches'],1)
            self.assertEqual(done['reused_batches'],0)

    def test_zip_traversal_is_rejected(self):
        cfg=self.tiny_config()
        with tempfile.TemporaryDirectory() as tmp:
            archive=Path(tmp)/'bad.zip'
            with zipfile.ZipFile(archive,'w') as z:
                z.writestr('config.json',json.dumps(cfg))
                z.writestr('../escape','bad')
            with self.assertRaises(ValueError):
                s.restore_archive(archive,Path(tmp)/'target',cfg)
            self.assertFalse((Path(tmp)/'escape').exists())

    def test_different_configuration_is_not_restored(self):
        cfg=self.tiny_config()
        with tempfile.TemporaryDirectory() as tmp:
            partial=self.quiet_run(cfg,Path(tmp)/'first',max_new_batches=1)
            changed=dict(cfg,seed=999)
            with self.assertRaises(ValueError):
                s.restore_archive(partial['archive'],Path(tmp)/'second',changed)

    def test_colab_download_is_requested(self):
        cfg=self.tiny_config()
        colab=types.ModuleType('google.colab')
        colab.files=types.SimpleNamespace(download=mock.Mock())
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(sys.modules,{'google.colab':colab}):
            with contextlib.redirect_stdout(io.StringIO()):
                result=s.run_suite(cfg,tmp,max_minutes=0,auto_download=True,max_new_batches=1)
            colab.files.download.assert_called_once_with(result['archive'])

    def test_zero_time_budget_is_an_immediate_safe_pause(self):
        cfg=self.tiny_config()
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                result=s.run_suite(cfg,tmp,max_minutes=1e-12,auto_download=False)
            self.assertEqual(result['status'],'PARTIAL')
            self.assertEqual(result['new_batches'],0)

    def test_main_design_counts_and_no_test_threshold_leak(self):
        cfg=s.scientific_config('main')
        cases=s.build_design(cfg)
        tasks=s.tasks_for(cases,cfg)
        self.assertEqual(len(cases),146)
        self.assertEqual(len(tasks),197)
        self.assertEqual(sum(t['total'] for t in tasks),4779949)
        for c in cases:
            if c['experiment']==3:
                self.assertNotIn('fault',c['reference_params'])
            if c['experiment']==2:
                self.assertEqual(c['reference_params']['control_factor'],1.)


class GeneratedArtifactTests(unittest.TestCase):
    def test_one_code_cell_matches_script(self):
        root=Path(__file__).resolve().parents[2]
        source=(root/'numerics/colab/covariance_consistency_one_cell.py').read_text()
        notebook=json.loads((root/'numerics/colab/covariance_consistency_one_cell.ipynb').read_text())
        cells=[c for c in notebook['cells'] if c['cell_type']=='code']
        self.assertEqual(len(cells),1)
        self.assertEqual(''.join(cells[0]['source']),source)
        compile(source,'one_cell','exec')
        canonical=(root/'numerics/scripts/covariance_consistency_colab_v1.py').read_text()
        self.assertEqual(source.split('# === ENGINE START ===',1)[1],canonical.split('# === ENGINE START ===',1)[1])


if __name__=='__main__':
    unittest.main()
