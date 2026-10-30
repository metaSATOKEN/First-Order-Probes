# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Small engineering fixtures only. Never invokes the Colab pilot/main runner."""
from pathlib import Path
import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from types import ModuleType, SimpleNamespace
import zipfile
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import fusion_confirmation_pr0002_v1 as f
import pr0002_checks as checks


class TestPR0002(unittest.TestCase):
    def test_deterministic_acceptance(self):
        self.assertEqual(checks.deterministic_checks()["status"],"PASS")

    def test_frozen_kernel_hash(self):
        self.assertEqual(f.sha_file(f.kernel.__file__),f.KERNEL_SHA)

    def test_all_conditions_kernel_parity(self):
        # 16 x 64 distinct pilot fixtures, replicate49 (not the Colab timing pilot).
        for cid,c in f.conditions().items():
            role={"normal":"normal_eval","fault":"fault_eval","structural":"structural_eval"}[c["kind"]]
            t=dict(id=f"49/{cid}/{role}",replicate=49,condition=cid,role=role,total=64)
            actual=f.generate("pilot",t,0,64)
            prior=f.kernel.simulate_digital(c["params"],f.rng_for("pilot",t,0),64)
            expected=np.column_stack([prior[k] for k in f.COLUMNS])
            np.testing.assert_array_equal(actual,expected)
            if cid=="REVERSE_M":self.assertLess(actual[:,13].max(),2e-12)

    def test_noise_and_fault_order_independent_fixtures(self):
        def pair(rng,count,length,*args):
            x=np.arange(length,dtype=float)[None,:]/length
            return np.tile(x+.2,(count,1)),np.tile(.3-x,(count,1))
        def noise(p,rng,count,length,factor=1.):
            x=np.arange(length)[None,:]
            return np.tile(.02*np.cos(x),(count,1)),np.tile(.03*np.sin(x),(count,1))
        for cid,c in f.conditions().items():
            p=c["params"];n=round(p["T"]/p["dt"]);w=int(np.ceil(20*max(p["tau_i"],p["tau_j"])/p["dt"]))
            length=n+2*w;d=int(p["strength"]) if p.get("fault")=="delay" else 0
            a,b=pair(None,2,length+2*abs(d));a=a[:,abs(d):abs(d)+length];b=b[:,abs(d)-d:abs(d)-d+length]
            if p.get("fault")=="gain":b=b*(1+p["strength"])
            if p.get("fault")=="private":b=b+np.sqrt(.02)*pair(None,2,length)[1]
            if p.get("fault")=="upstream":
                u,v=pair(None,2,length);a=a+np.sqrt(.5)*u;b=b+np.sqrt(.5)*v
            ni,nj=noise(p,None,2,length);a=a+ni;b=b+nj
            if p.get("fault")=="time_reverse":a=a[:,::-1];b=b[:,::-1]
            ni,nj=noise(p,None,2,n+w)
            def moment(x,y):
                return np.column_stack([np.sum(x[:,w:w+n]**2,axis=1)/n,np.sum(y[:,w:w+n]**2,axis=1)/n,
                                        np.sum(x[:,w:w+n]*y[:,w:w+n],axis=1)/n])
            O,N=moment(a,b),moment(ni,nj)
            ai,aj=np.exp(-p["dt"]/np.array([p["tau_i"],p["tau_j"]]))
            def energy(x,y):
                u=np.zeros(2);v=np.zeros(2);tot=np.zeros(2)
                for k in range(x.shape[1]):
                    u=aj*u+(1-aj)*x[:,k];v=ai*v+(1-ai)*y[:,k]
                    if w<=k<w+n:tot+=(u-v)**2
                return tot/n
            with patch.object(f.kernel,"digital_pair",pair),patch.object(f.kernel,"make_noise",noise):
                got=f.kernel.simulate_digital(p,None,2)
            np.testing.assert_allclose(np.column_stack([got["full_"+s] for s in ("vi","vj","c")]),O-N,rtol=2e-12,atol=2e-12)
            np.testing.assert_allclose(got["CR"],energy(a,b)-energy(ni,nj),rtol=2e-12,atol=2e-12)

    def test_trace_independent_score_recalculation(self):
        t=dict(id="00/N_M/normal_eval",replicate=0,condition="N_M",role="normal_eval",total=64)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"trace.npz"
            f.generate("pilot",t,0,64,p)
            self.assertEqual(checks.check_trace(p)["status"],"PASS")

    def test_resume_and_microbatch_invariance(self):
        t=dict(id="49/N_S/normal_eval",replicate=49,condition="N_S",role="normal_eval",total=128)
        full=f.generate("pilot",t,0,128)
        split=np.concatenate([f.generate("pilot",t,0,64),f.generate("pilot",t,64,64)])
        np.testing.assert_array_equal(full,split)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"original";f.save_chunk(root,t,0,full,"manifest",.1)
            f.write_json(root/"state.json",dict(active_seconds=123.,status="PARTIAL"))
            z=f.archive(root,"PARTIAL");restored=f.restore(z,Path(tmp)/"resumed")
            got,_=f.read_chunk(f.chunk_path(restored,t,0),t,0,"manifest")
            np.testing.assert_array_equal(got,full)
            self.assertEqual(f.read_json(restored/"state.json")["active_seconds"],123.)
            with self.assertRaises(FileExistsError):f.restore(z,restored)
            with self.assertRaises(ValueError):f.read_chunk(f.chunk_path(restored,t,0),t,0,"other")

    def test_corruption_and_unsafe_zip_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"root";f.write_json(root/"state.json",dict(active_seconds=1))
            good=f.archive(root,"PARTIAL")
            for i,entry in enumerate(("../escape", "/absolute", "a\\b")):
                p=Path(tmp)/f"bad{i}.zip"
                with zipfile.ZipFile(p,"w") as z:z.writestr(entry,"x");z.writestr("_archive_index.json","{}")
                with self.assertRaises(ValueError):f.restore(p,Path(tmp)/f"dest{i}")
            corrupt=Path(tmp)/"corrupt.zip"
            with zipfile.ZipFile(good) as src,zipfile.ZipFile(corrupt,"w") as dst:
                for n in src.namelist():dst.writestr(n,b"{}" if n=="state.json" else src.read(n))
            with self.assertRaises(ValueError):f.restore(corrupt,Path(tmp)/"dest_bad")
            self.assertFalse((Path(tmp)/"dest_bad").exists())

    def test_no_implicit_local_or_main_run(self):
        with self.assertRaises(RuntimeError):f.run("confirm","/tmp/never",{},"test")
        with self.assertRaises(ValueError):f.summarize("/tmp/never",f.specification("pilot"),"x")

    def test_exclusive_output_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            with f.run_lock(tmp):
                with self.assertRaises(RuntimeError):
                    with f.run_lock(tmp):pass
            with f.run_lock(tmp):pass

    def test_download_adapter_when_only_colab_environment_is_present(self):
        colab=ModuleType("google.colab");colab.files=SimpleNamespace(download=MagicMock())
        display=ModuleType("IPython.display");display.display=MagicMock();display.FileLink=MagicMock()
        # Imports use real sys.modules, while the runner's pre-import detection sees none.
        with patch.dict(os.environ,{"COLAB_RELEASE_TAG":"mock"}), \
             patch.dict(sys.modules,{"google.colab":colab,"IPython.display":display}), \
             patch.object(f,"sys",SimpleNamespace(modules={})):
            f.notify_download(Path("/content/test.zip"),True)
        colab.files.download.assert_called_once_with("/content/test.zip")
        display.display.assert_called_once()

    def test_zero_and_negative_values_not_excluded(self):
        a=np.zeros((10,14));a[:,11]=-2.;a[:,12]=-3.;a[:,6]=-1.
        np.testing.assert_array_equal(f.scores(a),np.tile([2.,-3.],(10,1)))

    def test_manifest_mismatch_and_nonfinite_chunk(self):
        t=dict(id="00/N_M/normal_eval",replicate=0,condition="N_M",role="normal_eval",total=2)
        with tempfile.TemporaryDirectory() as tmp:
            a=np.zeros((2,14));a[0,0]=np.nan
            f.save_chunk(tmp,t,0,a,"x",.1)
            with self.assertRaises(ValueError):f.read_chunk(f.chunk_path(tmp,t,0),t,0,"x")
            unknown=Path(tmp)/"chunks"/"unexpected.npz";unknown.write_bytes(b"test")
            with self.assertRaises(ValueError):f.audit_chunks(tmp,dict(tasks=[t]),"x")

    def test_runner_pause_restore_with_stub_scores_no_simulation(self):
        project=Path(__file__).resolve().parents[2]
        names={"PR-0002.md":project/"audits/prereg/PR-0002_fusion_independent_confirmation.md",
               "fusion_confirmation_pr0002_v1.py":Path(f.__file__),
               "covariance_consistency_colab_v1.py":Path(f.kernel.__file__),
               "pr0002_checks.py":Path(checks.__file__)}
        bundle={k:p.read_text() for k,p in names.items()}
        def fixture(mode,t,start,count,trace_path=None):
            a=np.zeros((count,14));a[:,0]=np.arange(start,start+count);return a
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{"COLAB_RELEASE_TAG":"mock-only"}), \
            patch.object(f,"generate",fixture),patch.object(f,"notify_download") as download, \
            patch.object(checks,"check_trace",return_value={"status":"MOCK_ONLY"}):
            with self.assertRaises(PermissionError):f.run("confirm",tmp,bundle,"cell")
            first=f.run("pilot",Path(tmp)/"original",bundle,"cell",max_new_chunks=3)
            self.assertEqual(first["status"],"PARTIAL")
            prior=f.read_json(Path(first["root"])/"state.json")["active_seconds"]
            second=f.run("pilot",Path(tmp)/"new-runtime",bundle,"cell",restore_zip=first["archive"])
            self.assertEqual(second["status"],"PILOT_COMPLETE")
            state=f.read_json(Path(second["root"])/"state.json")
            self.assertGreater(state["active_seconds"],prior)
            self.assertEqual(state["committed_records"],8192)
            self.assertEqual(download.call_count,2)
            spec=f.specification("pilot");mf=f.read_json(Path(second["root"])/"manifest.json")
            mh=f.sha_bytes(f.canonical(mf).encode())
            self.assertEqual(f.audit_chunks(second["root"],spec,mh)[0],8192)
            # Must not produce performance tables in a pilot, even after all fixture chunks.
            self.assertFalse((Path(second["root"])/"analysis").exists())

    def test_aggregate_outputs_with_deterministic_score_fixture(self):
        # Exercise the 50-repeat output path without generating any probe time series.
        spec=f.specification("confirm")
        def load(root,t,mh):
            a=np.zeros((t["total"],14));u=np.arange(t["total"])/t["total"]
            a[:,11]=u;a[:,12]=(np.arange(t["total"])*17%t["total"])/t["total"]
            if t["role"] in ("fault_eval","structural_eval"):
                a[:,11]+=.01+t["replicate"]*.0001
            return a
        timing=[dict(task={"condition":cid},generation_seconds=1.) for cid in f.conditions()]
        with tempfile.TemporaryDirectory() as tmp,patch.object(f,"audit_chunks",return_value=(10999800,timing)), \
             patch.object(f,"load_task",load),patch.object(f,"plot_results"):
            f.write_json(Path(tmp)/"state.json",dict(failures=[]))
            f.summarize(tmp,spec,"fixture-only")
            out=Path(tmp)/"analysis"
            result=f.read_json(out/"inference.json")
            self.assertEqual(len(result["primary"]),14)
            self.assertEqual(len(result["fpr"]),16)
            self.assertEqual(len(list((out/"decisions").glob("*.npz"))),800)
            self.assertEqual(len(list((out/"calibration").glob("*.json"))),200)
            self.assertTrue((out/"normal_tail_overlap.csv").exists())


if __name__=="__main__":
    unittest.main(verbosity=2)
