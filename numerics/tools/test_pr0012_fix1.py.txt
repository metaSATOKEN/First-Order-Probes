# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Saved real preflight replay + artificial regression. No physical generation."""
import ast,base64,io,json,sys,tempfile,time,types,unittest,zipfile,zlib
from pathlib import Path
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
FIX=ROOT/'numerics/repairs/pr0012_preflight_fix1'
sys.path[:0]=[str(FIX),str(ROOT/'numerics/scripts'),str(ROOT/'numerics/tools')]
import pr0012_runner as r
import pr0012_launcher as l
import pr0012_preflight as p
import pr0012_spec as s
import pr0012_methods as m
import pr0012_io as d
import test_pr0012 as prior
import build_pr0012_fix1_colab as builder
ZIP=ROOT/'audits/verification/PR0012_failure_606fc449db00/original_TECHNICAL_FAILURE.zip'


def saved():
    cs=[];gs=[]
    with zipfile.ZipFile(ZIP) as z:
        for case in s.conditions():
            with np.load(io.BytesIO(z.read('sessions/b6069349acc7/preflight/'+case+'.npz')),allow_pickle=False) as f:
                cs.append(f['cpu']);gs.append(f['gpu'])
    return cs,gs


def bundle(test):
    py=ROOT/'numerics/colab/pr0012_mixture_recovery_b50_fix1_one_cell.py';v={}
    for n in ast.parse(py.read_text()).body:
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('_PAYLOAD','_PAYLOAD_SHA256'):
            v[n.targets[0].id]=ast.literal_eval(n.value)
    raw=zlib.decompress(base64.b64decode(v['_PAYLOAD']));test.assertEqual(d.sha(raw),v['_PAYLOAD_SHA256'])
    sources=json.loads(raw)['sources']
    for n,path in builder.PATHS.items():test.assertEqual(sources[n],(ROOT/path).read_text())
    nb=d.read(py.with_suffix('.ipynb'));test.assertEqual(len(nb['cells']),1)
    test.assertEqual(''.join(nb['cells'][0]['source']),py.read_text());test.assertFalse(nb['cells'][0]['outputs'])
    test.assertIn(l.RUNTIME_FOLDER,py.read_text())


class FixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cs,cls.gs=saved();cls.cpu=np.concatenate(cls.cs);cls.gpu=np.concatenate(cls.gs)

    def test_saved_real_failure_reproduces_then_classifies(self):
        state,ordered=m.fit(self.cpu);cal=np.tile(self.cpu,(16,1))[:15999].copy();cal[:,11:13]+=[.000173,.000317]
        meta,cut,_=m.calibrate(state,ordered,cal)
        counts=[int(np.count_nonzero(m.decide(state,ordered,meta,cut,self.cpu,mi)!=m.decide(state,ordered,meta,cut,self.gpu,mi))) for mi in (0,1)]
        self.assertEqual(counts,[0,2])
        a=p.audit(self.cpu,self.gpu);self.assertTrue(a['passed']);self.assertEqual(a['boundary_branch_flips'],2)
        self.assertEqual(a['unexplained_failures'],0)
        self.assertEqual(a['status'],'PASS_WITH_BOUNDARY_AMBIGUITY')
        self.assertTrue(all(x['halfplane_mismatches']==0 for x in a['methods']))

    def test_exact_and_corrupt_raw_and_projection(self):
        self.assertEqual(p.audit(self.cpu,self.cpu.copy())['status'],'PASS_EXACT')
        for value in (float('nan'),float('inf'),.01):
            bad=self.gpu.copy();bad[0,11]=value if not np.isfinite(value) else bad[0,11]+value
            self.assertFalse(p.audit(self.cpu,bad)['passed'])
        self.assertFalse(p.audit(self.cpu,self.gpu.astype(np.float32))['passed'])

    def test_false_bits_rejected_even_when_numeric_data_identical(self):
        original=m.decide
        def bad(*args,**kw):
            out=original(*args,**kw).copy();out[0,0]^=1;return out
        with patch.object(m,'decide',side_effect=bad):
            a=p.audit(self.cpu,self.cpu.copy())
        self.assertFalse(a['passed']);self.assertGreater(a['unexplained_failures'],0)

    def test_strict_ties_inactive_and_away_boundary_sides(self):
        state=dict(valid=True,model=dict(mean=[0.,0.],weights=[[1.,0.],[0.,1.]],disabled=[]))
        ordered=np.array([[1.,1.],[2.,2.],[3.,3.]],dtype=np.float64)
        cut=np.ones((2,2,3));cut[:,0]=2.;cut[1,1]=3.
        x=np.zeros((9,17));x[:,11]=[-3,-2.5,-2,-1,0,1,2,2.5,3];x[:,12]=x[:,11]
        a=p.compare_readouts(x,x.copy(),state,ordered,dict(valid=True),cut)
        self.assertTrue(a['passed']);self.assertEqual(a['status'],'PASS_EXACT')
        bits=m.decide(state,ordered,dict(valid=True),cut,x,1)
        self.assertEqual(bits[2,0],8);self.assertEqual(bits[6,0],8)
        self.assertTrue(np.all(bits[:,1]==8));self.assertNotEqual(bits[1,0],8);self.assertNotEqual(bits[7,0],8)
        y=x.copy();y[6,11]=np.nextafter(2.,np.inf)
        self.assertTrue(p.compare_readouts(x,y,state,ordered,dict(valid=True),cut)['passed'])
        # Degenerate all-boundary coverage is not called a successful check.
        z=np.zeros((9,17));z[:,11:13]=2.
        self.assertFalse(p.compare_readouts(z,z.copy(),state,ordered,dict(valid=True),cut)['passed'])

    def replay_preflight(self,root,corrupt=False):
        cases=list(s.conditions())
        def inputs(case,*args):return dict(fixture=np.array(cases.index(case)))
        def cpu(params,z,**kw):return self.cs[int(z['fixture'])],dict(trace=np.zeros((64,2)))
        def gpu(g,params,z,**kw):
            a=self.gs[int(z['fixture'])].copy()
            if corrupt:a[0,0]+=.01
            return a,dict(trace=np.zeros((3,2))),dict(test_replay=True)
        with patch.object(s,'make_inputs',side_effect=inputs),patch.object(r.core,'calculate',side_effect=cpu),patch.object(r,'gpu_values',side_effect=gpu):
            r.preflight(root,None,'replay',types.SimpleNamespace(check=lambda:None))

    def test_full_preflight_entrypoint_saved_arrays_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as td:
            self.replay_preflight(td)
            f=Path(td)/'sessions/replay/preflight'
            self.assertEqual(len(list((f/'checks').glob('*.json'))),17)
            self.assertTrue(d.read(f/'result.json')['passed'])
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError,'CPU/GPU'):self.replay_preflight(td,True)
            f=Path(td)/'sessions/replay/preflight'
            self.assertFalse(d.read(f/'result.json')['passed'])

    def test_scientific_runner_functions_unchanged(self):
        def defs(path):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        a=defs(ROOT/'numerics/scripts/pr0012_runner.py');b=defs(FIX/'pr0012_runner.py')
        self.assertEqual(set(a),set(b));self.assertNotEqual(a.pop('preflight'),b.pop('preflight'));self.assertEqual(a,b)
        self.assertEqual(s.specification()['master_seed'],2026101820) if 'master_seed' in s.specification() else self.assertEqual(s.MASTER,2026101820)

    def test_known_old_failure_no_picker_fresh_and_same_fix_live(self):
        self.assertEqual(l.matching_archives([ZIP],'fix-test'),[])
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);sources={'PR-0012.md':'fixture','PR0012-development.json':'fixture'}
            with patch.object(l.shutil,'disk_usage',return_value=types.SimpleNamespace(free=100*1024**3)),patch.object(l,'upload_streaming',side_effect=AssertionError('No picker')):
                root,mode=l.resolve_run(base,'fix-test',sources,content=base/'empty')
                self.assertEqual(mode,'fresh');self.assertEqual(l.resolve_run(base,'fix-test',sources)[1],'existing')

    def test_other_version_and_corrupt_archives_never_silent_fresh(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);f=base/'wrong.zip'
            with zipfile.ZipFile(f,'w') as z:z.writestr('ARCHIVE_INDEX.json',json.dumps(dict(cell_id='other')))
            with self.assertRaisesRegex(ValueError,'Other-version'):l.matching_archives([f],'fix')
            f.write_bytes(b'corrupt')
            with self.assertRaises(zipfile.BadZipFile):l.matching_archives([f],'fix')
            sources={'PR-0012.md':'fixture','PR0012-development.json':'fixture'}
            with self.assertRaises(ValueError):l.resolve_run(base/'explicit','fix',sources,restore_zips=str(ZIP))


if __name__=='__main__':
    assert Path(r.__file__).parent==FIX and Path(l.__file__).parent==FIX
    prior.BrowserTests.test_bundle_sources_not_executed_and_exact=bundle
    started=time.monotonic();suite=unittest.TestSuite([
        unittest.defaultTestLoader.loadTestsFromModule(prior),unittest.defaultTestLoader.loadTestsFromTestCase(FixTests)])
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite);print(stream.getvalue())
    report=dict(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        seconds=time.monotonic()-started,scientific_records_generated=0,new_physical_records_generated=0,
        actual_new_gpu_run=False,saved_gpu_records_replayed=1088,actual_browser_tested=False,
        python=sys.version,numpy=np.__version__)
    if len(sys.argv)>1:
        out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
        d.atomic(out/'test.log',stream.getvalue().encode());d.write(out/'results.json',report)
        cs,gs=saved();d.write(out/'saved_preflight_replay.json',p.audit(np.concatenate(cs),np.concatenate(gs)))
    print(json.dumps(report));sys.exit(0 if result.wasSuccessful() else 1)
