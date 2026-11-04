# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Portable technical checks using saved arrays and artificial statistics only.

No generator, GPU, Colab connection, environment installation, or network call.
Run in a fresh process: python -B .../tests/test_readable_fix1.py
"""
import ast
import base64
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zlib

sys.dont_write_bytecode = True
BUNDLE = Path(__file__).resolve().parents[1]
PUBLIC = BUNDLE.parents[2]
sys.path.insert(0, str(BUNDLE / 'sources'))
specification = importlib.util.spec_from_file_location('readable_entry', BUNDLE / 'run_colab.py')
entry = importlib.util.module_from_spec(specification)
specification.loader.exec_module(entry)
import numpy as np
import pr0012_methods as methods
import pr0012_preflight as preflight
import pr0012_runner as runner
import pr0012_spec as spec
import pr0006_methods as earlier


def capsule_sources():
    # Independent reconstruction against the historical payload digest, not a
    # second copy of a compressed executable. The 28 member hashes are public.
    manifest = json.loads((BUNDLE / 'bundle_manifest.json').read_text())
    sources = {name: (BUNDLE / item['path']).read_text(encoding='utf-8')
               for name, item in manifest['members'].items()}
    raw = json.dumps(dict(schema=1, sources=sources), sort_keys=True, ensure_ascii=False).encode()
    assert hashlib.sha256(raw).hexdigest() == '4045cf2857875c9d0978a19961e3daea38a382303418f2267572023eb1c3789f'
    assert manifest['cell_id'] == 'd41a11833edd6d21'
    return sources


class PackagingTests(unittest.TestCase):
    def test_all_28_members_equal_completed_capsule(self):
        self.assertEqual(entry.load_sources(), capsule_sources())

    def test_manifest_and_source_tampering_rejected(self):
        mutations = ('change', 'delete', 'extra', 'symlink', 'manifest', 'path')
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                root = Path(temp) / 'bundle'
                shutil.copytree(BUNDLE, root)
                path = root / 'sources/pr0012_methods.py'
                if mutation == 'change':
                    path.write_bytes(path.read_bytes() + b'\n')
                elif mutation == 'delete':
                    path.unlink()
                elif mutation == 'extra':
                    (root / 'sources/unexpected.py').write_text('raise RuntimeError()')
                elif mutation == 'symlink':
                    path.unlink(); path.symlink_to(BUNDLE / 'sources/pr0012_methods.py')
                else:
                    manifest_path = root / 'bundle_manifest.json'
                    data = json.loads(manifest_path.read_text())
                    if mutation == 'manifest':
                        data['cell_id'] = 'old-version'
                    else:
                        data['members']['pr0012_methods.py']['path'] = '../outside.py'
                    manifest_path.write_text(json.dumps(data))
                with self.assertRaises((ValueError, FileNotFoundError)):
                    entry.load_sources(root)

    def test_default_and_check_are_read_only(self):
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in BUNDLE.rglob('*') if p.is_file()}
        for args in ([], ['--check']):
            result = subprocess.run([sys.executable, '-B', str(BUNDLE / 'run_colab.py'), *args],
                                    capture_output=True, text=True, check=True)
            self.assertFalse(json.loads(result.stdout)['simulation_started'])
        after = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in BUNDLE.rglob('*') if p.is_file()}
        self.assertEqual(before, after)

    def test_check_never_imports_launcher(self):
        with patch.object(entry.importlib, 'import_module', side_effect=AssertionError('No dynamic import')):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(entry.main(['--check'])['status'], 'PASS')

    def test_local_run_is_blocked(self):
        with patch.object(entry.sys, 'platform', 'darwin'):
            with self.assertRaisesRegex(RuntimeError, 'Colab'):
                entry.main(['--run'])

    def test_mixed_import_rejected(self):
        sources = entry.load_sources()
        wrong = types.SimpleNamespace(__file__=str(PUBLIC / 'numerics/scripts/pr0012_runner.py'))
        with patch.dict(sys.modules, {'pr0012_runner': wrong}):
            with self.assertRaisesRegex(RuntimeError, 'Conflicting'):
                entry.check_import_origins(sources, BUNDLE / 'sources')

    def test_thin_delegation_preserves_arguments(self):
        sources = entry.load_sources()
        calls = []
        fake = types.SimpleNamespace(__file__=str(BUNDLE / 'sources/pr0012_launcher.py'),
                                     run=lambda *args: calls.append(args) or {'status': 'MOCK_ONLY'})
        with patch.object(entry, 'require_colab'), patch.object(entry.importlib, 'import_module', return_value=fake):
            result = entry.launch(sources, ['matching.zip'], False)
        self.assertEqual(result['status'], 'MOCK_ONLY')
        self.assertEqual(calls, [(sources, entry.CELL_ID, BUNDLE / 'sources', ['matching.zip'], False)])

    def test_cli_restore_options_preserve_frozen_interface(self):
        for arguments, expected in ((['--run'], ('', True)),
                                    (['--run', '--upload'], ('UPLOAD', True)),
                                    (['--run', '--restore-zip', 'a.zip', '--restore-zip', 'b.zip', '--no-download'],
                                     (['a.zip', 'b.zip'], False))):
            with patch.object(entry, 'launch', return_value={'status': 'MOCK_ONLY'}) as launch:
                with contextlib.redirect_stdout(io.StringIO()):
                    entry.main(arguments)
                self.assertEqual(launch.call_args.args[1:], expected)

    def test_notebook_defaults_check_only_in_kernel(self):
        notebook = json.loads((BUNDLE / 'run_colab.ipynb').read_text())
        cells = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
        self.assertEqual(len(cells), 3)
        calls = []
        namespace = {'entry': {'main': lambda args: calls.append(args)}}
        # Execute only the check and default-off start cells, not the path loader.
        with contextlib.redirect_stdout(io.StringIO()):
            for code in cells[1:]:
                exec(compile(code, 'notebook', 'exec'), namespace)
        self.assertEqual(calls, [['--check']])
        self.assertFalse(namespace['START_FULL_RUN'])
        self.assertNotIn('subprocess', cells[0])

    def test_runner_scientific_functions_unchanged_by_fix1(self):
        def definitions(path):
            return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text()).body
                    if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        old = definitions(PUBLIC / 'numerics/scripts/pr0012_runner.py')
        fixed = definitions(BUNDLE / 'sources/pr0012_runner.py')
        self.assertNotEqual(old.pop('preflight'), fixed.pop('preflight'))
        self.assertEqual(old, fixed)

    def test_frozen_spec_and_role_separation(self):
        self.assertEqual((spec.MASTER, spec.PREFLIGHT_MASTER), (2026101820, 2026101821))
        self.assertEqual((spec.REPEATS, spec.SHARD_REPEATS), (50, 5))
        self.assertEqual(sum(t['count'] for t in spec.tasks(0)), 479984)
        self.assertEqual(len(spec.conditions()), 17)
        self.assertEqual(len(spec.primary()), 13)
        self.assertEqual(len(set(spec.ROLES.values())), len(spec.ROLES))


class NumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = BUNDLE / 'tests/fixtures/saved_preflight.npz'
        metadata = json.loads(path.with_suffix('.json').read_text())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata['fixture_sha256']
        with np.load(path, allow_pickle=False) as saved:
            cls.cpu, cls.gpu = saved['cpu'], saved['gpu']
        assert cls.cpu.shape == cls.gpu.shape == (1088, 17)

    def test_saved_gpu_boundary_replay(self):
        result = preflight.audit(self.cpu, self.gpu)
        self.assertTrue(result['passed'])
        self.assertEqual(result['status'], 'PASS_WITH_BOUNDARY_AMBIGUITY')
        self.assertEqual(result['boundary_branch_flips'], 2)
        self.assertEqual(result['unexplained_failures'], 0)
        self.assertFalse(result['scientific_thresholds_changed'])

    def test_numeric_corruption_and_wrong_precision_rejected(self):
        self.assertEqual(preflight.audit(self.cpu, self.cpu.copy())['status'], 'PASS_EXACT')
        for value in (np.nan, np.inf, .01):
            bad = self.gpu.copy()
            bad[0, 11] = value if not np.isfinite(value) else bad[0, 11] + value
            self.assertFalse(preflight.audit(self.cpu, bad)['passed'])
        self.assertFalse(preflight.audit(self.cpu, self.gpu.astype(np.float32))['passed'])

    def test_false_decisions_rejected(self):
        original = methods.decide
        def wrong(*args, **kwargs):
            result = original(*args, **kwargs).copy()
            result[0, 0] ^= 1
            return result
        with patch.object(methods, 'decide', side_effect=wrong):
            self.assertFalse(preflight.audit(self.cpu, self.cpu.copy())['passed'])

    def test_strict_ties_and_rank_cap(self):
        state = dict(valid=True, model=dict(mean=[0., 0.], weights=[[1., 0.], [0., 1.]], disabled=[]))
        ordered = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float64)
        cut = np.ones((2, 2, 3)); cut[:, 0] = 2.; cut[1, 1] = 3.
        values = np.zeros((9, 17)); values[:, 11] = [-3, -2.5, -2, -1, 0, 1, 2, 2.5, 3]
        values[:, 12] = values[:, 11]
        result = preflight.compare_readouts(values, values.copy(), state, ordered, dict(valid=True), cut)
        self.assertTrue(result['passed'])
        bits = methods.decide(state, ordered, dict(valid=True), cut, values, 1)
        self.assertEqual(bits[2, 0], 8); self.assertEqual(bits[6, 0], 8)
        self.assertTrue(np.all(bits[:, 1] == 8))
        self.assertNotEqual(bits[1, 0], 8); self.assertNotEqual(bits[7, 0], 8)

    def test_readouts_equal_earlier_fixed_methods(self):
        state, ordered = methods.fit(self.cpu)
        calibration = np.tile(self.cpu, (16, 1))[:15999].copy()
        calibration[:, 11:13] += [.000173, .000317]
        meta, cut, indices = methods.calibrate(state, ordered, calibration)
        old_meta, old_cut, old_indices = earlier.calibrate(state, ordered, calibration)
        np.testing.assert_array_equal(cut, old_cut[:, [0, 5]])
        np.testing.assert_array_equal(indices, old_indices[:, [0, 5]])
        old_bits = earlier.decide(state, ordered, old_meta, old_cut, self.gpu)
        for method, old_index in ((0, 0), (1, 5)):
            np.testing.assert_array_equal(methods.decide(state, ordered, meta, cut, self.gpu, method),
                                          ((old_bits >> (4 * old_index)) & 15).astype(np.uint8))

    def test_invalid_calibration_is_explicit(self):
        state, ordered = methods.fit(self.cpu)
        bad = self.cpu.copy(); bad[3, 11] = np.nan
        meta, cut, _ = methods.calibrate(state, ordered, bad)
        self.assertFalse(meta['valid'])
        self.assertTrue((methods.decide(state, ordered, meta, cut, self.cpu) == 0).all())
        np.testing.assert_array_equal(methods.gates(bad, 0)[0], [2, 2])

    def test_validation_count_boundaries(self):
        for alpha, support, exceed in ((.01, 211, 291), (.001, 12, 39)):
            for count, expected in ((support, 'SUPPORTED'), (support + 1, 'UNRESOLVED'),
                                    (exceed - 1, 'UNRESOLVED'), (exceed, 'EXCEEDED')):
                self.assertEqual(methods.evidence(count, 20000, 20000, alpha)['label'], expected)
        self.assertEqual(methods.evidence(0, 19999, 20000, .01)['label'], 'UNAVAILABLE')
        self.assertLessEqual(methods.cp(35, 50, .05 / 26)[0], .5)
        self.assertGreater(methods.cp(36, 50, .05 / 26)[0], .5)

    def test_policy_selection_uses_only_gate_and_validation(self):
        labels = ('SUPPORTED', 'EXCEEDED', 'UNRESOLVED', 'UNAVAILABLE')
        for frozen in labels:
            for recovery in labels:
                np.testing.assert_array_equal(methods.select(0, 3, frozen, recovery), [4, 4, 4])
                np.testing.assert_array_equal(methods.select(2, 3, frozen, recovery), [-2, -2, -2])
                expected = [0, 0, 0] if frozen == 'SUPPORTED' else [0, -1, 7 if recovery == 'SUPPORTED' else -1]
                np.testing.assert_array_equal(methods.select(1, 3, frozen, recovery), expected)

    def test_streaming_counts_and_scalar_table(self):
        n = 43
        w = (8 + np.arange(n * 8 * 2).reshape(n, 8, 2) % 8).astype(np.uint8)
        w[::7, 3] = 0
        lf = w[:, [0, 1, 7]]
        whole = methods.empty()
        whole['selection'][0, :, 0] = [0, -1, 7]
        whole['selection'][0, :, 1] = [6, -2, 5]
        chunked = {key: value.copy() for key, value in whole.items()}
        methods.accumulate(whole, 0, 0, w, lf)
        for start, end in ((0, 9), (9, 25), (25, n)):
            methods.accumulate(chunked, 0, 0, w[start:end], lf[start:end])
        for key in whole:
            np.testing.assert_array_equal(whole[key], chunked[key])
        methods.validate(whole, False)
        for ai in range(2):
            table = np.zeros((4, 4), np.int64)
            for i in range(n):
                states = []
                for index in whole['selection'][0, 0, ai]:
                    states.append(2 if index == -1 else 3 if index == -2 or w[i, index, ai] == 0
                                  else 0 if w[i, index, ai] & 7 else 1)
                table[states[2], states[0]] += 1
            np.testing.assert_array_equal(table, whole['pairs'][0, 0, 0, ai, 1])


if __name__ == '__main__':
    unittest.main(verbosity=2)
