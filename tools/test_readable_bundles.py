# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Small packaging tests only; no scientific generators or network access."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import readable_bundle

ROOT = Path(__file__).resolve().parents[1]
ENTRIES = json.loads((ROOT/'provenance/READABLE_MIGRATION.json').read_text())['bundles']


class ReadableBundles(unittest.TestCase):
    def test_all_entries_check_without_importing_science(self):
        for info in ENTRIES:
            with self.subTest(info['directory']):
                path = ROOT/info['directory']/'run_colab.py'
                result = subprocess.run([sys.executable,'-B',str(path),'--check'],
                                        check=True,capture_output=True,text=True)
                data = json.loads(result.stdout)
                self.assertFalse(data['simulation_started'])
                self.assertFalse(data['environment_installed'])

    def test_integrity_and_manifest_followup_tampering(self):
        for info in ENTRIES:
            if info.get('existing_readable'):  # covered by its own FIX1 suite
                continue
            original = ROOT/info['directory']
            for mutation in ('change','delete','extra','symlink','path','coordinated'):
                with self.subTest(bundle=info['directory'],mutation=mutation), tempfile.TemporaryDirectory() as temp:
                    target = Path(temp)/'bundle';shutil.copytree(original,target)
                    manifest = json.loads((target/'bundle_manifest.json').read_text())
                    name = next(iter(manifest['members']))
                    member = manifest['members'][name]
                    path = target/member['path']
                    if mutation in ('change','coordinated'):
                        path.write_bytes(path.read_bytes()+b'\n')
                    elif mutation=='delete':path.unlink()
                    elif mutation=='extra':(target/'sources/extra.py').write_text('raise RuntimeError()')
                    elif mutation=='symlink':path.unlink();path.symlink_to(original/member['path'])
                    elif mutation=='path':member['path']='../outside'
                    if mutation=='coordinated':
                        member['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
                        member['bytes']=path.stat().st_size
                    (target/'bundle_manifest.json').write_text(json.dumps(manifest))
                    with self.assertRaises((ValueError,FileNotFoundError)):
                        readable_bundle.check(target,info['payload_sha256'])

    def test_runtime_boundary(self):
        with patch.object(readable_bundle.sys,'platform','darwin'):
            with self.assertRaisesRegex(RuntimeError,'Colab'):
                readable_bundle.execute(Path('.'),{},b'')

    def test_notebooks_are_explicit_and_clear(self):
        for info in ENTRIES:
            notebook=json.loads((ROOT/info['directory']/'run_colab.ipynb').read_text())
            codes=[''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code']
            self.assertTrue(any('START_FULL_RUN = False' in c for c in codes))
            self.assertTrue(any('--check' in c for c in codes))
            for cell in notebook['cells']:
                if cell['cell_type']=='code':
                    self.assertEqual(cell['outputs'],[])
                    self.assertIsNone(cell['execution_count'])
                    ast.parse(''.join(cell['source']))


if __name__=='__main__':unittest.main(verbosity=2)
