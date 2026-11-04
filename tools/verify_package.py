# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Read-only integrity/link/syntax check of this selected distribution."""
from pathlib import Path
from urllib.parse import unquote, urlsplit
import ast
import base64
import hashlib
import json
import re
import sys
import zlib

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r'!?\[[^\]]*\]\(([^)]+)\)')
EXCLUDED_DIRS = {'.venv', 'venv', '__pycache__', '.git', '.pytest_cache', '.lake'}
MANUSCRIPT_PDF = 'Calibration_Update_Audits_for_First_Order_Probes_EN_v1.0.pdf'
ENCODED_TOKEN = re.compile(r'(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{512,}={0,2}(?![A-Za-z0-9+/])')


def compressed_code_bundles(text):
    """Detect the historical source-capsule format; license text is allowed."""
    count = 0
    for token in ENCODED_TOKEN.findall(text):
        try:
            value = json.loads(zlib.decompress(base64.b64decode(token, validate=True)))
        except (ValueError, zlib.error, UnicodeDecodeError):
            continue
        if isinstance(value, dict):
            sources = value.get('sources', value)
            if isinstance(sources, dict) and any(
                    isinstance(name, str) and isinstance(body, str)
                    and name.endswith(('.py', '.lean')) for name, body in sources.items()):
                count += 1
    return count


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    bootstrap = '--bootstrap' in sys.argv
    files = sorted(p for p in ROOT.rglob('*') if p.is_file()
                   and not any(x in EXCLUDED_DIRS for x in p.relative_to(ROOT).parts))
    errors = []
    manifest_path = ROOT / 'provenance/PACKAGE_MANIFEST.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())['files']
        actual = {str(p.relative_to(ROOT)) for p in files if p != manifest_path}
        if actual != set(manifest):
            errors.append({'manifest_file_set': {'missing': sorted(set(manifest) - actual),
                                                 'extra': sorted(actual - set(manifest))}})
        for name, item in manifest.items():
            p = ROOT / name
            if p.is_file() and (p.stat().st_size != item['bytes'] or sha(p) != item['sha256']):
                errors.append({'manifest_mismatch': name})
    elif not bootstrap:
        errors.append({'missing': 'provenance/PACKAGE_MANIFEST.json'})
    links = python_files = notebook_cells = text_files = 0
    for p in files:
        name = str(p.relative_to(ROOT))
        if p.is_symlink():
            errors.append({'symlink': name})
        if p.suffix in {'.py', '.ipynb'} and re.search(r'one[_-]?cell', p.name, re.I):
            errors.append({'retired_onecell_artifact': name})
        if p.stat().st_size >= 100 * 1024**2:
            errors.append({'file_too_large': name})
        if (p.suffix.lower() in {'.zip', '.pyc'}
                or (p.suffix.lower() == '.pdf' and name != MANUSCRIPT_PDF)
                or p.name == '.DS_Store' or p.name.startswith('.env')):
            errors.append({'excluded_artifact': name})
        if p.suffix in {'.md', '.tex', '.json', '.txt', '.log', '.py', '.ipynb', '.toml', '.lean', '.cff'} or p.name in {'README', 'NOTICE', 'LICENSE', '.gitignore'}:
            text = p.read_text(encoding='utf-8')
            text_files += 1
            if '/' + 'Users/' in text:
                errors.append({'machine_local_home_path': name})
            if compressed_code_bundles(text):
                errors.append({'compressed_scientific_code_bundle': name})
        if p.suffix == '.md':
            text = p.read_text()
            # Fenced examples are not rendered links.
            text = re.sub(r'^```.*?^```\s*$', '', text, flags=re.M | re.S)
            for target in LINK.findall(text):
                target = target.strip('<>')
                if urlsplit(target).scheme or target.startswith('#'):
                    continue
                dest = (p.parent / unquote(target.split('#', 1)[0])).resolve()
                links += 1
                if not dest.is_relative_to(ROOT) or not dest.exists():
                    errors.append({'broken_link': name, 'target': target})
                elif '#item-' in target:
                    fragment = target.split('#', 1)[1]
                    if '## ' + fragment not in dest.read_text():
                        errors.append({'missing_catalog_anchor': name, 'target': target})
        if p.suffix == '.py':
            try:
                ast.parse(p.read_text(), filename=name)
                python_files += 1
            except SyntaxError as exc:
                errors.append({'python_syntax': name, 'line': exc.lineno})
        if p.suffix == '.ipynb':
            nb = json.loads(p.read_text())
            for i, cell in enumerate(nb.get('cells', [])):
                if cell.get('cell_type') != 'code':
                    continue
                code = ''.join(cell.get('source', []))
                if cell.get('outputs') or cell.get('execution_count') is not None:
                    errors.append({'notebook_saved_output': name, 'cell': i})
                try:
                    ast.parse(code, filename=name + ':' + str(i))
                    notebook_cells += 1
                except SyntaxError as exc:
                    errors.append({'notebook_syntax': name, 'cell': i, 'line': exc.lineno})
    total = sum(p.stat().st_size for p in files)
    if total >= 250 * 1024**2:
        errors.append({'package_too_large': total})
    result = {'status': 'PASS' if not errors else 'FAIL', 'manifest_checked': manifest_path.exists(),
              'files': len(files), 'bytes': total, 'local_links': links,
              'python_files': python_files, 'notebook_code_cells': notebook_cells,
              'text_files_scanned_for_local_home_paths': text_files, 'errors': errors}
    print(json.dumps(result, indent=2))
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
