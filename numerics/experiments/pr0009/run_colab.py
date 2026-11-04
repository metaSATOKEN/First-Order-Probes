# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Read-only check by default; --run explicitly starts/resumes the Colab experiment."""
from pathlib import Path
import sys
BUNDLE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BUNDLE_DIR.parents[2] / 'tools'))
from readable_bundle import check, execute
EXPECTED_SHA256 = '92810c7853eab2d733bf49c9f3c3e753c6c42f65990e8b52dac4a8617c56d16f'

def main(argv=None):
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--check', action='store_true')
    modes.add_argument('--run', action='store_true')
    args = parser.parse_args(argv)
    sources, raw = check(BUNDLE_DIR, EXPECTED_SHA256)
    result = execute(BUNDLE_DIR, sources, raw) if args.run else {
        'status': 'PASS', 'members': len(sources), 'payload_sha256': EXPECTED_SHA256,
        'simulation_started': False, 'environment_installed': False}
    print(json.dumps(result, default=str, indent=2))
    return result

if __name__ == '__main__':
    main()
