"""Read-only verification of the retained BioMoAX checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((Path(__file__).resolve().parents[1] / 'manifests/checkpoints.json').read_text())
    failed = []
    for item in manifest['checkpoints']:
        path = args.project_root / item['path']
        digest = hashlib.sha256()
        if path.is_file():
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(chunk)
        passed = path.is_file() and path.stat().st_size == item['bytes'] and digest.hexdigest() == item['sha256']
        print(item['id'], 'PASS' if passed else 'FAIL')
        if not passed:
            failed.append(item['id'])
    raise SystemExit(1 if failed else 0)

if __name__ == '__main__':
    main()
