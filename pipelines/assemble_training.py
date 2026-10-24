"""Assemble already processed training inputs without changing the originals."""
import argparse
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys

from biomoax.data import CELLS, H5_FILES, METADATA, check_inputs
from common import sha256


def required_files():
    return list(dict.fromkeys([*H5_FILES, *METADATA,
        *(f'features/l1000_{cell}_{count}.npy' for cell, count in CELLS.items())]))


def resolve_mapping(manifest, required):
    data = json.loads(manifest.read_text())
    if not isinstance(data, dict) or set(data) != set(required):
        raise ValueError('Manifest keys must match the required processed training paths exactly')
    resolved = {}
    for name, value in data.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or '\\' in name:
            raise ValueError('Unsafe destination path: '+name)
        if not isinstance(value, str) or not value:
            raise ValueError('Source paths must be nonempty strings')
        source = Path(value)
        source = (source if source.is_absolute() else manifest.parent/source).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        resolved[name] = source
    return resolved


def stage(mapping, output, copy_arrays=False):
    output.mkdir(parents=True, exist_ok=False)
    inventory = []
    for name, source in mapping.items():
        target = output/'data/processed'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        # Training can materialize metadata/cache files: never link metadata to originals.
        linked = not copy_arrays and source.suffix in {'.h5', '.npy'}
        if linked:
            target.symlink_to(source)
        else:
            shutil.copy2(source, target)
        entry = dict(destination=name, source=str(source), bytes=source.stat().st_size,
                     mode='symlink' if linked else 'copy')
        if not linked:
            entry['sha256'] = sha256(target)
        inventory.append(entry)
    return inventory


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--manifest', type=Path, help='JSON: required destination path -> existing source file')
    source.add_argument('--source-root', type=Path, help='Existing project root with data/processed layout')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--copy-arrays', action='store_true', help='Copy large arrays instead of symlinking')
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    required = required_files()
    if a.manifest:
        mapping = resolve_mapping(a.manifest, required)
    else:
        mapping = {name:(a.source_root/'data/processed'/name).resolve() for name in required}
        missing = [name for name, path in mapping.items() if not path.is_file()]
        if missing:
            raise ValueError('Missing processed inputs: '+', '.join(missing))
    inventory = stage(mapping, a.output, a.copy_arrays)
    report = check_inputs(a.output)
    (a.output/'input_check.json').write_text(json.dumps(report, indent=2)+'\n')
    (a.output/'assembly_sources.json').write_text(json.dumps(inventory, indent=2)+'\n')
    if not report['passed']:
        raise ValueError('Training contract failed; see input_check.json. No completion marker written.')
    subprocess.run([sys.executable, str(Path(__file__).with_name('prepare_records.py')),
                    '--project-root', str(a.output.resolve()), '--output', str((a.output/'records').resolve())], check=True)
    (a.output/'assembly_complete.json').write_text(json.dumps({
        'status':'complete', 'checked_files':report['checked_files'],
        'validation':'Input shapes and core processed row/partition mappings; no GPU training',
        'arrays':'copied' if a.copy_arrays else 'linked; source files must remain available',
    }, indent=2)+'\n')
    print('Prepared PROJECT_ROOT: '+str(a.output.resolve()))


if __name__ == '__main__':
    main()
