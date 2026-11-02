"""Download a published BioMoAX checkpoint and verify its size and SHA-256."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def checksum(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def download(item, project_root):
    if not item.get('url'):
        raise ValueError(f"No public download is configured for {item['id']}")
    root = Path(project_root).resolve()
    target = (root / item['path']).resolve()
    if not target.is_relative_to(root):
        raise ValueError('Checkpoint path must stay within the project root')
    if target.exists():
        if target.stat().st_size == item['bytes'] and checksum(target) == item['sha256']:
            return target
        raise ValueError(f'Existing file does not match the published checksum: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + '.part')
    if temporary.exists():
        raise ValueError(f'Remove the incomplete download before retrying: {temporary}')
    try:
        request = urllib.request.Request(item['url'], headers={'User-Agent': 'BioMoAX-checkpoint-downloader'})
        with urllib.request.urlopen(request, timeout=180) as source, temporary.open('xb') as destination:
            for chunk in iter(lambda: source.read(1024 * 1024), b''):
                destination.write(chunk)
        if temporary.stat().st_size != item['bytes'] or checksum(temporary) != item['sha256']:
            raise ValueError('Downloaded checkpoint failed size or SHA-256 verification')
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def main():
    manifest = json.loads((Path(__file__).resolve().parents[1] / 'manifests/checkpoints.json').read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path('.'))
    args = parser.parse_args()
    item = next((x for x in manifest['checkpoints'] if x['id'] == manifest['default_checkpoint']), None)
    if item is None:
        parser.error('The pretrained model is missing from the download manifest')
    try:
        path = download(item, args.project_root)
    except ValueError as error:
        parser.exit(1, str(error) + '\n')
    print(f'BioMoAX model ready: {path}')


if __name__ == '__main__':
    main()
