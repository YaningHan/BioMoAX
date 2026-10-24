"""Validate explicit site identities and local input paths, without downloading."""
import argparse
import csv
import json
from pathlib import Path
from common import CHANNELS, sha256

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    with a.manifest.open() as f:
        reader = csv.DictReader(f)
        required = {'plate','well','site','kind','label',*CHANNELS,*['illum_'+c for c in CHANNELS]}
        if not required <= set(reader.fieldnames or []):
            raise ValueError('Missing columns: '+str(sorted(required-set(reader.fieldnames or []))))
        rows = list(reader)
    if not rows:
        raise ValueError('Empty manifest')
    seen = set()
    for row in rows:
        key = tuple(row[k] for k in ('plate','well','site'))
        if any(not v.strip() for v in key) or key in seen:
            raise ValueError('Empty or duplicate site identity: '+str(key))
        seen.add(key)
        if row['kind'] not in ('crispr','compound') or int(row['label']) < 0:
            raise ValueError('Invalid kind or label')
        for col in [*CHANNELS,*['illum_'+c for c in CHANNELS]]:
            path = Path(row[col])
            if not path.is_absolute():
                path = a.manifest.resolve().parent/path
            if not path.is_file():
                raise FileNotFoundError(path)
    with a.output.open('x') as f:
        json.dump(dict(status='passed', n_sites=len(rows), manifest_sha256=sha256(a.manifest),
                       scope='Identity and file-presence checks; image decoding occurs in segmentation'), f, indent=2)

if __name__ == '__main__':
    main()
