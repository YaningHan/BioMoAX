import json
from pathlib import Path
import tempfile
import unittest
from assemble_training import resolve_mapping, stage


class AssemblyTests(unittest.TestCase):
    def test_mapping_and_safe_staging(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'meta.csv').write_text('id\n1\n')
            (root/'array.npy').write_bytes(b'fixture')
            mapping={'metadata/meta.csv':'meta.csv','features/array.npy':'array.npy'}
            manifest=root/'inputs.json'; manifest.write_text(json.dumps(mapping))
            resolved=resolve_mapping(manifest,list(mapping))
            out=root/'assembled'; stage(resolved,out)
            self.assertTrue((out/'data/processed/features/array.npy').is_symlink())
            meta=out/'data/processed/metadata/meta.csv'
            self.assertFalse(meta.is_symlink())
            meta.write_text('changed')
            self.assertEqual((root/'meta.csv').read_text(),'id\n1\n')
            with self.assertRaises(FileExistsError): stage(resolved,out)

    def test_reject_incomplete_and_unsafe_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); manifest=root/'inputs.json'
            manifest.write_text('{}')
            with self.assertRaises(ValueError): resolve_mapping(manifest,['missing'])
            manifest.write_text(json.dumps({'../escape':'file'}))
            with self.assertRaisesRegex(ValueError,'Unsafe'): resolve_mapping(manifest,['../escape'])


if __name__=='__main__': unittest.main()
