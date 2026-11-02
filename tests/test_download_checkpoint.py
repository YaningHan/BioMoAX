import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('checkpoint_download', Path(__file__).resolve().parents[1] / 'scripts/download_checkpoint.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CheckpointDownloadTests(unittest.TestCase):
    def test_verified_download_and_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.pt'
            source.write_bytes(b'checkpoint fixture')
            item = dict(id='fixture', path='checkpoints/model.pt', bytes=source.stat().st_size,
                        sha256=hashlib.sha256(source.read_bytes()).hexdigest(), url=source.as_uri())
            target = module.download(item, root / 'destination')
            self.assertEqual(target.read_bytes(), source.read_bytes())
            self.assertEqual(module.download(item, root / 'destination'), target)

    def test_corrupt_download_is_not_installed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.pt'
            source.write_bytes(b'corrupt')
            item = dict(id='fixture', path='checkpoints/model.pt', bytes=7,
                        sha256='0' * 64, url=source.as_uri())
            with self.assertRaises(ValueError):
                module.download(item, root / 'destination')
            self.assertFalse((root / 'destination/checkpoints/model.pt').exists())
            self.assertFalse((root / 'destination/checkpoints/model.pt.part').exists())

    def test_unpublished_checkpoint_is_not_downloaded(self):
        with self.assertRaises(ValueError):
            module.download(dict(id='unpublished', url=None), '.')
