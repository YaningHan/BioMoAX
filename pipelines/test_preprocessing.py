"""CPU contract tests, not an end-to-end reproduction claim."""
import tempfile
import unittest
from pathlib import Path
import json
import h5py
import numpy as np
from common import CHANNELS, dp_normalize, new_output, validate_crops
from segment import correct, crop_mask

class ProcessingTests(unittest.TestCase):
    def test_channel_permutation_and_minmax(self):
        x = np.zeros((1,5,128,128), np.uint16)
        for c in range(5):
            x[0,c,0,0] = 100
            x[0,c,1,1] = 10*(c+1)
        np.testing.assert_allclose(dp_normalize(x)[0,1,1], [.1,.3,.2,.4,.5])
        self.assertTrue(np.isfinite(dp_normalize(np.zeros_like(x))).all())

    def test_illumination(self):
        x = np.full((2,2), 12, np.uint16); illum = np.full((2,2), 2, np.float32)
        np.testing.assert_allclose(correct(x, illum, 'crispr'), 6)
        np.testing.assert_allclose(correct(x, illum, 'compound'), 12, rtol=1e-6)
        with self.assertRaises(ValueError):
            correct(x, illum*0, 'compound')

    def test_crop_boundaries(self):
        x = np.ones((5,160,160), np.float32)*7
        mask = np.zeros((160,160), np.int32)
        mask[75:85,75:85] = 1; mask[:10,:10] = 2
        crops, origins = crop_mask(x, mask)
        self.assertEqual(crops.shape, (1,5,128,128))
        self.assertEqual(origins[0]['mask_id'], 1)
        self.assertTrue((crops == 7).all())

    def test_contract_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            with h5py.File(Path(tmp)/'x.h5', 'w') as h:
                h.create_dataset('crops', shape=(1,5,128,128), dtype='uint16')
                with self.assertRaises(ValueError):
                    validate_crops(h)
                h.attrs['channel_order'] = json.dumps(CHANNELS)
                self.assertEqual(len(validate_crops(h)), 1)
            with self.assertRaises(FileExistsError):
                new_output(tmp)

if __name__ == '__main__':
    unittest.main()
