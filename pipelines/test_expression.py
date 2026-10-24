import tempfile
import unittest
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from expression import extract


class ExpressionTests(unittest.TestCase):
    def test_cell_selection_order_and_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            pd.DataFrame({'record_id': ['second', 'first', 'missing'],
                          'broad_id': ['BRD-B', 'BRD-A', 'BRD-C']}).to_csv(p/'records.csv', index=False)
            pd.DataFrame({'sample_id': ['s0', 's1', 's2', 's3'],
                          'pert_id': ['BRD-A-1', 'BRD-A-2', 'BRD-B', 'BRD-A'],
                          'pert_type': ['trt_cp']*4,
                          'cell_iname': ['U2OS', 'U2OS', 'U2OS', 'A549']}).to_csv(p/'inst.txt', sep='\t', index=False)
            with h5py.File(p/'matrix.gctx', 'w') as f:
                f['0/META/COL/id'] = np.array(['s0', 's1', 's2', 's3'], dtype='S')
                f['0/META/ROW/id'] = np.array(['g2', 'g1'], dtype='S')
                f['0/DATA/0/matrix'] = np.array([[1, 3], [3, 5], [8, 9], [99, 99]], dtype='float32')
            args = dict(records=p/'records.csv', instinfo=p/'inst.txt', gctx=p/'matrix.gctx', cell='U2OS', output=p/'out')
            extract(**args)
            a = np.load(p/'out/expression.npy')
            np.testing.assert_array_equal(a[:2], [[8, 9], [2, 4]])
            self.assertTrue(np.isnan(a[2]).all())
            self.assertEqual(pd.read_csv(p/'out/genes.csv').gene_id.tolist(), ['g2', 'g1'])
            self.assertEqual(pd.read_csv(p/'out/samples.csv').sample_id.tolist(), ['s2', 's0', 's1'])
            with self.assertRaises(FileExistsError):
                extract(**args)


if __name__ == '__main__':
    unittest.main()
