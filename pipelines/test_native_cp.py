import unittest
import numpy as np
import pandas as pd
from native_cp import align


class NativeCPTests(unittest.TestCase):
    def test_mapping_order_and_missing(self):
        profiles = pd.DataFrame({'old':[8,9,99], 'f':[2.,3.,7.]})
        records = pd.DataFrame({'record_id':['b','missing','a'], 'label':[1,2,0]})
        mapping = pd.DataFrame({'source_label':[8,8,9], 'label':[0,0,1]})
        x, present = align(profiles, records, ['f'], 'old', mapping)
        np.testing.assert_array_equal(x[:,0], [3,0,2])
        np.testing.assert_array_equal(present, [True,False,True])

    def test_reject_ambiguous_mapping(self):
        profiles = pd.DataFrame({'old':[8], 'f':[2.]})
        records = pd.DataFrame({'record_id':['a'], 'label':[0]})
        mapping = pd.DataFrame({'source_label':[8,8], 'label':[0,1]})
        with self.assertRaisesRegex(ValueError,'one-to-one'):
            align(profiles, records, ['f'], 'old', mapping)

    def test_reject_nonfinite(self):
        with self.assertRaisesRegex(ValueError,'Nonfinite'):
            align(pd.DataFrame({'label':[0], 'f':[np.nan]}),
                  pd.DataFrame({'record_id':['a'], 'label':[0]}), ['f'], 'label')


if __name__ == '__main__':
    unittest.main()
