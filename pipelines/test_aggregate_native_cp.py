import unittest
import numpy as np
import pandas as pd
from aggregate_native_cp import aggregate


class AggregateTests(unittest.TestCase):
    def test_median_and_imputation(self):
        wells=pd.DataFrame({'brd_core':['a','a','a','b','c'],'int_label':[0,0,0,1,2],
                            'f':[1,2,100,np.nan,6]})
        result=aggregate(wells,['f'],'brd_core','int_label')
        np.testing.assert_array_equal(result.f,[2,4,6])

    def test_reject_conflicting_identity(self):
        wells=pd.DataFrame({'brd_core':['a','a'],'int_label':[0,1],'f':[1,2]})
        with self.assertRaisesRegex(ValueError,'one-to-one'):
            aggregate(wells,['f'],'brd_core','int_label')

    def test_reject_empty_feature(self):
        wells=pd.DataFrame({'brd_core':['a'],'int_label':[0],'f':[np.nan]})
        with self.assertRaisesRegex(ValueError,'without any finite'):
            aggregate(wells,['f'],'brd_core','int_label')


if __name__=='__main__':
    unittest.main()
