import argparse
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from condition_features import build, aggregate


class ConditionTests(unittest.TestCase):
    def fixture(self, root):
        with h5py.File(root/'crops.h5', 'w') as h:
            h.create_dataset('crops', shape=(4,5,128,128), dtype='uint16')
            h['labels'] = [10, 20, 10, 20]
            h.attrs['channel_order'] = json.dumps(['DNA','ER','RNA','AGP','Mito'])
        pd.DataFrame({'record_id':['b','a'], 'label':[20,10]}).to_csv(root/'records.csv', index=False)
        np.save(root/'features.npy', [[2.,0.],[0.,4.],[4.,0.],[float('nan'),0.]])
        np.save(root/'rows.npy', [2,1,0,3])
        return argparse.Namespace(mode='baseline', crops=root/'crops.h5', records=root/'records.csv',
                                  output=root/'out', max_crops=0, features=root/'features.npy',
                                  source_rows=root/'rows.npy', valid_rows=None, aggregation='mean')

    def test_baseline_shell_to_task_selection(self):
        with tempfile.TemporaryDirectory(prefix='condition test ') as tmp:
            root=Path(tmp); a=self.fixture(root)
            repo=Path(__file__).resolve().parents[1]
            import os
            env=dict(os.environ, CORE_PYTHON=sys.executable, PYTHONDONTWRITEBYTECODE='1')
            subprocess.run(['bash',str(repo/'scripts/09_condition_features.sh'),'baseline',
                            '--crops',str(a.crops),'--records',str(a.records),'--features',str(a.features),
                            '--source-rows',str(a.source_rows),'--output',str(a.output)], check=True,env=env)
            np.testing.assert_array_equal(np.load(a.output/'features.npy'), [[0,4],[3,0]])
            pd.DataFrame({'record_id':['a','b']}).to_csv(root/'task.csv', index=False)
            subprocess.run([sys.executable,str(repo/'pipelines/select_task_features.py'),
                            '--features',str(a.output/'features.npy'),'--feature-ids',str(a.output/'feature_ids.csv'),
                            '--task-ids',str(root/'task.csv'),'--output',str(root/'selected.npy')], check=True,env=env)
            np.testing.assert_array_equal(np.load(root/'selected.npy'), [[3,0],[0,4]])
            with self.assertRaises(FileExistsError): build(a)

    def test_image_grouping_with_stub_encoder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); a=self.fixture(root)
            a.mode='image'; a.max_crops=1; a.batch_size=2; a.head='moa'; a.device='cpu'
            a.checkpoint=root/'checkpoint.stub'; a.checkpoint.write_bytes(b'test only')
            seen=[]
            def encode(x):
                seen.append(len(x)); return np.array([1.,0.])
            build(a, encoder=encode)
            self.assertEqual(seen,[1,1])
            self.assertEqual(pd.read_csv(a.output/'selected_crops.csv').source_row.tolist(),[1,0])

    def test_invalid_mapping_and_missing_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); a=self.fixture(root)
            np.save(a.source_rows,[0,0,1,2])
            with self.assertRaisesRegex(ValueError,'Duplicate'): build(a)
            np.save(a.source_rows,[2,1,0,3])
            a.valid_rows=root/'valid.npy'; np.save(a.valid_rows,np.array([True,False,True,False]))
            with self.assertRaisesRegex(ValueError,'No usable crops'): build(a)
            self.assertFalse(a.output.exists())

    def test_normalized_mean(self):
        np.testing.assert_allclose(aggregate([[2,0],[0,5]],'normalized-mean'),[2**-.5,2**-.5])


if __name__=='__main__': unittest.main()
