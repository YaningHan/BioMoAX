import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd
from identity_retrieval import evaluate

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from biomoax.baselines import CrossModalRidge
from biomoax.inference import encode_condition


class RetrievalEntryTests(unittest.TestCase):
    def test_crop_encoding_to_identity(self):
        import torch
        class Encoder(torch.nn.Module):
            def forward(self,x): return x.mean(dim=(2,3))
        class Head(torch.nn.Module):
            def forward(self,x): return torch.nn.functional.normalize(x,dim=-1)
        _,rows=self.fixture()
        crops=np.zeros((10,5,128,128),dtype=np.uint16)
        crops[:5,0]=1000; crops[5:,1]=2000
        models=(Encoder(),None,Head(),None)
        with patch('biomoax.inference.gpu_normalize',lambda x:x):
            features=encode_condition(models,crops,'image',batch_size=3,per_crop=True)
        metrics,ranks,_=evaluate(features,rows,'mean',[42,142])
        self.assertTrue((ranks['rank']==1).all())
        self.assertTrue((metrics.hit_rate==100).all())

    def fixture(self):
        rows=pd.DataFrame({'record_id':['a']*5+['b']*5,'label':[10]*5+[20]*5,'source_row':range(10)})
        x=np.array([[2.,0.]]*5+[[0.,3.]]*5)
        return x,rows

    def test_identity_splits_and_shell(self):
        x,rows=self.fixture()
        metrics,ranks,splits=evaluate(x,rows,'cellprofiler',[42,142])
        self.assertTrue((ranks['rank']==1).all())
        self.assertTrue((metrics.hit_rate==100).all())
        for _,g in splits.groupby(['seed','record_id']):
            self.assertEqual((g.side=='query').sum(),1)
            self.assertEqual(g.source_row.nunique(),5)
        shuffled=np.array([9,3,5,0,6,4,7,2,8,1])
        other=evaluate(x[shuffled],rows.iloc[shuffled].reset_index(drop=True),'cellprofiler',[42,142])
        pd.testing.assert_frame_equal(splits,other[2])
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); np.save(p/'x.npy',x); rows.to_csv(p/'rows.csv',index=False)
            command=['bash',str(REPO/'scripts/14_identity_retrieval.sh'),'--features',str(p/'x.npy'),
                '--crop-ids',str(p/'rows.csv'),'--method','cellprofiler','--output',str(p/'out')]
            env=dict(os.environ,CORE_PYTHON=sys.executable,PYTHONDONTWRITEBYTECODE='1')
            subprocess.run(command,env=env,check=True)
            self.assertTrue((p/'out/summary.csv').exists())
            self.assertNotEqual(subprocess.run(command,env=env,capture_output=True).returncode,0)

    def test_identity_missing_values_and_invalid_ids(self):
        x,rows=self.fixture(); x[:,1]=np.nan
        metrics,_,_=evaluate(x,rows,'cellprofiler',[42])
        self.assertTrue(np.isfinite(metrics.hit_rate).all())
        with self.assertRaisesRegex(ValueError,'finite'): evaluate(x,rows,'mean',[42])
        rows.loc[1,'source_row']=0
        with self.assertRaisesRegex(ValueError,'Duplicate'): evaluate(x,rows,'cellprofiler',[42])

    def test_ridge_cli_both_directions(self):
        rng=np.random.default_rng(42); x=rng.normal(size=(12,3));y=rng.normal(size=(12,4))
        model=CrossModalRidge().fit(x,y)
        with tempfile.TemporaryDirectory(prefix='ridge entry ') as tmp:
            p=Path(tmp); joblib.dump(model,p/'model.joblib')
            env=dict(os.environ,CORE_PYTHON=sys.executable,PYTHONDONTWRITEBYTECODE='1')
            for direction,query,gallery,modality in [('i2e',x[:3],y[3:],'expression'),('e2i',y[:3],x[3:],'image')]:
                np.save(p/'q.npy',query);np.save(p/'g.npy',gallery)
                command=['bash',str(REPO/'scripts/13_ridge_predict.sh'),'--model',str(p/'model.joblib'),
                    '--query',str(p/'q.npy'),'--gallery',str(p/'g.npy'),'--direction',direction,'--output',str(p/direction)]
                subprocess.run(command,env=env,check=True)
                np.testing.assert_allclose(np.load(p/direction/'query.npy'),model.predict(query,direction))
                np.testing.assert_allclose(np.load(p/direction/'gallery.npy'),model.gallery(gallery,modality))
                self.assertNotEqual(subprocess.run(command,env=env,capture_output=True).returncode,0)
            np.save(p/'q.npy',np.full((3,4),np.nan))
            command[-1]=str(p/'invalid')
            self.assertNotEqual(subprocess.run(command,env=env,capture_output=True).returncode,0)
            self.assertFalse((p/'invalid').exists())


if __name__=='__main__': unittest.main()
