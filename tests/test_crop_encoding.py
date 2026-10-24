import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from biomoax.inference import encode_condition


class CropEncoder(nn.Module):
    def forward(self, x):
        return x.mean(dim=(2,3))


class UnitHead(nn.Module):
    def forward(self, x):
        return F.normalize(x,dim=-1)


class CropEncodingTests(unittest.TestCase):
    def test_rows_batches_and_default_mean(self):
        models=(CropEncoder(),nn.Identity(),UnitHead(),UnitHead())
        values=np.zeros((5,5,128,128),dtype=np.uint16)
        for i in range(5): values[i,i]=1000+i
        with patch('biomoax.inference.gpu_normalize',lambda x:x):
            rows=encode_condition(models,values,'image',batch_size=2,per_crop=True)
            other=encode_condition(models,values,'image',batch_size=4,per_crop=True)
            mean=encode_condition(models,values,'image',batch_size=2)
        np.testing.assert_allclose(rows,np.eye(5),atol=1e-6)
        np.testing.assert_allclose(rows,other)
        expected=rows.mean(0,keepdims=True);expected/=np.linalg.norm(expected,axis=1,keepdims=True)
        np.testing.assert_allclose(mean,expected,atol=1e-6)

    def test_expression_rejects_per_crop(self):
        with self.assertRaisesRegex(ValueError,'image inputs'):
            encode_condition(None,np.ones((2,5)),'expression',per_crop=True)


if __name__=='__main__': unittest.main()
