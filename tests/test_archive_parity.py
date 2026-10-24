"""Check extracted definitions against the bundled archived implementation."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
import torch
import biomoax.model as public


class ArchiveParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.previous = os.environ.get("BIOMOAX_PROJECT_ROOT")
        os.environ["BIOMOAX_PROJECT_ROOT"] = cls.temp.name
        path = Path(public.__file__).parent / "legacy/train_moax_v7.py"
        spec = importlib.util.spec_from_file_location("paper_archive_test",path)
        cls.archive = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.archive)

    @classmethod
    def tearDownClass(cls):
        if cls.previous is None: os.environ.pop("BIOMOAX_PROJECT_ROOT",None)
        else: os.environ["BIOMOAX_PROJECT_ROOT"] = cls.previous
        cls.temp.cleanup()

    def test_image_state_and_forward(self):
        torch.manual_seed(17)
        a = self.archive.ImageBackbone(vit_dim=24,depth=1).eval()
        b = public.ImageBackbone(vit_dim=24,depth=1).eval()
        b.load_state_dict(a.state_dict(),strict=True)
        x=torch.rand(2,5,128,128)
        with torch.no_grad():
            torch.testing.assert_close(a(x),b(x),rtol=0,atol=0)

    def test_expression_heads_normalization_and_losses(self):
        a=self.archive.ExprBackbone(6,trunk_dim=8).eval()
        b=public.ExprBackbone(6,trunk_dim=8).eval();b.load_state_dict(a.state_dict())
        h=self.archive.ProjHead(8,out_dim=4).eval()
        hp=public.ProjHead(8,out_dim=4).eval();hp.load_state_dict(h.state_dict())
        with torch.no_grad():
            x=torch.randn(3,6);z=h(a(x));zp=hp(b(x))
            torch.testing.assert_close(z,zp,rtol=0,atol=0)
            mask=torch.tensor([[False,True,False],[True,False,False],[False,False,False]])
            temp=torch.tensor(-2.)
            torch.testing.assert_close(public.supcon(z,mask,temp),self.archive.supcon(z,mask,temp),rtol=0,atol=0)
            torch.testing.assert_close(public.cross_supcon(z,z,mask,temp),self.archive.cross_supcon(z,z,mask,temp),rtol=0,atol=0)
            image=torch.rand(2,5,8,8)
            torch.testing.assert_close(public.gpu_normalize(image),self.archive.gpu_normalize(image),rtol=0,atol=0)


if __name__=='__main__':unittest.main()
