import unittest
import numpy as np
import torch
from biomoax.metrics import retrieve
from biomoax.baselines import CrossModalRidge
from biomoax.model import cross_supcon, supcon, ProjHead


class CoreTests(unittest.TestCase):
    def test_exclusion_eligibility_and_chance(self):
        q = np.array([[1.,0.],[0.,1.],[1.,1.]])
        g = np.array([[1.,0.],[0.,1.],[-1.,0.]])
        pos = np.array([[True,False,True],[False,True,False],[False,False,False]])
        excluded = np.zeros_like(pos); excluded[0,0] = True
        r = retrieve(q,g,pos,excluded,(1,3))
        self.assertEqual(r["n_eligible"],2)
        self.assertEqual(r["top1_percent"],50.)
        self.assertEqual(r["top3_percent"],100.)
        self.assertAlmostEqual(r["random_top1_percent"],100*(.5+1/3)/2)

    def test_rejects_zero_profiles(self):
        with self.assertRaises(ValueError): retrieve([[0.,0.]],[[1.,0.]],np.array([[True]]))

    def test_ridge_training_scaler_and_independent_directions(self):
        from sklearn.linear_model import Ridge
        rng = np.random.default_rng(7)
        x = rng.normal(size=(20,4)); y = rng.normal(size=(20,3))
        m = CrossModalRidge().fit(x,y)
        np.testing.assert_allclose(m.image_scaler.mean_, x.mean(0))
        expected = Ridge(alpha=1.,solver="cholesky").fit(m.expression_scaler.transform(y),m.image_scaler.transform(x))
        np.testing.assert_allclose(m.predict(y,"e2i"),expected.predict(m.expression_scaler.transform(y)))
        before = m.expression_scaler.mean_.copy();m.predict(x*100,"i2e")
        np.testing.assert_array_equal(before,m.expression_scaler.mean_)

    def test_no_positive_cross_anchor_contributes_zero(self):
        z=torch.eye(2); mask=torch.tensor([[True,False],[False,False]])
        loss=cross_supcon(z,z,mask,torch.tensor(0.))
        expected=torch.logsumexp(torch.tensor([1.,0.]),0)-1
        self.assertTrue(torch.allclose(loss,expected/2))

    def test_heads_normalize_and_losses_have_gradients(self):
        h=ProjHead(4);z=h(torch.randn(3,4))
        torch.testing.assert_close(z.norm(dim=1),torch.ones(3))
        mask=torch.tensor([[False,True,False],[True,False,False],[False,False,False]])
        loss=supcon(z,mask,torch.tensor(-2.));loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in h.parameters()))


if __name__=="__main__": unittest.main()
