import torch
import unittest
from deeptensor.models import DeepTensor, skip, SimpleForward2D
from deeptensor.losses import L1Norm, TVNorm
from deeptensor.physics import video2codedvideo
from deeptensor.baselines import parafac_als_cuda

class TestSmokeModels(unittest.TestCase):
    def test_simple_forward_2d(self):
        model = SimpleForward2D(n_inputs=3, n_outputs=10, nconvs=[16, 32])
        inp = torch.randn(2, 3, 32, 32)
        out = model(inp)
        self.assertEqual(out.shape, (2, 10, 32, 32))

    def test_losses(self):
        l1 = L1Norm()
        tv = TVNorm()
        t1 = torch.randn(1, 10, 32, 32)
        t2 = torch.randn(1, 10, 32, 32)
        loss1 = l1(t1 - t2)
        loss_tv = tv(t1)
        self.assertGreaterEqual(loss1.item(), 0.0)
        self.assertGreaterEqual(loss_tv.item(), 0.0)

    def test_video2codedvideo(self):
        video = torch.rand(1, 8, 32, 32)
        mask = torch.rand(1, 8, 32, 32)
        coded = video2codedvideo(video, mask, nframes=8)
        self.assertEqual(coded.shape, (1, 1, 32, 32))

    def test_parafac_als_cuda_smoke(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required for PARAFAC ALS")
        target = torch.rand(32, 28, 20, device="cuda")
        recon, factors = parafac_als_cuda(target, rank=5, num_iters=3)
        self.assertEqual(recon.shape, (32, 28, 20))
        self.assertEqual(len(factors), 3)

if __name__ == "__main__":
    unittest.main()
