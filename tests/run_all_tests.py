import unittest
import os
import sys
import torch
import numpy as np

# Ensure root is in sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

class TestDeepTensorPackage(unittest.TestCase):
    def test_01_version_and_imports(self):
        import deeptensor
        self.assertEqual(deeptensor.__version__, "1.0.0")

        from deeptensor.models import DeepTensor, skip, get_net, SimpleForward2D
        from deeptensor.models.attention import HybridSpectralNet, SpectralTransformer
        from deeptensor.physics import radon, video2codedvideo
        from deeptensor.losses import L1Norm, L2Norm, TVNorm
        from deeptensor.baselines import parafac_als_cuda, lr_decompose
        from deeptensor.training import CosineAnnealingWarmupRestarts, DeepTensorTrainer
        from deeptensor.utils import calculate_psnr, calculate_ssim, get_device

    def test_02_metrics(self):
        from deeptensor.utils import calculate_psnr, calculate_ssim
        img1 = np.ones((64, 64), dtype=np.float32)
        img2 = np.ones((64, 64), dtype=np.float32) * 0.99
        psnr = calculate_psnr(img1, img2, data_range=1.0)
        ssim = calculate_ssim(img1, img2, data_range=1.0)
        self.assertGreater(psnr, 35.0)
        self.assertGreater(ssim, 0.95)

    def test_03_simple_forward_2d(self):
        from deeptensor.models import SimpleForward2D
        model = SimpleForward2D(n_inputs=3, n_outputs=10, nconvs=[16, 32])
        inp = torch.randn(2, 3, 32, 32)
        out = model(inp)
        self.assertEqual(out.shape, (2, 10, 32, 32))

    def test_04_losses(self):
        from deeptensor.losses import L1Norm, TVNorm
        l1 = L1Norm()
        tv = TVNorm()
        t1 = torch.randn(1, 10, 32, 32)
        t2 = torch.randn(1, 10, 32, 32)
        loss1 = l1(t1 - t2)
        loss_tv = tv(t1)
        self.assertGreaterEqual(loss1.item(), 0.0)
        self.assertGreaterEqual(loss_tv.item(), 0.0)

    def test_05_physics_video2coded(self):
        from deeptensor.physics import video2codedvideo
        video = torch.rand(1, 8, 32, 32)
        mask = torch.rand(1, 8, 32, 32)
        coded = video2codedvideo(video, mask, nframes=8)
        self.assertEqual(coded.shape, (1, 1, 32, 32))

    def test_06_datasets_exist(self):
        import scipy.io as io
        data_dir = os.path.join(ROOT, 'data')
        expected_mats = {
            'weizzman.mat': (840, 103, 71),
            'icvl.mat': (174, 164, 260),
            'yale_b.mat': (192, 168, 160),
            'pet.mat': (128, 128, 56),
            'cat_video.mat': (128, 128, 64)
        }
        for filename, expected_shape in expected_mats.items():
            fp = os.path.join(data_dir, filename)
            self.assertTrue(os.path.exists(fp), f"Missing dataset: {filename}")
            mat = io.loadmat(fp)
            found_shape = None
            for k in ['faces', 'hypercube', 'cat_video', 'weizzman']:
                if k in mat:
                    found_shape = mat[k].shape
                    break
            if found_shape is None:
                keys = [k for k in mat.keys() if not k.startswith('__')]
                found_shape = mat[keys[0]].shape
            self.assertEqual(found_shape, expected_shape, f"{filename} expected shape {expected_shape}, got {found_shape}")

    def test_07_parafac_als_cuda_smoke(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        from deeptensor.baselines import parafac_als_cuda
        target = torch.rand(32, 28, 20, device="cuda")
        recon, factors = parafac_als_cuda(target, rank=5, num_iters=3)
        self.assertEqual(recon.shape, (32, 28, 20))
        self.assertEqual(len(factors), 3)

if __name__ == '__main__':
    unittest.main(verbosity=2)
