import unittest
import torch
import numpy as np

class TestDeeptensorImports(unittest.TestCase):
    def test_deeptensor_imports(self):
        import deeptensor
        self.assertTrue(hasattr(deeptensor, "__version__"))
        self.assertEqual(deeptensor.__version__, "1.0.0")

        from deeptensor.models import DeepTensor, skip, get_net, SimpleForward2D
        from deeptensor.models.attention import HybridSpectralNet, SpectralTransformer
        from deeptensor.physics import radon, video2codedvideo
        from deeptensor.losses import L1Norm, L2Norm, TVNorm
        from deeptensor.baselines import parafac_als_cuda, lr_decompose
        from deeptensor.training import CosineAnnealingWarmupRestarts, DeepTensorTrainer
        from deeptensor.utils import calculate_psnr, calculate_ssim, get_device

    def test_psnr_ssim_calculation(self):
        from deeptensor.utils import calculate_psnr, calculate_ssim
        img1 = np.ones((64, 64), dtype=np.float32)
        img2 = np.ones((64, 64), dtype=np.float32) * 0.99
        psnr = calculate_psnr(img1, img2, data_range=1.0)
        ssim = calculate_ssim(img1, img2, data_range=1.0)
        self.assertGreater(psnr, 35.0)
        self.assertGreater(ssim, 0.95)

if __name__ == "__main__":
    unittest.main()
