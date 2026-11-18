import sys
import torch

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")

try:
    import deeptensor
    print(f"Imported deeptensor version {deeptensor.__version__}")
    
    from deeptensor.models import DeepTensor, skip, get_net
    print("Imported deeptensor.models (DeepTensor, skip, get_net)")
    
    from deeptensor.models.attention import HybridSpectralNet, SpectralTransformer
    print("Imported deeptensor.models.attention (HybridSpectralNet, SpectralTransformer)")
    
    from deeptensor.losses import L1Norm, L2Norm, TVNorm
    print("Imported deeptensor.losses (L1Norm, L2Norm, TVNorm)")
    
    from deeptensor.physics import radon, video2codedvideo
    print("Imported deeptensor.physics (radon, video2codedvideo)")
    
    from deeptensor.baselines import parafac_als_cuda, lr_decompose
    print("Imported deeptensor.baselines (parafac_als_cuda, lr_decompose)")
    
    from deeptensor.utils import calculate_psnr, calculate_ssim, get_device
    print(f"Imported deeptensor.utils, device resolved to: {get_device()}")
    
    print("\nALL PACKAGE IMPORTS PASSED SUCCESSFULLY!")
except Exception as e:
    print(f"FAILED IMPORT: {e}")
    import traceback
    traceback.print_exc()

