import os
import shutil

ROOT = os.path.abspath('e:/Rakhal_Tensor')
SRC = os.path.join(ROOT, 'src', 'deeptensor')

# 1. __init__.py
with open(os.path.join(SRC, '__init__.py'), 'w', encoding='utf-8') as f:
    f.write('''"""DeepTensor++: Deep Matrix and Tensor Factorization Package."""

__version__ = "1.0.0"
__author__ = "Rakhal Krishna"

from .models.deeptensor import DeepTensor
from .models.deep_prior import skip, get_net
from .losses.objectives import L1Norm, L2Norm, TVNorm
from .utils.metrics import calculate_psnr, calculate_ssim
from .utils.device import get_device

__all__ = [
    "DeepTensor",
    "skip",
    "get_net",
    "L1Norm",
    "L2Norm",
    "TVNorm",
    "calculate_psnr",
    "calculate_ssim",
    "get_device",
]
''')

# 2. Backbones: copy modules/deep_models/* to src/deeptensor/models/backbones/
src_backbones = os.path.join(SRC, 'models', 'backbones')
mod_backbones = os.path.join(ROOT, 'modules', 'deep_models')
for fn in os.listdir(mod_backbones):
    if fn.endswith('.py'):
        shutil.copy2(os.path.join(mod_backbones, fn), os.path.join(src_backbones, fn))

# Fix imports in backbones:
# In downsampler.py, common.py etc., change relative imports if needed
for fn in os.listdir(src_backbones):
    fp = os.path.join(src_backbones, fn)
    with open(fp, 'r', encoding='utf-8') as f:
        content = f.read()
    # update imports
    content = content.replace('from .common import', 'from .common import')
    content = content.replace('import common', 'from . import common')
    content = content.replace('import downsampler', 'from . import downsampler')
    with open(fp, 'w', encoding='utf-8') as f:
        f.write(content)

# 3. Models
shutil.copy2(os.path.join(ROOT, 'modules', 'models.py'), os.path.join(SRC, 'models', 'deeptensor.py'))
# Update imports in deeptensor.py
with open(os.path.join(SRC, 'models', 'deeptensor.py'), 'r', encoding='utf-8') as f:
    c = f.read()
c = c.replace('from deep_prior import', 'from .deep_prior import')
c = c.replace('import deep_prior', 'from . import deep_prior')
c = c.replace('from deep_decoder import', 'from .deep_decoder import')
c = c.replace('import deep_decoder', 'from . import deep_decoder')
c = c.replace('from utils import', 'from ..utils.visualization import')
with open(os.path.join(SRC, 'models', 'deeptensor.py'), 'w', encoding='utf-8') as f:
    f.write(c)

shutil.copy2(os.path.join(ROOT, 'modules', 'deep_prior.py'), os.path.join(SRC, 'models', 'deep_prior.py'))
with open(os.path.join(SRC, 'models', 'deep_prior.py'), 'r', encoding='utf-8') as f:
    c = f.read()
c = c.replace('from deep_models.skip import skip', 'from .backbones.skip import skip')
c = c.replace('from deep_models import get_net', 'from .backbones import get_net')
c = c.replace('from deep_models.downsampler import Downsampler', 'from .backbones.downsampler import Downsampler')
with open(os.path.join(SRC, 'models', 'deep_prior.py'), 'w', encoding='utf-8') as f:
    f.write(c)

shutil.copy2(os.path.join(ROOT, 'modules', 'deep_decoder.py'), os.path.join(SRC, 'models', 'deep_decoder.py'))

# 4. Attention
shutil.copy2(os.path.join(ROOT, 'modules', 'attention_spectral.py'), os.path.join(SRC, 'models', 'attention', 'spatial_spectral.py'))
shutil.copy2(os.path.join(ROOT, 'modules', 'spectral_transformer.py'), os.path.join(SRC, 'models', 'attention', 'spectral_transformer.py'))
with open(os.path.join(SRC, 'models', 'attention', '__init__.py'), 'w', encoding='utf-8') as f:
    f.write('''from .spatial_spectral import HybridSpectralNet, SpectralSelfAttention, SpectralPointwiseControl
from .spectral_transformer import SpectralTransformer, ControlSpectralTransformer

__all__ = [
    "HybridSpectralNet",
    "SpectralSelfAttention",
    "SpectralPointwiseControl",
    "SpectralTransformer",
    "ControlSpectralTransformer",
]
''')

# models/__init__.py
with open(os.path.join(SRC, 'models', '__init__.py'), 'w', encoding='utf-8') as f:
    f.write('''from .deeptensor import DeepTensor, SimpleForward2D
from .deep_prior import skip, get_net, fill_noise, get_inp
from .deep_decoder import *

__all__ = [
    "DeepTensor",
    "SimpleForward2D",
    "skip",
    "get_net",
    "fill_noise",
    "get_inp",
]
''')

# 5. Physics: Radon & Video CS
shutil.copy2(os.path.join(ROOT, 'modules', 'lin_inverse.py'), os.path.join(SRC, 'physics', 'operators.py'))
with open(os.path.join(SRC, 'physics', 'radon.py'), 'w', encoding='utf-8') as f:
    f.write('''from .operators import radon
__all__ = ["radon"]
''')
with open(os.path.join(SRC, 'physics', 'video_cs.py'), 'w', encoding='utf-8') as f:
    f.write('''from .operators import video2codedvideo, get_video_coding_frames
__all__ = ["video2codedvideo", "get_video_coding_frames"]
''')
with open(os.path.join(SRC, 'physics', '__init__.py'), 'w', encoding='utf-8') as f:
    f.write('''from .operators import radon, video2codedvideo, get_video_coding_frames
__all__ = ["radon", "video2codedvideo", "get_video_coding_frames"]
''')

# 6. Losses
shutil.copy2(os.path.join(ROOT, 'modules', 'losses.py'), os.path.join(SRC, 'losses', 'objectives.py'))
with open(os.path.join(SRC, 'losses', '__init__.py'), 'w', encoding='utf-8') as f:
    f.write('''from .objectives import L1Norm, L2Norm, TVNorm
__all__ = ["L1Norm", "L2Norm", "TVNorm"]
''')

# 7. Baselines
shutil.copy2(os.path.join(ROOT, 'modules', 'spectral.py'), os.path.join(SRC, 'baselines', 'spectral.py'))
# Extract CUDA PARAFAC ALS from run_figure13b_yale_rank_sweep_improved.py into src/deeptensor/baselines/parafac_als.py
with open(os.path.join(SRC, 'baselines', 'parafac_als.py'), 'w', encoding='utf-8') as f:
    f.write('''import torch

def parafac_als_cuda(tensor_target, rank, num_iters=30, verbose=False):
    """
    Exact, vectorized PyTorch CUDA Alternating Least Squares (ALS) for 3-way PARAFAC decomposition.
    tensor_target: (H, W, T) on GPU
    rank: int
    """
    device = tensor_target.device
    H, W, T = tensor_target.shape

    # Unfoldings
    # Mode-0 (H, W*T)
    X0 = tensor_target.reshape(H, W * T)
    # Mode-1 (W, H*T)
    X1 = tensor_target.permute(1, 0, 2).reshape(W, H * T)
    # Mode-2 (T, H*W)
    X2 = tensor_target.permute(2, 0, 1).reshape(T, H * W)

    # Initialization via random normal
    gen = torch.Generator(device=device).manual_seed(42)
    A = torch.randn(H, rank, device=device, generator=gen) * 0.1
    B = torch.randn(W, rank, device=device, generator=gen) * 0.1
    C = torch.randn(T, rank, device=device, generator=gen) * 0.1

    # Normalize initial factors
    A = A / (torch.norm(A, dim=0, keepdim=True) + 1e-12)
    B = B / (torch.norm(B, dim=0, keepdim=True) + 1e-12)
    C = C / (torch.norm(C, dim=0, keepdim=True) + 1e-12)

    def khatri_rao(Mat1, Mat2):
        # Mat1: (I, R), Mat2: (J, R) -> (I*J, R)
        R = Mat1.shape[1]
        return (Mat1.unsqueeze(1) * Mat2.unsqueeze(0)).reshape(-1, R)

    reg = 1e-6 * torch.eye(rank, device=device)

    for it in range(num_iters):
        # Update A: X0 approx A (C (KR) B)^T
        KR_CB = khatri_rao(C, B)
        VtV = (C.T @ C) * (B.T @ B) + reg
        A = torch.linalg.solve(VtV, (X0 @ KR_CB).T).T

        # Update B: X1 approx B (C (KR) A)^T
        KR_CA = khatri_rao(C, A)
        VtV = (C.T @ C) * (A.T @ A) + reg
        B = torch.linalg.solve(VtV, (X1 @ KR_CA).T).T

        # Update C: X2 approx C (B (KR) A)^T
        KR_BA = khatri_rao(B, A)
        VtV = (B.T @ B) * (A.T @ A) + reg
        C = torch.linalg.solve(VtV, (X2 @ KR_BA).T).T

    # Reconstruct tensor: X_hat[h, w, t] = sum_r A[h, r] * B[w, r] * C[t, r]
    # Vectorized: einsum 'hr, wr, tr -> hwt'
    X_hat = torch.einsum('hr,wr,tr->hwt', A, B, C)
    return X_hat, (A, B, C)
''')

with open(os.path.join(SRC, 'baselines', '__init__.py'), 'w', encoding='utf-8') as f:
    f.write('''from .spectral import lr_decompose, tucker_decompose
from .parafac_als import parafac_als_cuda

__all__ = ["lr_decompose", "tucker_decompose", "parafac_als_cuda"]
''')

# 8. Training
shutil.copy2(os.path.join(ROOT, 'modules', 'cosine_annealing_with_warmup.py'), os.path.join(SRC, 'training', 'schedulers.py'))
with open(os.path.join(SRC, 'training', 'trainer.py'), 'w', encoding='utf-8') as f:
    f.write('''"""DeepTensor Unified Training Utilities."""
import torch
import time

class DeepTensorTrainer:
    """Helper class managing optimization loops and closures."""
    def __init__(self, model, optimizer, criterion, scheduler=None, device="cuda"):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def step(self, closure):
        return self.optimizer.step(closure)
''')
with open(os.path.join(SRC, 'training', '__init__.py'), 'w', encoding='utf-8') as f:
    f.write('''from .schedulers import CosineAnnealingWarmupRestarts
from .trainer import DeepTensorTrainer
__all__ = ["CosineAnnealingWarmupRestarts", "DeepTensorTrainer"]
''')

# 9. Utils
shutil.copy2(os.path.join(ROOT, 'modules', 'utils.py'), os.path.join(SRC, 'utils', 'utils_legacy.py'))
# Device helper
with open(os.path.join(SRC, 'utils', 'device.py'), 'w', encoding='utf-8') as f:
    f.write('''import torch

def get_device(requested=None):
    """Resolve compute device safely (CUDA or CPU fallback)."""
    if requested and requested != "auto":
        if "cuda" in str(requested).lower() and torch.cuda.is_available():
            return torch.device(requested)
        return torch.device("cpu")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
''')

# Metrics
with open(os.path.join(SRC, 'utils', 'metrics.py'), 'w', encoding='utf-8') as f:
    f.write('''import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from skimage.metrics import structural_similarity as ssim_func

def calculate_psnr(gt, pred, data_range=None):
    """Compute PSNR between ground truth and prediction."""
    gt_np = gt.detach().cpu().numpy() if hasattr(gt, "detach") else np.asarray(gt)
    pred_np = pred.detach().cpu().numpy() if hasattr(pred, "detach") else np.asarray(pred)
    if data_range is None:
        data_range = float(max(gt_np.max() - gt_np.min(), 1.0))
    return float(psnr_func(gt_np, pred_np, data_range=data_range))

def calculate_ssim(gt, pred, data_range=None, channel_axis=None):
    """Compute SSIM between ground truth and prediction."""
    gt_np = gt.detach().cpu().numpy() if hasattr(gt, "detach") else np.asarray(gt)
    pred_np = pred.detach().cpu().numpy() if hasattr(pred, "detach") else np.asarray(pred)
    if data_range is None:
        data_range = float(max(gt_np.max() - gt_np.min(), 1.0))
    try:
        return float(ssim_func(gt_np, pred_np, data_range=data_range, channel_axis=channel_axis))
    except TypeError:
        # Fallback for older skimage versions with multichannel parameter
        return float(ssim_func(gt_np, pred_np, data_range=data_range, multichannel=(channel_axis is not None)))
''')

# Visualization
with open(os.path.join(SRC, 'utils', 'visualization.py'), 'w', encoding='utf-8') as f:
    f.write('''from .utils_legacy import build_montage, resize, measure, get_inp
__all__ = ["build_montage", "resize", "measure", "get_inp"]
''')

with open(os.path.join(SRC, 'utils', '__init__.py'), 'w', encoding='utf-8') as f:
    f.write('''from .device import get_device
from .metrics import calculate_psnr, calculate_ssim
from .visualization import build_montage, resize, measure, get_inp

__all__ = [
    "get_device",
    "calculate_psnr",
    "calculate_ssim",
    "build_montage",
    "resize",
    "measure",
    "get_inp",
]
''')

print("src/deeptensor package populated successfully.")

