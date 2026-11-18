import numpy as np
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
