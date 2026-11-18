import sys, os
sys.path.insert(0, os.path.abspath('.'))
import torch
import numpy as np
from scipy import io
from deeptensor import utils
from deeptensor.models.factory import build_deeptensor_model
from deeptensor.utils.comparison import compute_metrics

data_path = 'data/icvl.mat'
data = io.loadmat(data_path)
cube = data['hypercube'].astype(np.float32)
cube = utils.resize(cube / cube.max(), 1)
H, W, nwvl = cube.shape

ckpt_path = 'results/extensions/adaptive_hybrid_v1_full/figure09_hyperspectral/best_checkpoint.pth'
ckpt = torch.load(ckpt_path, map_location='cuda', weights_only=False)

model = build_deeptensor_model(
    rank=20,
    seq_len=nwvl,
    spatial_type='skip',
    spectral_type='adaptive_hybrid',
    n_inputs=20,
    d_model=128,
    n_heads=4,
    n_layers=4,
    d_ffn=512,
    router_type='cagc',
    pe_type='sin_pe',
).to('cuda')
model.load_state_dict(ckpt['model_state_dict'])

im_inp = ckpt['config']['im_inp'].to('cuda')
spec_inp = ckpt['config']['spec_inp'].to('cuda')

# Let's test eval mode vs train mode, and with perturbation vs without
for mode in ['eval', 'train']:
    if mode == 'eval':
        model.eval()
    else:
        model.train()
    
    with torch.no_grad():
        out_clean = model(im_inp, spec_inp)
        recon_clean = out_clean[0].cpu().numpy().reshape(H, W, nwvl)
        m_clean = compute_metrics(cube, recon_clean)
        print(f"[{mode}] Clean input: PSNR={m_clean['psnr']:.4f} dB, SSIM={m_clean['ssim']:.4f}, RMSE={m_clean['rmse']:.6f}")

        # With small noise
        reg_noise_std = 1.0 / 30.0
        im_pert = im_inp + torch.randn_like(im_inp) * reg_noise_std
        spec_pert = spec_inp + torch.randn_like(spec_inp) * reg_noise_std
        out_pert = model(im_pert, spec_pert)
        recon_pert = out_pert[0].cpu().numpy().reshape(H, W, nwvl)
        m_pert = compute_metrics(cube, recon_pert)
        print(f"[{mode}] Perturbed input: PSNR={m_pert['psnr']:.4f} dB, SSIM={m_pert['ssim']:.4f}, RMSE={m_pert['rmse']:.6f}")
