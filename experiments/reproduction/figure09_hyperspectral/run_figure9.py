#!/usr/bin/env python

'''
This script replicates the DeepTensor, SVD, and BM3D results in Figure 9 of the paper:
"DeepTensor: Low-Rank Tensor Decomposition With Deep Network Priors" (Section 4.1, Page 8)

Experiment:
- Hyperspectral Image Denoising on ICVL dataset (data/icvl.mat)
- Downsampled to (174, 164, 260)
- Noise: Poisson (tau=100) + Gaussian readout (snr=2) -> Input PSNR ~ 28.43 dB
- Baselines:
  1. Rank-20 SVD (Truncated Singular Value Decomposition)
  2. BM3D Hyperspectral Denoising
  3. Rank-20 DeepTensor (2D DIP spatial + 1D DIP spectral, 5000 epochs)

Outputs faithfully saved in: RESULTS-BASELINE/figure9/
'''

import os
import sys
import copy
import json
import time
import argparse
import numpy as np
from scipy import io
from skimage.metrics import structural_similarity as ssim_func
import bm3d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from deeptensor import models, utils, losses, deep_prior
from deeptensor.baselines import spectral

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_metrics(gt, rec):
    data_range = float(gt.max() - gt.min())
    psnr_val = float(utils.psnr(gt, rec))
    ssim_val = float(ssim_func(gt, rec, channel_axis=-1, data_range=data_range))
    rmse_val = float(np.sqrt(np.mean((gt - rec) ** 2)))
    return psnr_val, ssim_val, rmse_val

if __name__ == '__main__':
    print('============================================================')
    print('DEEPTENSOR BASELINE: FIGURE 9 EXPERIMENT (HYPERSPECTRAL DENOISING)')
    print('============================================================')
    print(f'Device: {device}')
    
    out_dir = os.path.join(project_root, 'results', 'baseline', 'figure09_hyperspectral')
    os.makedirs(f'{out_dir}/images', exist_ok=True)
    os.makedirs(f'{out_dir}/plots', exist_ok=True)
    os.makedirs(f'{out_dir}/checkpoint', exist_ok=True)
    
    expname = 'icvl'
    rank = 20
    scaling = 1
    tau = 100
    noise_snr = 2
    
    learning_rate = 1e-3
    epochs = 5000
    reg_noise_std = 1.0 / 30.0
    exp_weight = 0.99
    seed = 42
    
    # 1. Load Data
    print('\n[1/4] Loading ICVL Hyperspectral Dataset...')
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    data = io.loadmat(os.path.join(project_root, 'data', f'{expname}.mat'))
    cube = data['hypercube'].astype(np.float32)
    wavelengths = data['wavelengths'].astype(np.float32).ravel()
    cube = utils.resize(cube / cube.max(), scaling)
    H, W, nwvl = cube.shape
    print(f'  Dataset Shape: {H} x {W} x {nwvl} channels')
    
    # Generate Noisy Observation
    cube_noisy = utils.measure(cube, noise_snr, tau)
    noisy_psnr, noisy_ssim, noisy_rmse = compute_metrics(cube, cube_noisy)
    print(f'  Input Noisy PSNR: {noisy_psnr:.2f} dB | SSIM: {noisy_ssim:.4f} | RMSE: {noisy_rmse:.4f}')
    
    # 2. SVD Rank-20 Baseline
    print('\n[2/4] Running Rank-20 SVD Baseline...')
    t0_svd = time.time()
    cube_svd = spectral.lr_decompose(cube_noisy, rank)
    svd_time = time.time() - t0_svd
    svd_psnr, svd_ssim, svd_rmse = compute_metrics(cube, cube_svd)
    print(f'  SVD Rank-20: PSNR = {svd_psnr:.2f} dB | SSIM = {svd_ssim:.4f} | RMSE = {svd_rmse:.4f} ({svd_time:.2f}s)')
    
    # 3. BM3D Baseline
    print('\n[3/4] Running BM3D Baseline across spectral channels...')
    t0_bm3d = time.time()
    sigma_psd = int(np.sqrt(tau)) / 255.0
    cube_bm3d = np.zeros_like(cube_noisy)
    for idx in range(nwvl):
        cube_bm3d[..., idx] = bm3d.bm3d(
            cube_noisy[..., idx],
            sigma_psd=sigma_psd,
            stage_arg=bm3d.BM3DStages.ALL_STAGES
        )
    bm3d_time = time.time() - t0_bm3d
    bm3d_psnr, bm3d_ssim, bm3d_rmse = compute_metrics(cube, cube_bm3d)
    print(f'  BM3D:        PSNR = {bm3d_psnr:.2f} dB | SSIM = {bm3d_ssim:.4f} | RMSE = {bm3d_rmse:.4f} ({bm3d_time:.2f}s)')
    
    # 4. DeepTensor Rank-20 Baseline
    print(f'\n[4/4] Training DeepTensor Rank-20 ({epochs} Epochs on CUDA)...')
    hsten = torch.tensor(cube.reshape(H * W, nwvl), dtype=torch.float32, device=device)[None, ...]
    hsten_noisy = torch.tensor(cube_noisy.reshape(H * W, nwvl), dtype=torch.float32, device=device)[None, ...]
    
    im_inp = utils.get_inp([1, rank, H, W], device=device)
    spec_inp = utils.get_inp([1, rank, nwvl], device=device)
    
    im_net = deep_prior.get_net(rank, 'skip', 'reflection', upsample_mode='bilinear',
                                skip_n33d=128, skip_n33u=128, num_scales=5,
                                n_channels=rank).to(device)
    spec_net = deep_prior.get_net(rank, 'skip1d', 'reflection', upsample_mode='linear',
                                  skip_n33d=128, skip_n33u=128, num_scales=5,
                                  n_channels=rank).to(device)
    im_net.train()
    spec_net.train()
    
    optimizer = torch.optim.Adam(
        list(im_net.parameters()) + list(spec_net.parameters()) + [im_inp, spec_inp],
        lr=learning_rate
    )
    criterion_l1 = losses.L1Norm()
    
    im_inp_per = im_inp.detach().clone()
    spec_inp_per = spec_inp.detach().clone()
    
    loss_history = []
    mse_history = []
    best_loss = float('inf')
    best_hs_estim = None
    best_im_net_state = None
    best_spec_net_state = None
    
    t0_dt = time.time()
    for epoch in range(epochs):
        im_inp_perturbed = im_inp + im_inp_per.normal_() * reg_noise_std
        spec_inp_perturbed = spec_inp + spec_inp_per.normal_() * reg_noise_std
        
        U_img = im_net(im_inp_perturbed)
        V = spec_net(spec_inp_perturbed)
        U = U_img.reshape(-1, rank, H * W).permute(0, 2, 1)
        
        hs_estim = torch.bmm(U, V)
        loss = criterion_l1(hsten_noisy - hs_estim)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        cur_loss = loss.item()
        loss_history.append(cur_loss)
        cur_mse = ((hs_estim - hsten) ** 2).mean().item()
        mse_history.append(cur_mse)
        
        if cur_loss < best_loss:
            best_loss = cur_loss
            best_hs_estim = hs_estim.detach().clone()
            if epoch % 500 == 0 or epoch == epochs - 1:
                best_im_net_state = copy.deepcopy(im_net.state_dict())
                best_spec_net_state = copy.deepcopy(spec_net.state_dict())
                
        if (epoch + 1) % 500 == 0 or epoch == 0:
            cur_psnr = 10 * np.log10(1.0 / cur_mse)
            print(f'  Epoch [{epoch+1:4d}/{epochs}] | Loss: {cur_loss:.5f} | MSE: {cur_mse:.6f} | Approx PSNR: {cur_psnr:.2f} dB')
            
    dt_time = time.time() - t0_dt
    cube_dt = best_hs_estim.squeeze().cpu().numpy().reshape(H, W, nwvl)
    dt_psnr, dt_ssim, dt_rmse = compute_metrics(cube, cube_dt)
    
    print('\n============================================================')
    print('FIGURE 9 FINAL RESULTS')
    print('============================================================')
    print(f'Input Noisy:   PSNR = {noisy_psnr:.2f} dB | SSIM = {noisy_ssim:.4f} | RMSE = {noisy_rmse:.4f}')
    print(f'Rank-20 SVD:   PSNR = {svd_psnr:.2f} dB | SSIM = {svd_ssim:.4f} | RMSE = {svd_rmse:.4f} | Time = {svd_time:.1f}s')
    print(f'BM3D:          PSNR = {bm3d_psnr:.2f} dB | SSIM = {bm3d_ssim:.4f} | RMSE = {bm3d_rmse:.4f} | Time = {bm3d_time:.1f}s')
    print(f'DeepTensor:    PSNR = {dt_psnr:.2f} dB | SSIM = {dt_ssim:.4f} | RMSE = {dt_rmse:.4f} | Time = {dt_time:.1f}s')
    print('============================================================')
    print(f'DeepTensor Gain over SVD:  +{dt_psnr - svd_psnr:.2f} dB')
    print(f'DeepTensor Gain over BM3D: +{dt_psnr - bm3d_psnr:.2f} dB')
    
    # 5. Save Checkpoint
    checkpoint_path = f'{out_dir}/checkpoint/figure9_deeptensor_checkpoint.pt'
    torch.save({
        'im_net': best_im_net_state if best_im_net_state is not None else im_net.state_dict(),
        'spec_net': best_spec_net_state if best_spec_net_state is not None else spec_net.state_dict(),
        'rank': rank,
        'metrics': {
            'PSNR': dt_psnr,
            'SSIM': dt_ssim,
            'RMSE': dt_rmse
        },
        'epochs': epochs
    }, checkpoint_path)
    print(f'Checkpoint saved to: {checkpoint_path}')
    
    # 6. Save Configuration
    config = {
        'experiment': 'Figure 9: Hyperspectral Image Denoising',
        'dataset': 'data/icvl.mat',
        'image_shape': [H, W, nwvl],
        'rank': rank,
        'noise_configuration': {
            'noise_snr': noise_snr,
            'tau': tau,
            'model': 'Poisson + Gaussian readout'
        },
        'optimizer': 'Adam',
        'learning_rate': learning_rate,
        'epochs': epochs,
        'device': str(device),
        'metrics': {
            'noisy': {'PSNR': noisy_psnr, 'SSIM': noisy_ssim, 'RMSE': noisy_rmse},
            'svd': {'PSNR': svd_psnr, 'SSIM': svd_ssim, 'RMSE': svd_rmse, 'time': svd_time},
            'bm3d': {'PSNR': bm3d_psnr, 'SSIM': bm3d_ssim, 'RMSE': bm3d_rmse, 'time': bm3d_time},
            'deeptensor': {'PSNR': dt_psnr, 'SSIM': dt_ssim, 'RMSE': dt_rmse, 'time': dt_time}
        }
    }
    with open(f'{out_dir}/configuration.json', 'w') as f:
        json.dump(config, f, indent=4)
        
    # 7. Save Numerical Metrics
    with open(f'{out_dir}/metrics.txt', 'w') as f:
        f.write('============================================================\n')
        f.write('DEEPTENSOR BASELINE: FIGURE 9 METRICS SUMMARY\n')
        f.write('============================================================\n')
        f.write('Experiment: Figure 9 (Hyperspectral Denoising)\n')
        f.write('Dataset: ICVL Hyperspectral Image (data/icvl.mat)\n')
        f.write(f'Dimensions: {H} x {W} x {nwvl}\n')
        f.write(f'Rank: {rank}\n')
        f.write(f'Noise: Poisson (tau={tau}) + Gaussian readout (snr={noise_snr})\n')
        f.write(f'Epochs: {epochs} | Optimizer: Adam (lr={learning_rate})\n')
        f.write(f'Device: {device}\n\n')
        f.write('METHOD          PSNR (dB)     SSIM         RMSE         RUNTIME (s)\n')
        f.write('--------------------------------------------------------------------\n')
        f.write(f'Noisy Input     {noisy_psnr:<13.2f} {noisy_ssim:<12.4f} {noisy_rmse:<12.4f} -\n')
        f.write(f'Rank-20 SVD     {svd_psnr:<13.2f} {svd_ssim:<12.4f} {svd_rmse:<12.4f} {svd_time:.2f}\n')
        f.write(f'BM3D            {bm3d_psnr:<13.2f} {bm3d_ssim:<12.4f} {bm3d_rmse:<12.4f} {bm3d_time:.2f}\n')
        f.write(f'DeepTensor      {dt_psnr:<13.2f} {dt_ssim:<12.4f} {dt_rmse:<12.4f} {dt_time:.2f}\n')
        f.write('--------------------------------------------------------------------\n\n')
        f.write(f'DeepTensor PSNR Gain over SVD:  +{dt_psnr - svd_psnr:.2f} dB\n')
        f.write(f'DeepTensor PSNR Gain over BM3D: +{dt_psnr - bm3d_psnr:.2f} dB\n')
        f.write(f'DeepTensor SSIM Gain over SVD:  +{dt_ssim - svd_ssim:.4f}\n')
        f.write(f'Final L1 Reconstruction Loss:   {best_loss:.6e}\n')
        
    # 8. Save Paper-Matched Visual Panel (Figure 9 in paper)
    # Target wavelength: ~710 nm corresponds approximately to index 130
    wvl_idx = 130 if nwvl > 130 else nwvl // 2
    
    # Generate RGB representation using 3 bands
    rgb_r = cube[..., int(nwvl * 0.7)]
    rgb_g = cube[..., int(nwvl * 0.45)]
    rgb_b = cube[..., int(nwvl * 0.2)]
    rgb_img = np.stack([rgb_r, rgb_g, rgb_b], axis=-1)
    rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-8)
    
    gt_band = cube[..., wvl_idx]
    noisy_band = cube_noisy[..., wvl_idx]
    svd_band = cube_svd[..., wvl_idx]
    bm3d_band = cube_bm3d[..., wvl_idx]
    dt_band = cube_dt[..., wvl_idx]
    
    diff_svd_50x = np.clip(np.abs(gt_band - svd_band) * 50.0, 0, 1)
    diff_bm3d_50x = np.clip(np.abs(gt_band - bm3d_band) * 50.0, 0, 1)
    diff_dt_50x = np.clip(np.abs(gt_band - dt_band) * 50.0, 0, 1)
    
    fig = plt.figure(figsize=(15, 10))
    # Row 1: RGB, Ground Truth, Noisy
    ax1 = fig.add_subplot(3, 3, 1)
    ax1.imshow(rgb_img); ax1.set_title('(a) Pseudo-RGB Image'); ax1.axis('off')
    ax2 = fig.add_subplot(3, 3, 2)
    ax2.imshow(gt_band, cmap='gray'); ax2.set_title(f'(b) Ground Truth ({wavelengths[wvl_idx]:.0f}nm)'); ax2.axis('off')
    ax3 = fig.add_subplot(3, 3, 3)
    ax3.imshow(noisy_band, cmap='gray'); ax3.set_title(f'(c) Noisy Input (PSNR: {noisy_psnr:.1f}dB)'); ax3.axis('off')
    
    # Row 2: DeepTensor & Abs Diff
    ax4 = fig.add_subplot(3, 3, 4)
    ax4.imshow(dt_band, cmap='gray'); ax4.set_title(f'(d) Rank-20 DeepTensor\nPSNR: {dt_psnr:.1f}dB | SSIM: {dt_ssim:.3f}'); ax4.axis('off')
    ax5 = fig.add_subplot(3, 3, 5)
    ax5.imshow(diff_dt_50x, cmap='hot'); ax5.set_title('(e) DeepTensor Abs Diff (50x)'); ax5.axis('off')
    
    # Row 3: SVD & Abs Diff, BM3D & Abs Diff
    ax6 = fig.add_subplot(3, 3, 6)
    ax6.imshow(svd_band, cmap='gray'); ax6.set_title(f'(f) Rank-20 SVD\nPSNR: {svd_psnr:.1f}dB | SSIM: {svd_ssim:.3f}'); ax6.axis('off')
    ax7 = fig.add_subplot(3, 3, 7)
    ax7.imshow(diff_svd_50x, cmap='hot'); ax7.set_title('(g) SVD Abs Diff (50x)'); ax7.axis('off')
    ax8 = fig.add_subplot(3, 3, 8)
    ax8.imshow(bm3d_band, cmap='gray'); ax8.set_title(f'(h) BM3D\nPSNR: {bm3d_psnr:.1f}dB | SSIM: {bm3d_ssim:.3f}'); ax8.axis('off')
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.imshow(diff_bm3d_50x, cmap='hot'); ax9.set_title('(i) BM3D Abs Diff (50x)'); ax9.axis('off')
    
    plt.tight_layout()
    fig.savefig(f'{out_dir}/images/figure9_hyperspectral_denoising_panel.png', dpi=300)
    plt.close(fig)
    
    # 9. Save Convergence Curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history, color='blue', label='L1 Loss')
    plt.title('Training L1 Loss vs Epoch')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    psnr_curve = [10 * np.log10(1.0 / m) for m in mse_history]
    plt.plot(psnr_curve, color='green', label='DeepTensor PSNR')
    plt.axhline(svd_psnr, color='crimson', linestyle='--', label=f'SVD ({svd_psnr:.1f} dB)')
    plt.axhline(bm3d_psnr, color='orange', linestyle=':', label=f'BM3D ({bm3d_psnr:.1f} dB)')
    plt.title('Convergence PSNR vs Epoch')
    plt.xlabel('Epoch'); plt.ylabel('PSNR (dB)'); plt.legend(); plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{out_dir}/plots/figure9_convergence_curves.png', dpi=300)
    plt.close()
    
    # 10. Save Results Array Checkpoint
    np.savez_compressed(
        f'{out_dir}/figure9_results.npz',
        cube_dt=cube_dt,
        cube_svd=cube_svd,
        cube_bm3d=cube_bm3d,
        cube_noisy=cube_noisy,
        metrics=config['metrics'],
        loss_history=loss_history
    )
    print(f'Figure 9 complete. All artifacts saved to: {out_dir}')
