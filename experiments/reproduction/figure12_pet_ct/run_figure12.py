#!/usr/bin/env python

'''
This script replicates the DeepTensor, 2D TV, and Algorithmic Reconstruction results
in Figure 12 of the paper:
"DeepTensor: Low-Rank Tensor Decomposition With Deep Network Priors" (Section 4.1, Page 8-9)

Experiment:
- 3D PET CT Volume Reconstruction from Limited, Noisy Tomographic Measurements
- Dataset: data/pet.mat (H=128, W=128, T=56)
- Model: Rank-1000 PARAFAC (CP) Decomposition
- Forward Model: 40 Tomographic Angles via Radon Transform
- Noise: Poisson (tau=1000) + Gaussian readout (snr=2)
- Baselines:
  1. Algorithmic Reconstruction (Linear Inversion without priors)
  2. 2D Total Variation (TV) Regularization
  3. Rank-1000 DeepTensor (3 x 1D DIP networks with positional encoding)

Outputs faithfully saved in: RESULTS-BASELINE/figure12/
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import tensorly
tensorly.set_backend('pytorch')

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from deeptensor import models, utils, losses, deep_prior, lin_inverse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_metrics(gt, rec):
    data_range = float(gt.max() - gt.min())
    psnr_val = float(utils.psnr(gt, rec))
    # ssim over 3D volume channels (slices along z)
    ssim_val = float(ssim_func(gt, rec, channel_axis=-1, data_range=data_range))
    rmse_val = float(np.sqrt(np.mean((gt - rec) ** 2)))
    return psnr_val, ssim_val, rmse_val

if __name__ == '__main__':
    print('============================================================')
    print('DEEPTENSOR BASELINE: FIGURE 12 EXPERIMENT (3D CT RECONSTRUCTION)')
    print('============================================================')
    print(f'Device: {device}')
    
    out_dir = os.path.join(project_root, 'results', 'baseline', 'figure12_pet_ct')
    os.makedirs(f'{out_dir}/images', exist_ok=True)
    os.makedirs(f'{out_dir}/plots', exist_ok=True)
    os.makedirs(f'{out_dir}/checkpoint', exist_ok=True)
    
    expname = 'pet'
    nangles = 40
    rank = 1000
    scaling = 1
    tau = 1000
    noise_snr = 2
    
    epochs = 2000
    learning_rate = 1e-3
    reg_noise_std = 1.0 / 30.0
    seed = 42
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    # 1. Load Data
    print('\n[1/4] Loading 3D PET Volume...')
    data = io.loadmat(os.path.join(project_root, 'data', f'{expname}.mat'))
    cube = data['hypercube'].astype(np.float32)
    cube = utils.resize(cube / cube.max(), scaling)
    H, W, T = cube.shape
    print(f'  3D PET Volume Dimensions: {H} x {W} x {T} slices')
    
    # 2. Simulate Noisy Radon Tomography Measurements
    print(f'\n[2/4] Simulating Tomographic Measurements across {nangles} angles...')
    cube_ten = torch.tensor(cube, dtype=torch.float32, device=device).permute(2, 0, 1)[None, ...]
    angles = torch.tensor(np.linspace(0, 180, nangles).astype(np.float32), device=device)
    
    clean_measurements = lin_inverse.radon(cube_ten, angles).detach().cpu().numpy()
    noisy_measurements = utils.measure(clean_measurements, noise_snr, tau)
    meas_ten = torch.tensor(noisy_measurements, dtype=torch.float32, device=device)
    sinogram_max = clean_measurements.max()
    print(f'  Sinogram Shape: {clean_measurements.shape}')
    
    # 3. Algorithmic Reconstruction Baseline (No prior)
    print('\n[3/4] Running Algorithmic Reconstruction & 2D TV Baselines...')
    t0_alg = time.time()
    rec_alg = utils.get_inp([H, W, T], device=device)
    opt_alg = torch.optim.Adam([rec_alg], lr=0.1)
    crit_l2 = losses.L2Norm()
    for _ in range(150):
        m_est = lin_inverse.radon(rec_alg[None, ...].permute(0, 3, 1, 2), angles)
        l = crit_l2(meas_ten - m_est)
        opt_alg.zero_grad()
        l.backward()
        opt_alg.step()
    alg_time = time.time() - t0_alg
    cube_alg = rec_alg.detach().cpu().numpy()
    alg_psnr, alg_ssim, alg_rmse = compute_metrics(cube, cube_alg)
    print(f'  Algorithmic Rec: PSNR = {alg_psnr:.2f} dB | SSIM = {alg_ssim:.4f} | RMSE = {alg_rmse:.4f} ({alg_time:.1f}s)')
    
    # 2D TV Baseline
    t0_tv = time.time()
    rec_tv = utils.get_inp([H, W, T], device=device)
    opt_tv = torch.optim.Adam([rec_tv], lr=0.1)
    crit_tv = losses.TVNorm()
    for _ in range(150):
        m_est = lin_inverse.radon(rec_tv[None, ...].permute(0, 3, 1, 2), angles)
        l = crit_l2(meas_ten - m_est) + 1.0 * crit_tv(rec_tv.permute(2, 0, 1)[None, ...])
        opt_tv.zero_grad()
        l.backward()
        opt_tv.step()
    tv_time = time.time() - t0_tv
    cube_tv = rec_tv.detach().cpu().numpy()
    tv_psnr, tv_ssim, tv_rmse = compute_metrics(cube, cube_tv)
    print(f'  2D TV Rec:       PSNR = {tv_psnr:.2f} dB | SSIM = {tv_ssim:.4f} | RMSE = {tv_rmse:.4f} ({tv_time:.1f}s)')
    
    # 4. Rank-1000 DeepTensor PARAFAC Reconstruction
    print(f'\n[4/4] Training Rank-1000 DeepTensor PARAFAC ({epochs} Epochs on CUDA)...')
    u_inp = utils.get_1d_posencode_inp(H, rank // 2, device=device)
    v_inp = utils.get_1d_posencode_inp(W, rank // 2, device=device)
    w_inp = utils.get_1d_posencode_inp(T, rank // 2, device=device)
    
    u_net = deep_prior.get_net(rank, 'skip1d', 'reflection', 'linear',
                               n_channels=rank, skip_n33d=128, skip_n33u=128,
                               num_scales=5).to(device)
    v_net = deep_prior.get_net(rank, 'skip1d', 'reflection', 'linear',
                               n_channels=rank, skip_n33d=128, skip_n33u=128,
                               num_scales=5).to(device)
    w_net = deep_prior.get_net(rank, 'skip1d', 'reflection', 'linear',
                               n_channels=rank, skip_n33d=128, skip_n33u=128,
                               num_scales=5).to(device)
                               
    u_net.train(); v_net.train(); w_net.train()
    
    core = utils.get_inp(rank, device=device)
    with torch.no_grad():
        core[...] = 1.0 / rank
        
    params = list(u_net.parameters()) + list(v_net.parameters()) + list(w_net.parameters()) + [u_inp, v_inp, w_inp, core]
    optimizer = torch.optim.Adam(lr=learning_rate, params=params)
    criterion_loss = losses.L2Norm()
    
    u_inp_per = u_inp.detach().clone()
    v_inp_per = v_inp.detach().clone()
    w_inp_per = w_inp.detach().clone()
    
    best_loss = float('inf')
    best_cube_estim = None
    best_weights = {}
    loss_history = []
    mse_history = []
    
    t0_dt = time.time()
    for epoch in range(epochs):
        u_inp_pert = u_inp + u_inp_per.normal_() * reg_noise_std
        v_inp_pert = v_inp + v_inp_per.normal_() * reg_noise_std
        w_inp_pert = w_inp + w_inp_per.normal_() * reg_noise_std
        
        Uk = u_net(u_inp_pert)
        Vk = v_net(v_inp_pert)
        Wk = w_net(w_inp_pert)
        
        factors = [Uk[0].permute(1, 0), Vk[0].permute(1, 0), Wk[0].permute(1, 0)]
        tensor_estim = tensorly.cp_to_tensor((core, factors))
        
        measurements_estim = lin_inverse.radon(tensor_estim[None, ...].permute(0, 3, 1, 2), angles)
        loss = criterion_loss(meas_ten - measurements_estim)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        cur_loss = loss.item()
        loss_history.append(cur_loss)
        cur_mse = ((tensor_estim.detach().cpu().numpy() - cube) ** 2).mean()
        mse_history.append(cur_mse)
        
        if cur_loss < best_loss:
            best_loss = cur_loss
            best_cube_estim = tensor_estim.detach().clone()
            if epoch % 500 == 0 or epoch == epochs - 1:
                best_weights = {
                    'u_net': copy.deepcopy(u_net.state_dict()),
                    'v_net': copy.deepcopy(v_net.state_dict()),
                    'w_net': copy.deepcopy(w_net.state_dict()),
                    'core': core.detach().clone()
                }
                
        if (epoch + 1) % 250 == 0 or epoch == 0:
            cur_psnr = 10 * np.log10(1.0 / cur_mse)
            print(f'  Epoch [{epoch+1:4d}/{epochs}] | Loss: {cur_loss:.5f} | MSE: {cur_mse:.6f} | Approx PSNR: {cur_psnr:.2f} dB')
            
    dt_time = time.time() - t0_dt
    cube_dt = best_cube_estim.squeeze().cpu().numpy()
    dt_psnr, dt_ssim, dt_rmse = compute_metrics(cube, cube_dt)
    
    print('\n============================================================')
    print('FIGURE 12 FINAL RESULTS')
    print('============================================================')
    print(f'Algorithmic Rec: PSNR = {alg_psnr:.2f} dB | SSIM = {alg_ssim:.4f} | RMSE = {alg_rmse:.4f} | Time = {alg_time:.1f}s')
    print(f'2D TV Rec:       PSNR = {tv_psnr:.2f} dB | SSIM = {tv_ssim:.4f} | RMSE = {tv_rmse:.4f} | Time = {tv_time:.1f}s')
    print(f'DeepTensor:      PSNR = {dt_psnr:.2f} dB | SSIM = {dt_ssim:.4f} | RMSE = {dt_rmse:.4f} | Time = {dt_time:.1f}s')
    print('============================================================')
    print(f'DeepTensor PSNR Gain over Algorithmic: +{dt_psnr - alg_psnr:.2f} dB')
    print(f'DeepTensor PSNR Gain over 2D TV:       +{dt_psnr - tv_psnr:.2f} dB')
    print(f'DeepTensor SSIM Gain over 2D TV:       +{dt_ssim - tv_ssim:.4f}')
    
    # 5. Save Checkpoint
    checkpoint_path = f'{out_dir}/checkpoint/figure12_deeptensor_checkpoint.pt'
    torch.save({
        'weights': best_weights,
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
        'experiment': 'Figure 12: Reconstruction from 3D CT Measurements',
        'dataset': 'data/pet.mat',
        'volume_dimensions': [H, W, T],
        'decomposition_rank': rank,
        'tomographic_angles': nangles,
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
            'algorithmic': {'PSNR': alg_psnr, 'SSIM': alg_ssim, 'RMSE': alg_rmse, 'time': alg_time},
            'tv_2d': {'PSNR': tv_psnr, 'SSIM': tv_ssim, 'RMSE': tv_rmse, 'time': tv_time},
            'deeptensor': {'PSNR': dt_psnr, 'SSIM': dt_ssim, 'RMSE': dt_rmse, 'time': dt_time}
        }
    }
    with open(f'{out_dir}/configuration.json', 'w') as f:
        json.dump(config, f, indent=4)
        
    # 7. Save Numerical Metrics
    with open(f'{out_dir}/metrics.txt', 'w') as f:
        f.write('============================================================\n')
        f.write('DEEPTENSOR BASELINE: FIGURE 12 METRICS SUMMARY\n')
        f.write('============================================================\n')
        f.write('Experiment: Figure 12 (3D PET CT Reconstruction)\n')
        f.write('Dataset: PET CT Scan (data/pet.mat)\n')
        f.write(f'Dimensions: {H} x {W} x {T}\n')
        f.write(f'PARAFAC Rank: {rank}\n')
        f.write(f'Tomographic Angles: {nangles}\n')
        f.write(f'Noise: Poisson (tau={tau}) + Gaussian readout (snr={noise_snr})\n')
        f.write(f'Epochs: {epochs} | Optimizer: Adam (lr={learning_rate})\n')
        f.write(f'Device: {device}\n\n')
        f.write('METHOD                PSNR (dB)     SSIM         RMSE         RUNTIME (s)\n')
        f.write('--------------------------------------------------------------------------\n')
        f.write(f'Algorithmic Rec       {alg_psnr:<13.2f} {alg_ssim:<12.4f} {alg_rmse:<12.4f} {alg_time:.2f}\n')
        f.write(f'2D TV Regularization  {tv_psnr:<13.2f} {tv_ssim:<12.4f} {tv_rmse:<12.4f} {tv_time:.2f}\n')
        f.write(f'DeepTensor (Rank-1000){dt_psnr:<13.2f} {dt_ssim:<12.4f} {dt_rmse:<12.4f} {dt_time:.2f}\n')
        f.write('--------------------------------------------------------------------------\n\n')
        f.write(f'DeepTensor PSNR Gain over Algorithmic: +{dt_psnr - alg_psnr:.2f} dB\n')
        f.write(f'DeepTensor PSNR Gain over 2D TV:       +{dt_psnr - tv_psnr:.2f} dB\n')
        f.write(f'DeepTensor SSIM Gain over 2D TV:       +{dt_ssim - tv_ssim:.4f}\n')
        f.write(f'Final Measurement Loss:                {best_loss:.6e}\n')
        
    # 8. Save Paper-Matched Visual Panel (Center Slice 28 as in Paper Fig. 12)
    slice_idx = 28
    gt_slice = cube[..., slice_idx]
    alg_slice = cube_alg[..., slice_idx]
    tv_slice = cube_tv[..., slice_idx]
    dt_slice = cube_dt[..., slice_idx]
    sino_slice = noisy_measurements[slice_idx]
    
    fig = plt.figure(figsize=(15, 8))
    # Col 1: CT measurements (Sinogram)
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(sino_slice, cmap='gray', aspect='auto')
    ax1.set_title(f'CT Meas. for Slice {slice_idx}\n({nangles} Angles)')
    ax1.set_xlabel('Sensor Detectors'); ax1.set_ylabel('Angles')
    
    # Col 2: Ground Truth
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(gt_slice, cmap='gray')
    ax2.set_title(f'Ground Truth (Slice {slice_idx})')
    ax2.axis('off')
    
    # Col 3: Algorithmic Rec
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(alg_slice, cmap='gray')
    ax3.set_title(f'Algorithmic Rec\nPSNR: {alg_psnr:.1f}dB | SSIM: {alg_ssim:.2f}')
    ax3.axis('off')
    
    # Col 4: 2D TV
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.imshow(tv_slice, cmap='gray')
    ax4.set_title(f'2D TV Regularization\nPSNR: {tv_psnr:.1f}dB | SSIM: {tv_ssim:.2f}')
    ax4.axis('off')
    
    # Col 5: DeepTensor
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.imshow(dt_slice, cmap='gray')
    ax5.set_title(f'DeepTensor (Rank-1000)\nPSNR: {dt_psnr:.1f}dB | SSIM: {dt_ssim:.2f}')
    ax5.axis('off')
    
    # Col 6: DeepTensor Absolute Error Map
    ax6 = fig.add_subplot(2, 3, 6)
    diff_slice = np.abs(gt_slice - dt_slice)
    im6 = ax6.imshow(diff_slice, cmap='hot')
    plt.colorbar(im6, ax=ax6)
    ax6.set_title(f'DeepTensor Abs Diff (Slice {slice_idx})')
    ax6.axis('off')
    
    plt.tight_layout()
    fig.savefig(f'{out_dir}/images/figure12_reconstruction_panel.png', dpi=300)
    plt.close(fig)
    
    # 9. Save Convergence Curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history, color='blue', label='Radon Loss')
    plt.yscale('log')
    plt.title('Measurement L2 Loss vs Epoch')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    psnr_curve = [10 * np.log10(1.0 / m) for m in mse_history]
    plt.plot(psnr_curve, color='green', label='DeepTensor PSNR')
    plt.axhline(alg_psnr, color='gray', linestyle=':', label=f'Algorithmic ({alg_psnr:.1f} dB)')
    plt.axhline(tv_psnr, color='crimson', linestyle='--', label=f'2D TV ({tv_psnr:.1f} dB)')
    plt.title('Reconstruction PSNR vs Epoch')
    plt.xlabel('Epoch'); plt.ylabel('PSNR (dB)'); plt.legend(); plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{out_dir}/plots/figure12_convergence_curves.png', dpi=300)
    plt.close()
    
    # 10. Save Results Array Checkpoint
    np.savez_compressed(
        f'{out_dir}/figure12_results.npz',
        cube_dt=cube_dt,
        cube_tv=cube_tv,
        cube_alg=cube_alg,
        noisy_measurements=noisy_measurements,
        metrics=config['metrics'],
        loss_history=loss_history
    )
    print(f'Figure 12 complete. All artifacts saved to: {out_dir}')
