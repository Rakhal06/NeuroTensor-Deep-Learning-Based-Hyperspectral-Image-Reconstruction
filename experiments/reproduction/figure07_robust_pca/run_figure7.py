#!/usr/bin/env python

'''
This script replicates results in Figure 7 of the DeepTensor research paper:
"DeepTensor: Low-Rank Tensor Decomposition With Deep Network Priors" (Section 3.2, Page 5)

Experiment:
Principal Component Analysis (PCA) under varying sample sizes and noise levels:
- Feature dimension: 64, Intrinsic dimension (rank): 10
- Sample sizes: [32, 64, 128, 256, 512, 1024]
- Noise standard deviations (sigma): [0.05, 0.1, 0.2, 0.5, 1.0]

Evaluation:
- SVD vs DeepTensor on Sample Covariance Matrix
- Reconstruction PSNR against Clean Ground-Truth Covariance Matrix

Outputs saved to: RESULTS-BASELINE/figure7/
'''

import os
import sys
import json
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from deeptensor import models, utils, losses, deep_prior

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_deeptensor_pca(covmat, covmat_gt, nrows, rank, epochs=500, lr=1e-4):
    '''
    Fit sample covariance matrix using DeepTensor 1D DIP network: U U^T ~ Cov
    '''
    covmat_ten = torch.tensor(covmat, dtype=torch.float32, device=device)[None, ...]
    covmat_gt_ten = torch.tensor(covmat_gt, dtype=torch.float32, device=device)[None, ...]
    
    u_inp = utils.get_inp([1, rank, nrows], device=device)
    u_net = deep_prior.get_net(rank, 'skip1d', 'reflection', 'linear',
                               n_channels=rank, skip_n33d=128, skip_n33u=128,
                               num_scales=5).to(device)
                               
    optimizer = torch.optim.Adam(list(u_net.parameters()) + [u_inp], lr=lr)
    criterion = losses.L2Norm()
    
    best_loss = float('inf')
    best_mat = None
    loss_curve = []
    
    for epoch in range(epochs):
        u_out = u_net(u_inp)
        covmat_estim = torch.bmm(u_out.permute(0, 2, 1), u_out)
        loss = criterion(covmat_estim - covmat_ten)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_val = loss.item()
        loss_curve.append(loss_val)
        if loss_val < best_loss:
            best_loss = loss_val
            best_mat = covmat_estim.detach().squeeze().cpu().numpy()
            
    dt_psnr = utils.psnr(covmat_gt, best_mat)
    return best_mat, dt_psnr, best_loss, loss_curve

if __name__ == '__main__':
    print('============================================================')
    print('DEEPTENSOR BASELINE: FIGURE 7 EXPERIMENT')
    print('============================================================')
    print(f'Device: {device}')
    
    out_dir = os.path.join(project_root, 'results', 'baseline', 'figure07_robust_pca')
    os.makedirs(f'{out_dir}/images', exist_ok=True)
    os.makedirs(f'{out_dir}/plots', exist_ok=True)
    
    nrows = 64
    rank = 10
    epochs = 400
    learning_rate = 1e-4
    
    # 1. Sweep sample size at fixed high noise (sigma = 0.5)
    sample_sizes = [32, 64, 128, 256, 512, 1024]
    fixed_sigma = 0.5
    
    # 2. Sweep noise level at small sample size (N = 32) and large sample size (N = 1024)
    noise_sigmas = [0.05, 0.1, 0.2, 0.5, 1.0]
    
    print('\n[Part 1] Evaluating PCA Covariance Recovery vs Sample Size (sigma=0.5)...')
    sample_size_results = {
        'samples': sample_sizes,
        'svd_psnr': [],
        'deeptensor_psnr': [],
        'noisy_cov_psnr': []
    }
    
    rep_sample_images = {}
    
    total_tic = time.time()
    
    for ndata in sample_sizes:
        np.random.seed(100 + ndata)
        torch.manual_seed(100 + ndata)
        
        mat, mat_gt, basis = utils.get_pca(
            nrows, ndata, rank,
            noise_type='gaussian',
            signal_type='gaussian',
            noise_snr=fixed_sigma,
            tau=1000
        )
        
        mat_centered = mat - mat.mean(0).reshape(1, ndata)
        mat_gt_centered = mat_gt - mat_gt.mean(0).reshape(1, ndata)
        
        covmat = mat_centered.dot(mat_centered.T)
        covmat_gt = mat_gt_centered.dot(mat_gt_centered.T)
        
        minval = covmat_gt.min()
        maxval = covmat_gt.max()
        denom = maxval - minval if (maxval - minval) > 1e-8 else 1.0
        covmat = (covmat - minval) / denom
        covmat_gt = (covmat_gt - minval) / denom
        
        noisy_cov_p = utils.psnr(covmat_gt, covmat)
        
        # SVD baseline
        covmat_svd = utils.lr_decompose(covmat, rank)
        svd_p = utils.psnr(covmat_gt, covmat_svd)
        
        # DeepTensor
        t0 = time.time()
        covmat_dt, dt_p, best_l, l_curve = train_deeptensor_pca(
            covmat, covmat_gt, nrows, rank, epochs=epochs, lr=learning_rate
        )
        dt_time = time.time() - t0
        
        sample_size_results['svd_psnr'].append(float(svd_p))
        sample_size_results['deeptensor_psnr'].append(float(dt_p))
        sample_size_results['noisy_cov_psnr'].append(float(noisy_cov_p))
        
        gain = dt_p - svd_p
        print(f'  Samples: {ndata:4d} | Noisy: {noisy_cov_p:.2f} dB | SVD: {svd_p:.2f} dB | DeepTensor: {dt_p:.2f} dB | Gain: {gain:+.2f} dB ({dt_time:.1f}s)')
        
        if ndata in [32, 1024]:
            rep_sample_images[ndata] = {
                'cov_gt': covmat_gt,
                'cov_noisy': covmat,
                'cov_svd': covmat_svd,
                'cov_dt': covmat_dt
            }
            
    print('\n[Part 2] Evaluating PCA Covariance Recovery vs Noise Level (M=32 samples)...')
    noise_results_32 = {
        'sigmas': noise_sigmas,
        'svd_psnr': [],
        'deeptensor_psnr': []
    }
    for sigma in noise_sigmas:
        np.random.seed(200 + int(sigma*100))
        torch.manual_seed(200 + int(sigma*100))
        
        mat, mat_gt, basis = utils.get_pca(
            nrows, 32, rank,
            noise_type='gaussian',
            signal_type='gaussian',
            noise_snr=sigma,
            tau=1000
        )
        mat_c = mat - mat.mean(0).reshape(1, 32)
        mat_gt_c = mat_gt - mat_gt.mean(0).reshape(1, 32)
        covmat = mat_c.dot(mat_c.T)
        covmat_gt = mat_gt_c.dot(mat_gt_c.T)
        
        minval = covmat_gt.min()
        maxval = covmat_gt.max()
        denom = maxval - minval if (maxval - minval) > 1e-8 else 1.0
        covmat = (covmat - minval) / denom
        covmat_gt = (covmat_gt - minval) / denom
        
        covmat_svd = utils.lr_decompose(covmat, rank)
        svd_p = utils.psnr(covmat_gt, covmat_svd)
        
        covmat_dt, dt_p, _, _ = train_deeptensor_pca(
            covmat, covmat_gt, nrows, rank, epochs=epochs, lr=learning_rate
        )
        
        noise_results_32['svd_psnr'].append(float(svd_p))
        noise_results_32['deeptensor_psnr'].append(float(dt_p))
        gain = dt_p - svd_p
        print(f'  Sigma: {sigma:.2f} | SVD: {svd_p:.2f} dB | DeepTensor: {dt_p:.2f} dB | Gain: {gain:+.2f} dB')
        
    total_time = time.time() - total_tic
    print(f'\nFigure 7 completed in {total_time:.1f} seconds.')
    
    # Save configuration
    config = {
        'experiment': 'Figure 7: DeepTensor PCA under Low SNR and Small Sample Sizes',
        'nrows': nrows,
        'intrinsic_rank': rank,
        'sample_sizes': sample_sizes,
        'fixed_sigma': fixed_sigma,
        'noise_sigmas': noise_sigmas,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'total_time_seconds': total_time,
        'device': str(device)
    }
    with open(f'{out_dir}/configuration.json', 'w') as f:
        json.dump(config, f, indent=4)
        
    # Save metrics.txt
    with open(f'{out_dir}/metrics.txt', 'w') as f:
        f.write('============================================================\n')
        f.write('DEEPTENSOR BASELINE: FIGURE 7 METRICS SUMMARY\n')
        f.write('============================================================\n')
        f.write(f'Device: {device}\n')
        f.write(f'Feature Dimension: {nrows} | Intrinsic Rank: {rank}\n')
        f.write(f'Execution Time: {total_time:.2f} seconds\n\n')
        
        f.write('PART 1: COVARIANCE RECONSTRUCTION PSNR (dB) VS SAMPLE SIZE (at sigma=0.5)\n')
        f.write('SAMPLES     NOISY_SAMPLE_COV     SVD_PSNR     DEEPTENSOR_PSNR     GAIN (DT - SVD)\n')
        f.write('--------------------------------------------------------------------------------\n')
        for i, ndata in enumerate(sample_sizes):
            nc = sample_size_results['noisy_cov_psnr'][i]
            s = sample_size_results['svd_psnr'][i]
            d = sample_size_results['deeptensor_psnr'][i]
            g = d - s
            f.write(f'{ndata:<11d} {nc:<20.2f} {s:<12.2f} {d:<19.2f} {g:+.2f} dB\n')
        f.write('\n')
        
        f.write('PART 2: COVARIANCE RECONSTRUCTION PSNR (dB) VS NOISE LEVEL (at M=32 samples)\n')
        f.write('SIGMA       SVD_PSNR     DEEPTENSOR_PSNR     GAIN (DT - SVD)\n')
        f.write('------------------------------------------------------------\n')
        for i, sig in enumerate(noise_sigmas):
            s = noise_results_32['svd_psnr'][i]
            d = noise_results_32['deeptensor_psnr'][i]
            g = d - s
            f.write(f'{sig:<11.2f} {s:<12.2f} {d:<19.2f} {g:+.2f} dB\n')
            
    # Save Figure 7 Plots (Matching Paper Figure 7 Panels)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: PSNR vs Sample Size
    axes[0].plot(sample_sizes, sample_size_results['svd_psnr'], 'o--', label='SVD', color='crimson', linewidth=2)
    axes[0].plot(sample_sizes, sample_size_results['deeptensor_psnr'], 's-', label='DeepTensor', color='royalblue', linewidth=2)
    axes[0].plot(sample_sizes, sample_size_results['noisy_cov_psnr'], 'k:', label='Sample Covariance', alpha=0.5)
    axes[0].set_xscale('log', base=2)
    axes[0].set_title('PSNR vs. Sample Size (Noise $\\sigma = 0.5$)', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Number of Data Samples ($M$)', fontsize=11)
    axes[0].set_ylabel('Covariance PSNR (dB)', fontsize=11)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(fontsize=10)
    
    # Right: PSNR vs Noise Level at M=32
    axes[1].plot(noise_sigmas, noise_results_32['svd_psnr'], 'o--', label='SVD', color='crimson', linewidth=2)
    axes[1].plot(noise_sigmas, noise_results_32['deeptensor_psnr'], 's-', label='DeepTensor', color='royalblue', linewidth=2)
    axes[1].set_title('PSNR vs. Noise Level (Small Sample Size $M = 32$)', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Noise Standard Deviation ($\\sigma$)', fontsize=11)
    axes[1].set_ylabel('Covariance PSNR (dB)', fontsize=11)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend(fontsize=10)
    
    plt.tight_layout()
    fig.savefig(f'{out_dir}/plots/figure7_pca_sample_size_and_noise.png', dpi=300)
    plt.close(fig)
    
    # Save Covariance Matrix Visual Reconstructions (M=32 vs M=1024)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for r_idx, ndata in enumerate([32, 1024]):
        data = rep_sample_images[ndata]
        gt = data['cov_gt']
        noisy = data['cov_noisy']
        svd = data['cov_svd']
        dt = data['cov_dt']
        
        axes[r_idx, 0].imshow(gt, cmap='viridis')
        axes[r_idx, 0].set_title(f'Ground Truth Covariance\n(M = {ndata})', fontsize=10)
        axes[r_idx, 0].axis('off')
        
        axes[r_idx, 1].imshow(noisy, cmap='viridis')
        axes[r_idx, 1].set_title(f'Sample Covariance ({utils.psnr(gt, noisy):.1f} dB)', fontsize=10)
        axes[r_idx, 1].axis('off')
        
        axes[r_idx, 2].imshow(svd, cmap='viridis')
        axes[r_idx, 2].set_title(f'SVD Covariance ({utils.psnr(gt, svd):.1f} dB)', fontsize=10)
        axes[r_idx, 2].axis('off')
        
        axes[r_idx, 3].imshow(dt, cmap='viridis')
        axes[r_idx, 3].set_title(f'DeepTensor Covariance ({utils.psnr(gt, dt):.1f} dB)', fontsize=10)
        axes[r_idx, 3].axis('off')
        
    plt.tight_layout()
    fig.savefig(f'{out_dir}/images/figure7_covariance_visual_comparison.png', dpi=300)
    plt.close(fig)
    
    # Save checkpoint array
    np.savez_compressed(
        f'{out_dir}/figure7_results.npz',
        sample_size_results=sample_size_results,
        noise_results_32=noise_results_32,
        config=config
    )
    print(f'Figure 7 execution complete. All outputs saved in: {out_dir}')