#!/usr/bin/env python

'''
This script replicates results in Figure 6 of the DeepTensor research paper:
"DeepTensor: Low-Rank Tensor Decomposition With Deep Network Priors" (Section 3.2, Page 5)

Experiment:
Low-rank matrix decomposition robustness across signal and noise distributions:
1. Gaussian signal + Gaussian noise (std = 0.1)
2. Piecewise constant signal + Poisson noise (mean in [0, 1000], tau=1000)
3. Piecewise constant signal + Rician noise (std = 0.02)

Across ranks: [10, 20, 30, 40, 50, 60]
Models: DeepTensor (1D DIP network for U and V) vs Truncated SVD.
Outputs saved to: RESULTS-BASELINE/figure6/
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

sys.path.append('modules')
import models
import utils
import losses
import deep_prior

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_deeptensor_matrix(mat, mat_gt, nrows, ncols, rank, epochs=600, lr=1e-4):
    '''
    Train DeepTensor 1D DIP networks on given matrix
    '''
    mat_ten = torch.tensor(mat, dtype=torch.float32, device=device)[None, ...]
    mat_gt_ten = torch.tensor(mat_gt, dtype=torch.float32, device=device)[None, ...]
    
    u_inp = utils.get_inp([1, rank, nrows], device=device)
    v_inp = utils.get_inp([1, rank, ncols], device=device)
    
    u_net = deep_prior.get_net(rank, 'skip1d', 'reflection', 'linear',
                               n_channels=rank, skip_n33d=128, skip_n33u=128,
                               num_scales=5).to(device)
    v_net = deep_prior.get_net(rank, 'skip1d', 'reflection', 'linear',
                               n_channels=rank, skip_n33d=128, skip_n33u=128,
                               num_scales=5).to(device)
                               
    optimizer = torch.optim.Adam(
        list(u_net.parameters()) + list(v_net.parameters()) + [u_inp, v_inp],
        lr=lr
    )
    criterion = losses.L2Norm()
    
    best_loss = float('inf')
    best_mat = None
    losses_history = []
    
    for epoch in range(epochs):
        u_out = u_net(u_inp).permute(0, 2, 1)
        v_out = v_net(v_inp)
        
        mat_estim = torch.bmm(u_out, v_out)
        loss = criterion(mat_estim - mat_ten)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_val = loss.item()
        losses_history.append(loss_val)
        if loss_val < best_loss:
            best_loss = loss_val
            best_mat = mat_estim.detach().squeeze().cpu().numpy()
            
    dt_psnr = utils.psnr(mat_gt, best_mat)
    return best_mat, dt_psnr, best_loss, losses_history

if __name__ == '__main__':
    print('============================================================')
    print('DEEPTENSOR BASELINE: FIGURE 6 EXPERIMENT')
    print('============================================================')
    print(f'Device: {device}')
    
    out_dir = 'RESULTS-BASELINE/figure6'
    os.makedirs(f'{out_dir}/images', exist_ok=True)
    os.makedirs(f'{out_dir}/plots', exist_ok=True)
    
    nrows = 64
    ncols = 64
    ranks = [10, 20, 30, 40, 50, 60]
    epochs_per_run = 600
    learning_rate = 1e-4
    
    regimes = [
        {
            'name': 'Gaussian Signal + Gaussian Noise',
            'id': 'gaussian_gaussian',
            'signal_type': 'gaussian',
            'noise_type': 'gaussian',
            'noise_snr': 0.1,
            'tau': 1000
        },
        {
            'name': 'Piecewise Constant Signal + Poisson Noise',
            'id': 'piecewise_poisson',
            'signal_type': 'piecewise',
            'noise_type': 'poisson',
            'noise_snr': 2.0,
            'tau': 1000
        },
        {
            'name': 'Piecewise Constant Signal + Rician Noise',
            'id': 'piecewise_rician',
            'signal_type': 'piecewise',
            'noise_type': 'rician',
            'noise_snr': 0.02,
            'tau': 1000
        }
    ]
    
    results = {}
    representative_images = {}
    
    total_tic = time.time()
    
    for regime in regimes:
        reg_id = regime['id']
        reg_name = regime['name']
        print(f'\n--- Evaluating Regime: {reg_name} ---')
        
        dt_psnrs = []
        svd_psnrs = []
        noisy_psnrs = []
        
        # We will keep the rank=30 matrices for visual comparison
        rep_data = None
        
        for rank in ranks:
            # Set seed for reproducible comparison across ranks
            np.random.seed(42 + rank)
            torch.manual_seed(42 + rank)
            
            mat, mat_gt = utils.get_matrix(
                nrows, ncols, rank,
                noise_type=regime['noise_type'],
                signal_type=regime['signal_type'],
                noise_snr=regime['noise_snr'],
                tau=regime['tau']
            )
            
            noisy_p = utils.psnr(mat_gt, mat)
            
            # 1. Truncated SVD baseline
            mat_svd = utils.lr_decompose(mat, rank)
            svd_p = utils.psnr(mat_gt, mat_svd)
            
            # 2. DeepTensor
            t0 = time.time()
            mat_dt, dt_p, best_loss, loss_curve = train_deeptensor_matrix(
                mat, mat_gt, nrows, ncols, rank,
                epochs=epochs_per_run, lr=learning_rate
            )
            elapsed = time.time() - t0
            
            dt_psnrs.append(float(dt_p))
            svd_psnrs.append(float(svd_p))
            noisy_psnrs.append(float(noisy_p))
            
            print(f'  Rank {rank:2d} | Noisy: {noisy_p:.2f} dB | SVD: {svd_p:.2f} dB | DeepTensor: {dt_p:.2f} dB | Time: {elapsed:.1f}s')
            
            if rank == 30:
                rep_data = {
                    'gt': mat_gt,
                    'noisy': mat,
                    'svd': mat_svd,
                    'dt': mat_dt,
                    'loss_curve': loss_curve
                }
                
        results[reg_id] = {
            'regime_name': reg_name,
            'ranks': ranks,
            'noisy_psnr': noisy_psnrs,
            'svd_psnr': svd_psnrs,
            'deeptensor_psnr': dt_psnrs
        }
        representative_images[reg_id] = rep_data
        
    total_time = time.time() - total_tic
    print(f'\nAll Figure 6 regimes completed in {total_time:.1f} seconds.')
    
    # Save configuration
    config = {
        'experiment': 'Figure 6: Robustness to Signal & Noise Distributions',
        'nrows': nrows,
        'ncols': ncols,
        'ranks': ranks,
        'epochs_per_run': epochs_per_run,
        'learning_rate': learning_rate,
        'regimes': regimes,
        'total_time_seconds': total_time,
        'device': str(device)
    }
    with open(f'{out_dir}/configuration.json', 'w') as f:
        json.dump(config, f, indent=4)
        
    # Save metrics.txt
    with open(f'{out_dir}/metrics.txt', 'w') as f:
        f.write('============================================================\n')
        f.write('DEEPTENSOR BASELINE: FIGURE 6 METRICS SUMMARY\n')
        f.write('============================================================\n')
        f.write(f'Device: {device}\n')
        f.write(f'Matrix Dimensions: {nrows} x {ncols}\n')
        f.write(f'Evaluated Ranks: {ranks}\n')
        f.write(f'Total Execution Time: {total_time:.2f} seconds\n\n')
        
        for reg in regimes:
            reg_id = reg['id']
            res = results[reg_id]
            f.write(f'REGIME: {reg["name"]}\n')
            f.write(f'Noise: {reg["noise_type"]} (snr={reg["noise_snr"]}, tau={reg["tau"]}) | Signal: {reg["signal_type"]}\n')
            f.write('RANK     INPUT_NOISY_PSNR     SVD_PSNR     DEEPTENSOR_PSNR     GAIN (DT - SVD)\n')
            f.write('-----------------------------------------------------------------------------\n')
            for i, r in enumerate(ranks):
                n_p = res['noisy_psnr'][i]
                s_p = res['svd_psnr'][i]
                d_p = res['deeptensor_psnr'][i]
                gain = d_p - s_p
                f.write(f'{r:<8d} {n_p:<20.2f} {s_p:<12.2f} {d_p:<19.2f} {gain:+.2f} dB\n')
            f.write('\n')
            
    # Save Figure 6 Paper-matched Curves (3 panels matching Fig. 6)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, reg in enumerate(regimes):
        reg_id = reg['id']
        res = results[reg_id]
        ax = axes[i]
        ax.plot(ranks, res['svd_psnr'], 'o--', label='SVD', color='crimson', linewidth=2)
        ax.plot(ranks, res['deeptensor_psnr'], 's-', label='DeepTensor', color='royalblue', linewidth=2)
        ax.plot(ranks, res['noisy_psnr'], 'k:', label='Noisy Input', alpha=0.5)
        ax.set_title(reg['name'], fontsize=11, fontweight='bold')
        ax.set_xlabel('Rank', fontsize=11)
        ax.set_ylabel('PSNR (dB)', fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=10)
        
    plt.tight_layout()
    fig.savefig(f'{out_dir}/plots/figure6_psnr_vs_rank.png', dpi=300)
    plt.close(fig)
    
    # Save Visual Reconstructions & Error Maps (Rank 30)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    row_titles = ['Gaussian + Gaussian', 'Piecewise + Poisson', 'Piecewise + Rician']
    
    for r_idx, reg in enumerate(regimes):
        rep = representative_images[reg['id']]
        gt = rep['gt']
        noisy = rep['noisy']
        svd = rep['svd']
        dt = rep['dt']
        
        # Col 0: Ground Truth
        axes[r_idx, 0].imshow(gt, cmap='viridis')
        axes[r_idx, 0].set_title(f'{row_titles[r_idx]}\nGround Truth', fontsize=10)
        axes[r_idx, 0].axis('off')
        
        # Col 1: Noisy Input
        axes[r_idx, 1].imshow(noisy, cmap='viridis')
        axes[r_idx, 1].set_title(f'Noisy ({utils.psnr(gt, noisy):.1f} dB)', fontsize=10)
        axes[r_idx, 1].axis('off')
        
        # Col 2: SVD
        axes[r_idx, 2].imshow(svd, cmap='viridis')
        axes[r_idx, 2].set_title(f'SVD ({utils.psnr(gt, svd):.1f} dB)', fontsize=10)
        axes[r_idx, 2].axis('off')
        
        # Col 3: DeepTensor
        axes[r_idx, 3].imshow(dt, cmap='viridis')
        axes[r_idx, 3].set_title(f'DeepTensor ({utils.psnr(gt, dt):.1f} dB)', fontsize=10)
        axes[r_idx, 3].axis('off')
        
    plt.tight_layout()
    fig.savefig(f'{out_dir}/images/figure6_visual_comparisons_rank30.png', dpi=300)
    plt.close(fig)
    
    # Save checkpoint arrays
    np.savez_compressed(
        f'{out_dir}/figure6_results.npz',
        results=results,
        config=config
    )
    print(f'Figure 6 execution complete. All outputs saved in: {out_dir}')
