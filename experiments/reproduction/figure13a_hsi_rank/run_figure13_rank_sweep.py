#!/usr/bin/env python
'''
Figure 13(a): Effect of Rank Choice on Hyperspectral Image Decomposition
Paper: "DeepTensor: Low-Rank Tensor Decomposition with Deep Network Priors" (Section 4.4, Page 9-10)

Experiment:
- ICVL Hyperspectral Image truncated to Rank-20 ground truth (174, 164, 260)
- Measurement Noise: Poisson (tau=100) + Gaussian (snr=2) -> Input PSNR ~ 25 dB
- Decomposition Rank swept from 1 to 30: [1, 5, 10, 15, 20, 25, 30]
- Evaluates DeepTensor vs Truncated SVD to demonstrate that DeepTensor is insensitive
  to rank over-estimation due to its deep network prior.
'''

import os
import sys
import time
import json
import argparse
import numpy as np
import scipy.io as io
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure root directory and modules are on sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from deeptensor import models, utils, losses, deep_prior, deep_decoder

def run_deeptensor_rank(cube_gt, cube_noisy, rank, epochs, lr, device):
    H, W, nwvl = cube_gt.shape
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    noisy_ten = torch.tensor(cube_noisy, device=device)[None, ...].permute(0, 3, 1, 2)
    gt_ten = torch.tensor(cube_gt, device=device)[None, ...]

    im_net = deep_prior.get_net(rank, 'skip', 'reflection', upsample_mode='bilinear',
                                skip_n33d=64, skip_n33u=64, num_scales=5, n_channels=rank).to(device)
    spec_net = deep_prior.get_net(rank, 'skip1d', 'reflection', upsample_mode='linear',
                                  skip_n33d=64, skip_n33u=64, num_scales=5, n_channels=rank).to(device)

    im_inp = utils.get_inp([1, rank, H, W], device=device)
    spec_inp = utils.get_inp([1, rank, nwvl], device=device)

    im_net.train()
    spec_net.train()

    optimizer = torch.optim.Adam(list(im_net.parameters()) + list(spec_net.parameters()) + [im_inp, spec_inp], lr=lr)
    criterion = losses.L1Norm()

    best_loss = float('inf')
    best_recon = None

    for epoch in range(1, epochs + 1):
        U_img = im_net(im_inp)
        V = spec_net(spec_inp)

        U = U_img.reshape(-1, rank, H * W).permute(0, 2, 1)
        mat_estim = torch.bmm(U, V)
        cube_estim = mat_estim.reshape(1, H, W, nwvl).permute(0, 3, 1, 2)

        loss = criterion(cube_estim - noisy_ten)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        if loss_val < best_loss:
            best_loss = loss_val
            best_recon = mat_estim.detach().clone()

        if epoch % 100 == 0 or epoch == epochs:
            print(f"  [Epoch {epoch:4d}/{epochs:4d}] Loss: {loss_val:.4e}", flush=True)

    recon_final = best_recon.cpu().squeeze().numpy().reshape(H, W, nwvl)
    psnr_val = float(utils.psnr(cube_gt, recon_final))
    return psnr_val, recon_final

def main():
    parser = argparse.ArgumentParser(description="Figure 13(a): Effect of Rank Choice on HSI")
    parser.add_argument('--dataset', type=str, default=os.path.join(project_root, 'data', 'icvl.mat'))
    parser.add_argument('--gt_rank', type=int, default=20, help="Ground-truth rank truncation")
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default=os.path.join(project_root, 'results', 'baseline', 'figure13a_hsi_rank'))
    args = parser.parse_args()

    plots_dir = os.path.join(args.output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    print("======================================================================")
    print("FIGURE 13(a): EFFECT OF RANK CHOICE (DEEPTENSOR VS SVD)")
    print("======================================================================")

    data = io.loadmat(args.dataset)
    cube_raw = data['hypercube'].astype(np.float32)
    cube_raw = cube_raw / cube_raw.max()
    H, W, nwvl = cube_raw.shape

    # Truncate to Rank-20 ground truth as specified in paper
    mat_raw = cube_raw.reshape(H * W, nwvl)
    u, s, vt = np.linalg.svd(mat_raw, full_matrices=False)
    mat_gt = np.dot(u[:, :args.gt_rank] * s[:args.gt_rank], vt[:args.gt_rank, :])
    cube_gt = mat_gt.reshape(H, W, nwvl)
    cube_gt = np.clip(cube_gt, 0.0, 1.0)
    print(f"Generated Ground-Truth Rank-{args.gt_rank} Truncated HSI: {cube_gt.shape}")

    # Add Poisson + Gaussian noise -> 25 dB input PSNR
    np.random.seed(42)
    torch.manual_seed(42)
    cube_noisy = utils.measure(cube_gt, noise_snr=2.0, tau=100)
    noisy_psnr = float(utils.psnr(cube_gt, cube_noisy))
    print(f"Added Poisson + Gaussian Readout Noise | Input PSNR: {noisy_psnr:.2f} dB")

    ranks = [1, 5, 10, 15, 20, 25, 30]
    svd_psnrs = []
    dt_psnrs = []

    mat_noisy = cube_noisy.reshape(H * W, nwvl)

    for r in ranks:
        print(f"\nEvaluating Rank R = {r}...")
        # SVD baseline
        u_n, s_n, vt_n = np.linalg.svd(mat_noisy, full_matrices=False)
        rec_svd = np.dot(u_n[:, :r] * s_n[:r], vt_n[:r, :]).reshape(H, W, nwvl)
        p_svd = float(utils.psnr(cube_gt, rec_svd))
        svd_psnrs.append(p_svd)

        # DeepTensor
        p_dt, _ = run_deeptensor_rank(cube_gt, cube_noisy, r, args.epochs, args.learning_rate, args.device)
        dt_psnrs.append(p_dt)
        print(f"Rank {r:2d} | SVD PSNR: {p_svd:.2f} dB | DeepTensor PSNR: {p_dt:.2f} dB")

    # Plot Figure 13(a)
    plt.figure(figsize=(7, 5))
    plt.plot(ranks, dt_psnrs, marker='o', linewidth=2.5, color='blue', label='DeepTensor (Ours)')
    plt.plot(ranks, svd_psnrs, marker='s', linewidth=2.0, linestyle='--', color='red', label='Truncated SVD')
    plt.axvline(x=args.gt_rank, color='gray', linestyle=':', label=f'True Rank ({args.gt_rank})')
    plt.xlabel('Decomposition Rank', fontsize=11)
    plt.ylabel('Reconstruction PSNR (dB)', fontsize=11)
    plt.title('Figure 13(a): Effect of Rank Choice on Matrix Decomposition', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()

    fig13a_path = os.path.join(plots_dir, 'figure13a_rank_sweep.png')
    plt.savefig(fig13a_path, dpi=200)
    plt.close()
    print(f"\nSaved Figure 13(a) plot to {fig13a_path}")

    results_data = {
        'ranks': ranks,
        'true_rank': args.gt_rank,
        'input_noisy_psnr': noisy_psnr,
        'svd_psnrs': svd_psnrs,
        'deeptensor_psnrs': dt_psnrs
    }

    with open(os.path.join(args.output_dir, 'figure13a_metrics.json'), 'w') as fp:
        json.dump(results_data, fp, indent=2)

    print("Successfully completed Figure 13(a) rank sweep experiment!")

if __name__ == '__main__':
    main()

