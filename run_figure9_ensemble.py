#!/usr/bin/env python
"""
Extension 5: Ensemble Disagreement as a Proxy for Reconstruction Uncertainty.

Performs N=5 ensemble training of DeepTensor under fixed noisy observation (seed 42).
Evaluates empirical ensemble disagreement (std dev across members) vs true reconstruction error,
controlling for ground-truth edge magnitude.
"""

import argparse
import copy
import os
import random
import sys
import time

import numpy as np
from scipy import io
from scipy.stats import pearsonr, spearmanr
from skimage.metrics import structural_similarity as ssim_func
import torch

sys.path.append('modules')

import deep_decoder
import deep_prior
import losses
import models
import spectral
import utils


def set_seed(seed):
    """Explicitly seed Python random, NumPy, and PyTorch CPU/CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def compute_edge_magnitude(image_3d):
    """
    Compute local spatial gradient magnitude from clean ground truth.
    image_3d: np.ndarray of shape (H, W, B)
    Returns edge_mag of shape (H, W, B)
    """
    gx, gy = np.gradient(image_3d, axis=(0, 1))
    edge_mag = np.sqrt(gx**2 + gy**2)
    return edge_mag


def main():
    parser = argparse.ArgumentParser(
        description='Extension 5: Ensemble Uncertainty Estimation'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=5000,
        help='Epochs per ensemble member (default: 5000)',
    )
    parser.add_argument(
        '--smoke_test',
        action='store_true',
        help='Run small smoke test (5 epochs)',
    )
    args = parser.parse_args()

    epochs = 5 if args.smoke_test else args.epochs
    out_dir = 'results_ensemble_smoke' if args.smoke_test else 'results_ensemble'
    os.makedirs(out_dir, exist_ok=True)

    expname = 'icvl'
    rank = 20
    n_inputs = rank
    init_nconv = 128
    nettype = 'dip'
    scaling = 1

    # Baseline noise parameters
    obs_seed = 42
    tau = 100
    noise_snr = 2

    # Learning constants
    learning_rate = 1e-3
    reg_noise_std = 1.0 / 30.0
    exp_weight = 0.99

    member_seeds = [43, 44, 45, 46, 47]

    print(f"=== Starting Extension 5 Ensemble Experiment ===")
    print(f"Mode: {'SMOKE TEST' if args.smoke_test else 'FULL EXPERIMENT'}")
    print(f"Epochs per member: {epochs}")
    print(f"Output directory: {out_dir}")
    print(f"Observation seed: {obs_seed}")
    print(f"Member seeds: {member_seeds}")

    # Load data
    data = io.loadmat(f'data/{expname}.mat')
    cube = data['hypercube'].astype(np.float32)
    cube = utils.resize(cube / cube.max(), scaling)
    H, W, nwvl = cube.shape
    hsmat = cube.reshape(H * W, nwvl)
    data_range = cube.max() - cube.min()

    # Generate fixed observation deterministically with master seed 42
    set_seed(obs_seed)
    cube_noisy = utils.measure(cube, noise_snr=noise_snr, tau=tau)
    obs_psnr = utils.psnr(cube, cube_noisy)
    print(f"Fixed noisy observation generated (PSNR: {obs_psnr:.4f} dB)")

    hsten = torch.tensor(hsmat)[None, ...].cuda()
    hsten_noisy = torch.tensor(cube_noisy.reshape(H * W, nwvl))[None, ...].cuda()
    criterion_l1 = losses.L1Norm()

    reconstructions = []
    member_metrics = []
    member_runtimes = []

    start_total_time = time.time()

    for m_idx, seed in enumerate(member_seeds, 1):
        print(f"\n--- Training Ensemble Member {m_idx}/5 (Seed: {seed}) ---")

        # Explicitly seed all RNGs immediately before model construction
        set_seed(seed)

        # Generate networks
        if nettype == 'dip':
            im_net = deep_prior.get_net(
                n_inputs,
                'skip',
                'reflection',
                upsample_mode='bilinear',
                skip_n33d=128,
                skip_n33u=128,
                num_scales=5,
                n_channels=rank,
            ).cuda()
            spec_net = deep_prior.get_net(
                n_inputs,
                'skip1d',
                'reflection',
                upsample_mode='linear',
                skip_n33d=128,
                skip_n33u=128,
                num_scales=5,
                n_channels=rank,
            ).cuda()
            im_inp = utils.get_inp([1, n_inputs, H, W])
            spec_inp = utils.get_inp([1, n_inputs, nwvl])
        else:
            raise ValueError(f"Unsupported nettype: {nettype}")

        # Ensure train() mode for network
        im_net.train()
        spec_net.train()

        net_params = list(im_net.parameters()) + list(spec_net.parameters())
        inp_params = [im_inp] + [spec_inp]

        im_inp_per = im_inp.detach().clone()
        spec_inp_per = spec_inp.detach().clone()

        best_loss = float('inf')
        best_hs_estim = None

        params = net_params + inp_params
        optimizer = torch.optim.Adam(lr=learning_rate, params=params)

        m_tic = time.time()
        for idx in range(epochs):
            im_inp_perturbed = im_inp + im_inp_per.normal_() * reg_noise_std
            spec_inp_perturbed = (
                spec_inp + spec_inp_per.normal_() * reg_noise_std
            )

            U_img = im_net(im_inp_perturbed)
            V = spec_net(spec_inp_perturbed)

            U = U_img.reshape(-1, rank, H * W).permute(0, 2, 1)
            hs_estim = torch.bmm(U, V)

            loss = criterion_l1(hsten_noisy - hs_estim)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_hs_estim = copy.deepcopy(hs_estim.detach().squeeze())

        m_toc = time.time()
        m_runtime = m_toc - m_tic
        member_runtimes.append(m_runtime)

        # Reshape reconstructed output
        cube_estim = best_hs_estim.cpu().numpy().reshape(H, W, nwvl)
        reconstructions.append(cube_estim)

        # Compute individual metrics
        m_psnr = utils.psnr(cube, cube_estim)
        m_ssim = ssim_func(
            cube, cube_estim, channel_axis=-1, data_range=data_range
        )
        m_rmse = np.sqrt(((cube - cube_estim) ** 2).mean())

        member_metrics.append({
            'member': m_idx,
            'seed': seed,
            'psnr': m_psnr,
            'ssim': m_ssim,
            'rmse': m_rmse,
            'runtime': m_runtime,
        })
        print(
            f"Member {m_idx} (Seed {seed}) Complete - PSNR: {m_psnr:.4f} dB, SSIM: {m_ssim:.6f}, RMSE: {m_rmse:.6f}, Time: {m_runtime:.2f}s"
        )

    total_runtime = time.time() - start_total_time
    vram_peak = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if torch.cuda.is_available()
        else 0.0
    )

    # Convert reconstructions list to numpy array of shape (5, H, W, B)
    y_members = np.stack(reconstructions, axis=0)  # (5, H, W, nwvl)

    # 1. Ensemble Mean Reconstruction
    y_mean = np.mean(y_members, axis=0)  # (H, W, nwvl)

    # 2. Per-voxel empirical standard deviation (Uncertainty Tensor)
    uncertainty_tensor = np.std(
        y_members, axis=0, ddof=1
    )  # (H, W, nwvl) sample std dev

    # 3. Absolute Reconstruction Error
    abs_error_tensor = np.abs(y_mean - cube)  # (H, W, nwvl)

    # 4. Clean GT Edge Magnitude
    edge_mag_tensor = compute_edge_magnitude(cube)  # (H, W, nwvl)

    # Flatten for global correlation analysis across full tensor H x W x B
    u_flat = uncertainty_tensor.flatten()
    e_flat = abs_error_tensor.flatten()
    g_flat = edge_mag_tensor.flatten()

    # Primary correlations (without p-values)
    pearson_u_e = pearsonr(u_flat, e_flat)[0]
    spearman_u_e = spearmanr(u_flat, e_flat)[0]

    # Edge control correlations
    pearson_g_e = pearsonr(g_flat, e_flat)[0]
    spearman_g_e = spearmanr(g_flat, e_flat)[0]

    pearson_g_u = pearsonr(g_flat, u_flat)[0]
    spearman_g_u = spearmanr(g_flat, u_flat)[0]

    # Quantile Analysis: top 10% most uncertain vs bottom 10% most certain
    k_10 = int(0.10 * len(u_flat))
    sorted_idx = np.argsort(u_flat)
    bottom_10_idx = sorted_idx[:k_10]
    top_10_idx = sorted_idx[-k_10:]

    bottom_10_mae = e_flat[bottom_10_idx].mean()
    top_10_mae = e_flat[top_10_idx].mean()
    ratio_mae = (
        top_10_mae / bottom_10_mae if bottom_10_mae > 0 else float('nan')
    )

    global_mean_uncertainty = u_flat.mean()

    # Individual Member Aggregates (Mean +/- Std)
    psnrs = [m['psnr'] for m in member_metrics]
    ssims = [m['ssim'] for m in member_metrics]
    rmses = [m['rmse'] for m in member_metrics]

    mean_indiv_psnr, std_indiv_psnr = np.mean(psnrs), np.std(psnrs, ddof=1)
    mean_indiv_ssim, std_indiv_ssim = np.mean(ssims), np.std(ssims, ddof=1)
    mean_indiv_rmse, std_indiv_rmse = np.mean(rmses), np.std(rmses, ddof=1)

    # Ensemble Mean Metrics (Secondary finding)
    ens_psnr = utils.psnr(cube, y_mean)
    ens_ssim = ssim_func(
        cube, y_mean, channel_axis=-1, data_range=data_range
    )
    ens_rmse = np.sqrt(((cube - y_mean) ** 2).mean())

    # Save arrays (convert reconstructions to float16 to keep npz under GitHub 100MB limit)
    arrays_path = os.path.join(out_dir, 'ensemble_results.npz')
    np.savez_compressed(
        arrays_path,
        reconstructions=y_members.astype(np.float16),
        ensemble_mean=y_mean,
        uncertainty=uncertainty_tensor,
        absolute_error=abs_error_tensor,
        edge_magnitude=edge_mag_tensor,
    )

    # Save Summary Text
    summary_path = os.path.join(out_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("==================================================\n")
        f.write(
            "EXTENSION 5: ENSEMBLE UNCERTAINTY ESTIMATION SUMMARY\n"
        )
        f.write("==================================================\n\n")
        f.write(f"Mode: {'SMOKE TEST' if args.smoke_test else 'FULL EXPERIMENT'}\n")
        f.write(f"Observation Master Seed: {obs_seed}\n")
        f.write(f"Member Seeds: {member_seeds}\n")
        f.write(f"Epochs per Member: {epochs}\n")
        f.write(f"Noise Config: tau={tau}, noise_snr={noise_snr} (Poisson + Gaussian mixed)\n")
        f.write(f"Observation PSNR: {obs_psnr:.4f} dB\n")
        f.write(f"Total Runtime: {total_runtime:.2f} s\n")
        f.write(f"Peak VRAM: {vram_peak:.2f} MB\n\n")

        f.write("--- INDIVIDUAL MEMBER METRICS ---\n")
        for m in member_metrics:
            f.write(
                f"Member {m['member']} (Seed {m['seed']}): PSNR={m['psnr']:.4f} dB, SSIM={m['ssim']:.6f}, RMSE={m['rmse']:.6f}, Time={m['runtime']:.2f}s\n"
            )
        f.write(
            f"Mean Individual PSNR: {mean_indiv_psnr:.4f} +/- {std_indiv_psnr:.4f} dB\n"
        )
        f.write(
            f"Mean Individual SSIM: {mean_indiv_ssim:.6f} +/- {std_indiv_ssim:.6f}\n"
        )
        f.write(
            f"Mean Individual RMSE: {mean_indiv_rmse:.6f} +/- {std_indiv_rmse:.6f}\n\n"
        )

        f.write("--- ENSEMBLE MEAN RECONSTRUCTION METRICS (SECONDARY) ---\n")
        f.write(f"Ensemble Mean PSNR: {ens_psnr:.4f} dB\n")
        f.write(f"Ensemble Mean SSIM: {ens_ssim:.6f}\n")
        f.write(f"Ensemble Mean RMSE: {ens_rmse:.6f}\n")
        f.write(f"PSNR Improvement vs Mean Individual: {ens_psnr - mean_indiv_psnr:+.4f} dB (Cost: ~5x compute)\n\n")

        f.write("--- PRIMARY UNCERTAINTY ANALYSIS (FULL HxWxB TENSOR) ---\n")
        f.write(f"Global Mean Uncertainty (Std Dev): {global_mean_uncertainty:.6e}\n")
        f.write(f"Pearson Correlation (Uncertainty vs Abs Error): {pearson_u_e:.6f}\n")
        f.write(f"Spearman Correlation (Uncertainty vs Abs Error): {spearman_u_e:.6f}\n\n")

        f.write("--- EDGE-MAGNITUDE CONTROL ANALYSIS ---\n")
        f.write(f"Pearson Correlation (GT Edge Mag vs Abs Error): {pearson_g_e:.6f}\n")
        f.write(f"Spearman Correlation (GT Edge Mag vs Abs Error): {spearman_g_e:.6f}\n")
        f.write(f"Pearson Correlation (GT Edge Mag vs Uncertainty): {pearson_g_u:.6f}\n")
        f.write(f"Spearman Correlation (GT Edge Mag vs Uncertainty): {spearman_g_u:.6f}\n\n")

        f.write("--- QUANTILE ANALYSIS ---\n")
        f.write(f"Top 10% Most Uncertain MAE: {top_10_mae:.6f}\n")
        f.write(f"Bottom 10% Most Certain MAE: {bottom_10_mae:.6f}\n")
        f.write(f"Ratio (Top 10% / Bottom 10% MAE): {ratio_mae:.4f}\n")

    print(f"\nAll results saved successfully to {out_dir}/")
    print(f"Summary: {summary_path}")
    print(f"Arrays: {arrays_path}")


if __name__ == '__main__':
    main()
