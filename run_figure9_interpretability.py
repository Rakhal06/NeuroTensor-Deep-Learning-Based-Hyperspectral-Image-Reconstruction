#!/usr/bin/env python
"""
Extension 7: Post-hoc Factor Analysis and Descriptive Visualization of DeepTensor Components.

Extracts spatial factors U and spectral factors V from frozen baseline R=20 DeepTensor on ICVL (seed 42, mixed noise).
Performs post-hoc energy-norm component ordering, progressive top-k ablation, spatial/spectral cosine similarity,
and generates four required descriptive visualizations.
"""

import argparse
import os
import sys
import copy
import time

import numpy as np
from scipy import io
from skimage.metrics import structural_similarity as ssim_func

import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append('modules')

import models
import utils
import losses
import spectral
import deep_prior
import deep_decoder


def run_experiment(smoke_test=False, epochs=5000, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    expname = 'icvl'
    rank = 20
    n_inputs = rank
    init_nconv = 128
    nettype = 'dip'
    
    scaling = 1
    tau = 100
    noise_snr = 2
    
    learning_rate = 1e-3
    if smoke_test:
        epochs = 10
    reg_noise_std = 1.0 / 30.0
    exp_weight = 0.99
    
    # Load data
    data = io.loadmat('data/%s.mat' % expname)
    cube = data['hypercube'].astype(np.float32)
    wavelengths = data['wavelengths'].astype(np.float32).ravel()
    cube = utils.resize(cube / cube.max(), scaling)
    
    H, W, nwvl = cube.shape
    hsmat = cube.reshape(H * W, nwvl)
    
    # Measure noisy observation
    np.random.seed(seed)
    cube_noisy = utils.measure(cube, noise_snr, tau)
    actual_psnr = utils.psnr(cube, cube_noisy)
    
    hsten = torch.tensor(hsmat)[None, ...].cuda()
    hsten_noisy = torch.tensor(cube_noisy.reshape(H * W, nwvl))[None, ...].cuda()
    
    criterion_l1 = losses.L1Norm()
    
    # Generate DIP networks
    im_net = deep_prior.get_net(n_inputs, 'skip', 'reflection',
                                upsample_mode='bilinear',
                                skip_n33d=128, skip_n33u=128,
                                num_scales=5, n_channels=rank).cuda()
    spec_net = deep_prior.get_net(n_inputs, 'skip1d', 'reflection',
                                 upsample_mode='linear',
                                 skip_n33d=128, skip_n33u=128,
                                 num_scales=5, n_channels=rank).cuda()
    
    im_inp = utils.get_inp([1, n_inputs, H, W])
    spec_inp = utils.get_inp([1, n_inputs, nwvl])
    
    im_net.train()
    spec_net.train()
    
    net_params = list(im_net.parameters()) + list(spec_net.parameters())
    inp_params = [im_inp] + [spec_inp]
    
    im_inp_per = im_inp.detach().clone()
    spec_inp_per = spec_inp.detach().clone()
    
    best_loss = float('inf')
    best_epoch = 0
    best_hs_estim = None
    best_U_img = None
    best_V = None
    
    params = net_params + inp_params
    optimizer = torch.optim.Adam(lr=learning_rate, params=params)
    
    tic = time.time()
    for idx in range(epochs):
        im_inp_perturbed = im_inp + im_inp_per.normal_() * reg_noise_std
        spec_inp_perturbed = spec_inp + spec_inp_per.normal_() * reg_noise_std
        
        U_img = im_net(im_inp_perturbed) # [1, rank, H, W]
        V = spec_net(spec_inp_perturbed) # [1, rank, nwvl]
        
        U = U_img.reshape(-1, rank, H * W).permute(0, 2, 1) # [1, H*W, rank]
        
        hs_estim = torch.bmm(U, V) # [1, H*W, nwvl]
        
        loss = criterion_l1(hsten_noisy - hs_estim)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_epoch = idx
            best_hs_estim = copy.deepcopy(hs_estim.detach().squeeze())
            best_U_img = copy.deepcopy(U_img.detach())
            best_V = copy.deepcopy(V.detach())
            
    toc = time.time()
    runtime = toc - tic
    vram_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
    
    # ----------------------------------------------------
    # STEP 2: FACTOR EXTRACTION & SANITY CHECKS
    # ----------------------------------------------------
    # U_img: [1, R, H, W] -> reshape to U: [H*W, R] preserving row-major spatial flattening
    # In PyTorch: U_img is [1, R, H, W].
    # Baseline reshape: U_img.reshape(-1, rank, H*W).permute(0, 2, 1) gives shape [1, H*W, R].
    # Squeezing batch dimension gives U tensor of shape (H*W, R).
    U_tensor = best_U_img.reshape(rank, H * W).permute(1, 0).cpu().numpy() # [H*W, R]
    V_tensor = best_V.squeeze(0).cpu().numpy() # [R, N_lambda]
    
    # Verify tensor shapes
    assert U_tensor.shape == (H * W, rank), f"U shape mismatch: {U_tensor.shape}"
    assert V_tensor.shape == (rank, nwvl), f"V shape mismatch: {V_tensor.shape}"
    assert np.all(np.isfinite(U_tensor)), "U contains non-finite values"
    assert np.all(np.isfinite(V_tensor)), "V contains non-finite values"
    
    # Reconstruct Y_hat = sum_r U_r x V_r = U @ V
    Y_hat_flat = U_tensor @ V_tensor # [H*W, N_lambda]
    cube_estim = best_hs_estim.cpu().numpy().reshape(H, W, nwvl)
    Y_hat_reconstructed = Y_hat_flat.reshape(H, W, nwvl)
    
    # Check reconstruction match
    reconstruction_diff = np.max(np.abs(cube_estim - Y_hat_reconstructed))
    print(f"Sanity Check - Max diff between model output & U@V reconstruction: {reconstruction_diff:.8e}")
    assert reconstruction_diff < 1e-5, f"Reconstruction mismatch: {reconstruction_diff}"
    
    # Compute full reconstruction metrics against GT strictly post-hoc
    data_range = cube.max() - cube.min()
    full_psnr = utils.psnr(cube, cube_estim)
    full_ssim = ssim_func(cube, cube_estim, channel_axis=-1, data_range=data_range)
    full_rmse = np.sqrt(((cube - cube_estim) ** 2).mean())
    
    print(f"Full Model - PSNR: {full_psnr:.4f} dB | SSIM: {full_ssim:.6f} | RMSE: {full_rmse:.6f}")
    
    if not smoke_test:
        assert 45.5 <= full_psnr <= 47.0, f"Full PSNR {full_psnr:.2f} dB outside baseline expected band"
        
    # ----------------------------------------------------
    # STEP 3: COMPONENT IMPORTANCE (E_r)
    # ----------------------------------------------------
    # E_r = ||U_r||_F * ||V_r||_F for r=0..R-1
    # U_tensor[:, r] is U_r (length H*W)
    # V_tensor[r, :] is V_r (length N_lambda)
    E_r = np.zeros(rank, dtype=np.float64)
    for r in range(rank):
        u_norm = np.linalg.norm(U_tensor[:, r]) # Frobenius norm of spatial factor column
        v_norm = np.linalg.norm(V_tensor[r, :]) # Frobenius norm of spectral factor row
        E_r[r] = u_norm * v_norm
        
    assert np.all(np.isfinite(E_r)), "E_r contains non-finite values"
    
    # Sort components descending by E_r
    sorted_indices = np.argsort(-E_r) # Indices of components sorted descending
    sorted_E_r = E_r[sorted_indices]
    
    # ----------------------------------------------------
    # STEP 4: PROGRESSIVE TOP-K ABLATION
    # ----------------------------------------------------
    k_psnr = np.zeros(rank, dtype=np.float64)
    k_ssim = np.zeros(rank, dtype=np.float64)
    k_rmse = np.zeros(rank, dtype=np.float64)
    
    Y_accum = np.zeros((H * W, nwvl), dtype=np.float64)
    for k_idx in range(rank):
        r_idx = sorted_indices[k_idx]
        u_r = U_tensor[:, r_idx:r_idx+1] # [H*W, 1]
        v_r = V_tensor[r_idx:r_idx+1, :] # [1, N_lambda]
        Y_accum += (u_r @ v_r)
        
        cube_k = Y_accum.reshape(H, W, nwvl).astype(np.float32)
        k_psnr[k_idx] = utils.psnr(cube, cube_k)
        k_ssim[k_idx] = ssim_func(cube, cube_k, channel_axis=-1, data_range=data_range)
        k_rmse[k_idx] = np.sqrt(((cube - cube_k) ** 2).mean())
        
    # Top-k at k=20 must equal full reconstruction
    top20_diff = np.max(np.abs(cube_estim - Y_accum.reshape(H, W, nwvl)))
    print(f"Sanity Check - Max diff between top-20 & full reconstruction: {top20_diff:.8e}")
    assert top20_diff < 1e-5, f"Top-20 reconstruction mismatch: {top20_diff}"
    
    # ----------------------------------------------------
    # STEP 5: PRE-REGISTERED DESCRIPTIVE PLATEAU POINT k*
    # ----------------------------------------------------
    # Pre-registered rule: k* is min k in 1..20 such that PSNR(k) >= PSNR(20) - 0.1 dB
    target_k_psnr = k_psnr[-1] - 0.1
    k_star_idx = np.where(k_psnr >= target_k_psnr)[0][0]
    k_star = int(k_star_idx + 1) # 1-indexed count
    
    print(f"Descriptive Energy-Concentration Point k*: {k_star} (PSNR(k*): {k_psnr[k_star-1]:.4f} dB vs PSNR(20): {k_psnr[-1]:.4f} dB)")
    
    # ----------------------------------------------------
    # STEP 6: COSINE SIMILARITY MATRICES
    # ----------------------------------------------------
    # Spatial cosine similarity: C_spatial[r, s] = u_hat_r^T u_hat_s
    # U_tensor: [H*W, R]
    U_norms = np.linalg.norm(U_tensor, axis=0, keepdims=True) # [1, R]
    U_normalized = U_tensor / np.maximum(U_norms, 1e-12) # [H*W, R]
    C_spatial = U_normalized.T @ U_normalized # [R, R]
    
    # Spectral cosine similarity: C_spectral[r, s] = v_hat_r^T v_hat_s
    # V_tensor: [R, N_lambda]
    V_norms = np.linalg.norm(V_tensor, axis=1, keepdims=True) # [R, 1]
    V_normalized = V_tensor / np.maximum(V_norms, 1e-12) # [R, N_lambda]
    C_spectral = V_normalized @ V_normalized.T # [R, R]
    
    assert C_spatial.shape == (rank, rank), "C_spatial shape mismatch"
    assert C_spectral.shape == (rank, rank), "C_spectral shape mismatch"
    assert np.all(np.isfinite(C_spatial)), "C_spatial non-finite"
    assert np.all(np.isfinite(C_spectral)), "C_spectral non-finite"
    assert np.allclose(np.diag(C_spatial), 1.0, atol=1e-4), "Spatial diagonal not 1"
    assert np.allclose(np.diag(C_spectral), 1.0, atol=1e-4), "Spectral diagonal not 1"
    
    # Compute mean absolute off-diagonal cosine similarity
    off_diag_mask = ~np.eye(rank, dtype=bool)
    mean_abs_offdiag_spatial = float(np.mean(np.abs(C_spatial[off_diag_mask])))
    mean_abs_offdiag_spectral = float(np.mean(np.abs(C_spectral[off_diag_mask])))
    
    print(f"Pairwise Component Correlation Structure:")
    print(f"  Mean Abs Off-Diagonal Spatial Cosine Similarity: {mean_abs_offdiag_spatial:.6f}")
    print(f"  Mean Abs Off-Diagonal Spectral Cosine Similarity: {mean_abs_offdiag_spectral:.6f}")
    
    # ----------------------------------------------------
    # STEP 7: GENERATE 4 VISUALIZATION TYPES
    # ----------------------------------------------------
    out_dir = "results_interpretability_smoke" if smoke_test else "results_interpretability"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Top-5 Spatial Maps + Spectral Curves
    fig, axes = plt.subplots(5, 2, figsize=(10, 15))
    top5_indices = sorted_indices[:5]
    for idx, r_idx in enumerate(top5_indices):
        u_map = U_tensor[:, r_idx].reshape(H, W)
        v_curve = V_tensor[r_idx, :]
        e_val = E_r[r_idx]
        
        # Spatial map
        im = axes[idx, 0].imshow(u_map, cmap='viridis')
        axes[idx, 0].set_title(f"Rank-1 Top #{idx+1} (Comp {r_idx}): Spatial $U_r$\nPost-hoc $E_r$ = {e_val:.4f}", fontsize=10)
        axes[idx, 0].axis('off')
        fig.colorbar(im, ax=axes[idx, 0], fraction=0.046, pad=0.04)
        
        # Spectral curve
        axes[idx, 1].plot(wavelengths, v_curve, color='navy', linewidth=1.5)
        axes[idx, 1].set_title(f"Top #{idx+1} (Comp {r_idx}): Spectral $V_r$", fontsize=10)
        axes[idx, 1].set_xlabel("Wavelength (nm)", fontsize=8)
        axes[idx, 1].set_ylabel("Amplitude", fontsize=8)
        axes[idx, 1].grid(True, linestyle='--', alpha=0.5)
        
    plt.tight_layout()
    top5_path = os.path.join(out_dir, "top5_components.png")
    plt.savefig(top5_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    # 2. Cumulative Progressive Top-k PSNR and SSIM
    fig, ax1 = plt.subplots(figsize=(8, 5))
    k_axis = np.arange(1, rank + 1)
    
    color1 = 'tab:blue'
    ax1.set_xlabel('Top-k Components (Sorted by $E_r$ Descending)', fontsize=11)
    ax1.set_ylabel('PSNR (dB)', color=color1, fontsize=11)
    line1 = ax1.plot(k_axis, k_psnr, color=color1, marker='o', linewidth=2, label='Progressive PSNR')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.axvline(x=k_star, color='red', linestyle='--', label=f'k* = {k_star} (0.1 dB plateau)')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax2 = ax1.twinx()
    color2 = 'tab:green'
    ax2.set_ylabel('SSIM', color=color2, fontsize=11)
    line2 = ax2.plot(k_axis, k_ssim, color=color2, marker='s', linewidth=2, linestyle=':', label='Progressive SSIM')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    lines = line1 + line2 + [ax1.get_legend_handles_labels()[0][1]]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower right')
    
    plt.title('Progressive Reconstruction Quality vs. Top-k Components', fontsize=12)
    plt.tight_layout()
    topk_path = os.path.join(out_dir, "progressive_topk_metrics.png")
    plt.savefig(topk_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    # 3. Spatial Cosine-Similarity Heatmap
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(C_spatial, cmap='coolwarm', vmin=-1.0, vmax=1.0)
    ax.set_title('Spatial Factor Cosine Similarity $C_{spatial}$', fontsize=12)
    ax.set_xlabel('Component Index $r$', fontsize=10)
    ax.set_ylabel('Component Index $s$', fontsize=10)
    ax.set_xticks(np.arange(rank))
    ax.set_yticks(np.arange(rank))
    fig.colorbar(im, ax=ax, label='Cosine Similarity')
    plt.tight_layout()
    spatial_cos_path = os.path.join(out_dir, "spatial_cosine_similarity.png")
    plt.savefig(spatial_cos_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    # 4. Spectral Cosine-Similarity Heatmap
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(C_spectral, cmap='coolwarm', vmin=-1.0, vmax=1.0)
    ax.set_title('Spectral Factor Cosine Similarity $C_{spectral}$', fontsize=12)
    ax.set_xlabel('Component Index $r$', fontsize=10)
    ax.set_ylabel('Component Index $s$', fontsize=10)
    ax.set_xticks(np.arange(rank))
    ax.set_yticks(np.arange(rank))
    fig.colorbar(im, ax=ax, label='Cosine Similarity')
    plt.tight_layout()
    spectral_cos_path = os.path.join(out_dir, "spectral_cosine_similarity.png")
    plt.savefig(spectral_cos_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    # Save npz archive
    npz_path = os.path.join(out_dir, "interpretability_results.npz")
    np.savez_compressed(
        npz_path,
        U=U_tensor,
        V=V_tensor,
        E_r=E_r,
        sorted_indices=sorted_indices,
        sorted_E_r=sorted_E_r,
        k_psnr=k_psnr,
        k_ssim=k_ssim,
        k_rmse=k_rmse,
        k_star=k_star,
        C_spatial=C_spatial,
        C_spectral=C_spectral,
        mean_abs_offdiag_spatial=mean_abs_offdiag_spatial,
        mean_abs_offdiag_spectral=mean_abs_offdiag_spectral
    )
    
    # Write summary text report
    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, 'w') as f:
        f.write("=== EXTENSION 7: FACTOR / COMPONENT INTERPRETABILITY RESULTS ===\n\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Rank R: {rank}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Runtime: {runtime:.2f} s\n")
        f.write(f"Peak VRAM: {vram_peak:.2f} MB\n\n")
        f.write("--- FULL MODEL RECONSTRUCTION METRICS ---\n")
        f.write(f"Full Model PSNR: {full_psnr:.4f} dB\n")
        f.write(f"Full Model SSIM: {full_ssim:.6f}\n")
        f.write(f"Full Model RMSE: {full_rmse:.6f}\n\n")
        f.write("--- COMPONENT IMPORTANCE (E_r = ||U_r||_F * ||V_r||_F) ---\n")
        for idx in range(rank):
            r_orig = sorted_indices[idx]
            f.write(f"Sorted Top #{idx+1:02d}: Orig Comp {r_orig:02d} | E_r = {sorted_E_r[idx]:.6f}\n")
        f.write("\n--- PROGRESSIVE TOP-K RECONSTRUCTION ABLATION ---\n")
        for k_idx in range(rank):
            f.write(f"k={k_idx+1:02d}: PSNR = {k_psnr[k_idx]:.4f} dB | SSIM = {k_ssim[k_idx]:.6f} | RMSE = {k_rmse[k_idx]:.6f}\n")
        f.write(f"\nDescriptive Energy-Concentration Point k*: {k_star} (Threshold: PSNR(20) - 0.1 dB = {target_k_psnr:.4f} dB)\n\n")
        f.write("--- PAIRWISE COMPONENT CORRELATION STRUCTURE ---\n")
        f.write(f"Mean Abs Off-Diagonal Spatial Cosine Similarity: {mean_abs_offdiag_spatial:.6f}\n")
        f.write(f"Mean Abs Off-Diagonal Spectral Cosine Similarity: {mean_abs_offdiag_spectral:.6f}\n")
        
    print(f"Analysis complete. Results saved to {out_dir}/")
    return summary_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extension 7: Factor Interpretability')
    parser.add_argument('--smoke_test', action='store_true', help='Run lightweight 10-epoch smoke test')
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    run_experiment(smoke_test=args.smoke_test, epochs=args.epochs, seed=args.seed)
