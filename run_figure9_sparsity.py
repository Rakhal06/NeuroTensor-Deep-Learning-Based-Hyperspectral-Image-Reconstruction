#!/usr/bin/env python
'''
    Research Extension 2: Factor Group-Sparsity (Group Lasso) Regularization

    Based on the validated DeepTensor baseline (commit a146cc7) and PROJECT_CONTEXT.md.
    This script implements Factor Group-Sparsity (Group Lasso) regularization on the
    original unnormalized DeepTensor factor matrices U and V.

    Core approach:
    - Original unnormalized DeepTensor architecture: Y_hat = U V^T
    - Over-parameterized capacity: R_MAX = 30
    - Joint factor norm for component r: g_r = sqrt(||U_r||_2^2 + ||V_r||_2^2)
    - Group-sparsity regularizer: R(U,V) = sum_{r=1}^{R_MAX} g_r
    - Objective: L = (95% visible L1 reconstruction loss) + lambda * R(U,V)
    - Held-out 5% validation framework (fixed mask, seed 42) for hyperparameter selection
    - Ground-truth PSNR/SSIM/RMSE used strictly for post-hoc evaluation after model selection
'''

import sys
import os
import tqdm
import copy
import time
import argparse

import numpy as np
from scipy import io
from skimage.metrics import structural_similarity as ssim_func

import torch

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
plt.gray()

sys.path.append('modules')

import models
import utils
import losses
import spectral
import deep_prior


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extension 2: Factor Group-Sparsity (Group Lasso) Regularization'
    )
    parser.add_argument(
        '--lambda_value',
        type=float,
        default=0.0,
        help='Group Lasso regularization multiplier lambda (e.g., 0, 1e-4, 1e-3, 1e-2)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for model initialization'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=5000,
        help='Total training epochs (default: 5000)'
    )
    parser.add_argument(
        '--r_max',
        type=int,
        default=30,
        help='Maximum rank capacity R_MAX (default: 30)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results_sparsity',
        help='Output directory for results and checkpoints'
    )
    return parser.parse_args()


def compute_group_regularizer(U, V):
    """
    Computes Factor Group-Sparsity (Group Lasso) Regularizer:
    For each component r in [0, R_MAX-1]:
        U_r = U[0, :, r] (length H*W)
        V_r = V[0, r, :] (length nwvl)
        g_r = sqrt(||U_r||_2^2 + ||V_r||_2^2)
    Regularization: R(U,V) = sum_r g_r

    Args:
        U: Spatial factor tensor of shape (1, H*W, R_MAX)
        V: Spectral factor tensor of shape (1, R_MAX, nwvl)

    Returns:
        reg_loss: scalar torch.Tensor for sum_r g_r
        g_r: torch.Tensor of shape (R_MAX,) containing individual group norms
        norm_U_sq: torch.Tensor of shape (R_MAX,) containing ||U_r||_2^2
        norm_V_sq: torch.Tensor of shape (R_MAX,) containing ||V_r||_2^2
    """
    norm_U_sq = (U[0] ** 2).sum(dim=0)  # (R_MAX,)
    norm_V_sq = (V[0] ** 2).sum(dim=1)  # (R_MAX,)

    # Stable square root computation
    g_r = torch.sqrt(norm_U_sq + norm_V_sq + 1e-12)
    reg_loss = g_r.sum()

    return reg_loss, g_r, norm_U_sq, norm_V_sq


def compute_effective_ranks(norm_U_sq, norm_V_sq):
    """
    Computes Component Power and Effective Ranks:
    power_r = ||U_r||_2 * ||V_r||_2
    Primary Effective Rank: count of components with power_r > 0.01 * max(power)
    Secondary Effective Rank: smallest number of components accounting for >=99% cumulative power
    """
    u_norms = torch.sqrt(norm_U_sq + 1e-12)
    v_norms = torch.sqrt(norm_V_sq + 1e-12)
    powers = u_norms * v_norms  # (R_MAX,)

    powers_np = powers.detach().cpu().numpy()
    max_power = powers_np.max()

    if max_power < 1e-12:
        primary_rank = 0
        secondary_rank = 0
    else:
        # Primary effective rank (> 1% of max power)
        primary_rank = int((powers_np > 0.01 * max_power).sum())

        # Secondary effective rank (>= 99% cumulative power)
        sorted_powers = np.sort(powers_np)[::-1]
        total_power = sorted_powers.sum()
        if total_power < 1e-12:
            secondary_rank = 0
        else:
            cum_power_ratio = np.cumsum(sorted_powers) / total_power
            secondary_rank = int(np.argmax(cum_power_ratio >= (0.99 - 1e-9)) + 1)

    return powers_np, primary_rank, secondary_rank


if __name__ == '__main__':
    args = parse_args()
    lambda_val = args.lambda_value
    model_seed = args.seed
    epochs = args.epochs
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    # ==================== CONFIGURATION ====================
    R_MAX = args.r_max
    n_inputs = 20
    expname = 'icvl'
    MASK_RATIO = 0.05
    nettype = 'dip'
    scaling = 1
    tau = 100
    noise_snr = 2
    learning_rate = 1e-3
    reg_noise_std = 1.0 / 30.0
    exp_weight = 0.99
    SVD_BASELINE_RANK = 20

    print("=" * 70)
    print("EXTENSION 2: FACTOR GROUP-SPARSITY (GROUP LASSO) REGULARIZATION")
    print("=" * 70)
    print(f"Lambda value (--lambda_value) : {lambda_val}")
    print(f"Model seed (--seed)           : {model_seed}")
    print(f"Epochs (--epochs)             : {epochs}")
    print(f"R_MAX                         : {R_MAX}")
    print(f"n_inputs                      : {n_inputs}")
    print(f"Learning rate                 : {learning_rate}")
    print(f"Output directory              : {output_dir}")
    print("=" * 70)

    # ==================== DATA LOADING ====================
    data = io.loadmat('data/%s.mat' % expname)
    cube = data['hypercube'].astype(np.float32)
    wavelengths = data['wavelengths'].astype(np.float32).ravel()
    cube = utils.resize(cube / cube.max(), scaling)

    H, W, nwvl = cube.shape
    print(f"Dataset: {expname} | Shape: ({H}, {W}, {nwvl})")

    hsmat = cube.reshape(H * W, nwvl)
    cube_noisy = utils.measure(cube, noise_snr, tau)

    noisy_psnr = utils.psnr(cube, cube_noisy)
    print(f"Input noisy PSNR: {noisy_psnr:.2f} dB")

    hsten = torch.tensor(hsmat)[None, ...].cuda()
    hsten_noisy = torch.tensor(cube_noisy.reshape(H * W, nwvl))[None, ...].cuda()

    criterion_l1 = losses.L1Norm()

    # ==================== MASK CREATION (FIXED SEED 42) ====================
    # Fixed mask generation seed = 42 across ALL lambda configurations
    np.random.seed(42)
    mask_np = np.random.rand(*cube_noisy.shape) > MASK_RATIO
    mask_train = torch.tensor(mask_np.reshape(H * W, nwvl))[None, ...].cuda()
    mask_val = ~mask_train
    print(f"Mask created (seed=42): 95% Train ({mask_train.sum().item()} pixels), 5% Val ({mask_val.sum().item()} pixels)")

    # ==================== MODEL SEED INITIALIZATION ====================
    # Set model initialization seed
    torch.manual_seed(model_seed)
    torch.cuda.manual_seed_all(model_seed)
    np.random.seed(model_seed)

    # ==================== NETWORK INITIALIZATION ====================
    print(f"\nInitializing unnormalized DIP networks with R_MAX={R_MAX} output channels...")
    if nettype == 'dip':
        im_net = deep_prior.get_net(n_inputs, 'skip', 'reflection',
                                    upsample_mode='bilinear',
                                    skip_n33d=128,
                                    skip_n33u=128,
                                    num_scales=5,
                                    n_channels=R_MAX).cuda()
        spec_net = deep_prior.get_net(n_inputs, 'skip1d', 'reflection',
                                      upsample_mode='linear',
                                      skip_n33d=128,
                                      skip_n33u=128,
                                      num_scales=5,
                                      n_channels=R_MAX).cuda()
        im_inp = utils.get_inp([1, n_inputs, H, W])
        spec_inp = utils.get_inp([1, n_inputs, nwvl])
    else:
        raise ValueError(f"Unsupported nettype: {nettype}")

    im_net.train()
    spec_net.train()

    net_params = list(im_net.parameters()) + list(spec_net.parameters())
    inp_params = [im_inp] + [spec_inp]

    im_inp_per = im_inp.detach().clone()
    spec_inp_per = spec_inp.detach().clone()

    params = net_params + inp_params
    optimizer = torch.optim.Adam(lr=learning_rate, params=params)

    # Logging structures
    train_loss_array = np.zeros(epochs)
    val_loss_array = np.zeros(epochs)
    reg_loss_array = np.zeros(epochs)
    total_loss_array = np.zeros(epochs)

    best_val_loss = float('inf')
    best_epoch = 0
    best_im_net_state = None
    best_spec_net_state = None
    best_im_inp = None
    best_spec_inp = None

    torch.cuda.reset_peak_memory_stats()

    print(f"Starting training for {epochs} epochs with lambda = {lambda_val}...")
    print("-" * 70)

    tic = time.time()
    tbar = tqdm.tqdm(range(epochs))
    for idx in tbar:
        # Perturb inputs (identical to baseline)
        im_inp_perturbed = im_inp + im_inp_per.normal_() * reg_noise_std
        spec_inp_perturbed = spec_inp + spec_inp_per.normal_() * reg_noise_std

        # Original unnormalized DeepTensor forward pass
        U_img = im_net(im_inp_perturbed)                      # (1, R_MAX, H, W)
        V = spec_net(spec_inp_perturbed)                       # (1, R_MAX, nwvl)
        U = U_img.reshape(-1, R_MAX, H * W).permute(0, 2, 1)   # (1, H*W, R_MAX)

        hs_estim = torch.bmm(U, V)                             # (1, H*W, nwvl)

        # Loss computation
        diff = hsten_noisy - hs_estim

        # 95% visible training L1 loss
        train_diff = diff[mask_train]
        train_l1_loss = train_diff.abs().mean()

        # 5% held-out validation L1 loss
        val_diff = diff[mask_val]
        val_l1_loss = val_diff.abs().mean()

        # Group Lasso Regularization on unnormalized factors U and V
        reg_loss, g_r, norm_U_sq, norm_V_sq = compute_group_regularizer(U, V)

        # Total training objective
        total_loss = train_l1_loss + lambda_val * reg_loss

        # Log metrics
        train_loss_array[idx] = train_l1_loss.item()
        val_loss_array[idx] = val_l1_loss.item()
        reg_loss_array[idx] = reg_loss.item()
        total_loss_array[idx] = total_loss.item()

        tbar.set_description(
            f'T-L1={train_l1_loss.item():.4e} | V-L1={val_l1_loss.item():.4e} | '
            f'Reg={reg_loss.item():.4e}'
        )

        # Optimization step on total_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Checkpoint tracking based STRICTLY on held-out validation L1 loss
        if val_l1_loss.item() < best_val_loss:
            best_val_loss = val_l1_loss.item()
            best_epoch = idx
            best_im_net_state = copy.deepcopy(im_net.state_dict())
            best_spec_net_state = copy.deepcopy(spec_net.state_dict())
            best_im_inp = im_inp.detach().clone()
            best_spec_inp = spec_inp.detach().clone()

    toc = time.time()
    runtime = toc - tic

    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # ==================== EVALUATION ON BEST CHECKPOINT ====================
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE — EVALUATING BEST-VALIDATION CHECKPOINT")
    print("=" * 70)

    # Restore best validation checkpoint
    im_net.load_state_dict(best_im_net_state)
    spec_net.load_state_dict(best_spec_net_state)
    im_inp = best_im_inp
    spec_inp = best_spec_inp

    im_net.train()
    spec_net.train()

    with torch.no_grad():
        U_img_best = im_net(im_inp)
        V_best = spec_net(spec_inp)
        U_best = U_img_best.reshape(-1, R_MAX, H * W).permute(0, 2, 1)

        hs_estim_best = torch.bmm(U_best, V_best)

        # Group norms and power metrics on best model
        _, g_r_best, norm_U_sq_best, norm_V_sq_best = compute_group_regularizer(U_best, V_best)
        powers_best, primary_rank, secondary_rank = compute_effective_ranks(norm_U_sq_best, norm_V_sq_best)

    best_hs_estim_np = hs_estim_best.squeeze().cpu().numpy()
    cube_estim = best_hs_estim_np.reshape(H, W, nwvl)
    cube_lr = spectral.lr_decompose(cube_noisy, SVD_BASELINE_RANK)

    data_range = cube.max() - cube.min()

    # Ground-truth evaluation (Post-hoc evaluation ONLY)
    svd_psnr = utils.psnr(cube, cube_lr)
    svd_ssim = ssim_func(cube, cube_lr, channel_axis=-1, data_range=data_range)

    dt_psnr = utils.psnr(cube, cube_estim)
    dt_ssim = ssim_func(cube, cube_estim, channel_axis=-1, data_range=data_range)
    dt_rmse = np.sqrt(((cube - cube_estim) ** 2).mean())

    g_r_list = g_r_best.detach().cpu().numpy().tolist()
    powers_list = powers_best.tolist()

    print(f"Lambda:                    {lambda_val}")
    print(f"Model Seed:                {model_seed}")
    print(f"Best Validation Epoch:     {best_epoch}")
    print(f"Best Validation L1 Loss:   {best_val_loss:.6e}")
    print(f"Primary Effective Rank:    {primary_rank}")
    print(f"Secondary 99% Rank:        {secondary_rank}")
    print(f"DeepTensor PSNR:           {dt_psnr:.2f} dB")
    print(f"DeepTensor SSIM:           {dt_ssim:.4f}")
    print(f"DeepTensor RMSE:           {dt_rmse:.6f}")
    print(f"Runtime:                   {runtime:.2f} seconds")
    print(f"Peak GPU VRAM:             {peak_vram_mb:.2f} MB")
    print("=" * 70)

    # Save detailed text summary
    summary_filename = os.path.join(output_dir, f"summary_lambda_{lambda_val}_seed_{model_seed}.txt")
    with open(summary_filename, 'w') as f:
        f.write("EXTENSION 2: FACTOR GROUP-SPARSITY (GROUP LASSO) REGULARIZATION RESULTS\n")
        f.write("=" * 70 + "\n")
        f.write(f"lambda_value:            {lambda_val}\n")
        f.write(f"model_seed:              {model_seed}\n")
        f.write(f"epochs:                  {epochs}\n")
        f.write(f"runtime_sec:              {runtime:.2f}\n")
        f.write(f"peak_vram_mb:            {peak_vram_mb:.2f}\n")
        f.write(f"best_val_epoch:          {best_epoch}\n")
        f.write(f"best_val_l1_loss:        {best_val_loss:.6e}\n")
        f.write(f"primary_effective_rank:  {primary_rank}\n")
        f.write(f"secondary_effective_rank:{secondary_rank}\n")
        f.write(f"psnr_db:                 {dt_psnr:.4f}\n")
        f.write(f"ssim:                    {dt_ssim:.6f}\n")
        f.write(f"rmse:                    {dt_rmse:.6f}\n")
        f.write(f"input_noisy_psnr:        {noisy_psnr:.4f}\n")
        f.write(f"svd_rank20_psnr:         {svd_psnr:.4f}\n")
        f.write(f"svd_rank20_ssim:         {svd_ssim:.6f}\n")
        f.write("-" * 70 + "\n")
        f.write("GROUP NORMS (g_r):\n")
        f.write(str(g_r_list) + "\n")
        f.write("-" * 70 + "\n")
        f.write("COMPONENT POWERS (power_r):\n")
        f.write(str(powers_list) + "\n")

    print(f"Summary written to {summary_filename}")

    # Plot results
    wvl_idx = nwvl // 2
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(cube[..., wvl_idx], cmap='gray')
    axes[0, 0].set_title('Ground truth')

    axes[0, 1].imshow(cube_lr[..., wvl_idx], cmap='gray')
    axes[0, 1].set_title(f'SVD rank-{SVD_BASELINE_RANK} | {svd_psnr:.1f} dB | {svd_ssim:.2f}')

    axes[0, 2].imshow(cube_estim[..., wvl_idx], cmap='gray')
    axes[0, 2].set_title(f'Group Lasso (λ={lambda_val}) | {dt_psnr:.1f} dB | {dt_ssim:.2f}')

    # Loss curve
    axes[1, 0].semilogy(train_loss_array, label='Train L1 (95%)')
    axes[1, 0].semilogy(val_loss_array, label='Val L1 (5%)', alpha=0.7)
    axes[1, 0].set_title('Reconstruction L1 Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('L1 Loss')
    axes[1, 0].legend()

    # Component power distribution
    axes[1, 1].bar(range(1, R_MAX + 1), powers_list, color='navy')
    axes[1, 1].set_title(f'Component Powers (R_eff={primary_rank})')
    axes[1, 1].set_xlabel('Component Index r')
    axes[1, 1].set_ylabel('Power')

    # Regularization loss curve
    axes[1, 2].plot(reg_loss_array, color='darkgreen')
    axes[1, 2].set_title('Group Regularizer R(U,V)')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('R(U,V)')

    plt.suptitle(f'Group Lasso Regularization — λ: {lambda_val}, R_primary: {primary_rank}, '
                 f'PSNR: {dt_psnr:.2f} dB, SSIM: {dt_ssim:.4f}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plot_filename = os.path.join(output_dir, f"plot_lambda_{lambda_val}_seed_{model_seed}.png")
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {plot_filename}\n")
