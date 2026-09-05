#!/usr/bin/env python

'''
    Research Extension 1: Adaptive Rank Selection — Plateau-Based

    Based on the validated DeepTensor baseline (commit a146cc7).
    This script implements adaptive rank selection using a plateau-based
    mechanism. Instead of fixing rank=20, the effective rank starts small
    and grows when the reconstruction loss plateaus.

    Core approach:
    - Networks are initialized with R_MAX output channels (over-parameterization)
    - During reconstruction, only the first 'current_rank' channels are used
      via tensor slicing after the full forward pass
    - Inactive output channels are excluded from the reconstruction, so the
      final output-layer weights for those channels receive zero gradient.
      Internal/shared network layers still receive gradients from the active
      channels and continue to be trained normally.
    - When the EMA-smoothed L1 loss improvement plateaus, active rank increases
    - At rank growth, Adam optimizer state (exp_avg, exp_avg_sq) is explicitly
      reset for the newly activated final-layer output parameters to ensure
      a clean learning start
    - All previously learned parameters and optimizer states for existing
      active channels are fully preserved
    - No network rebuilding or architecture changes occur at rank growth

    Rank-growth decisions depend ONLY on the self-supervised L1 reconstruction
    loss against the noisy observation. No clean ground-truth information is
    used in any training or rank-selection decision.

    Reproducibility note: No random seed is set, matching the baseline protocol.
    Results are stochastic due to random noise generation, network initialization,
    and input perturbation. Multiple runs are needed for statistical significance.

    Branch: extension/adaptive-rank
    Baseline: main / commit a146cc7
'''

import sys
import tqdm
import copy
import time

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

if __name__ == '__main__':
    # ==================== ADAPTIVE RANK CONFIGURATION ====================
    R_START = 4                     # Initial active rank
    R_MAX = 30                      # Maximum rank capacity (network output channels)
    R_STEP = 2                      # Rank increment per plateau event
    PLATEAU_WINDOW = 200            # Sliding window (epochs) for plateau detection
    PLATEAU_THRESHOLD = 1e-3        # Relative improvement threshold below which
                                    # a plateau is declared
    MIN_EPOCHS_BEFORE_GROWTH = 300  # Cooldown epochs after each rank increase.
                                    # Each rank increase adds only R_STEP=2 output
                                    # channels — a small perturbation. 300 epochs
                                    # is sufficient for these channels to receive
                                    # gradients and stabilize.
                                    # Reachability: from R_START=4 to R_MAX=30
                                    # requires 13 increases. With cooldown=300,
                                    # the maximum epochs consumed by rank growth
                                    # is 13 * 301 = 3913, leaving 1087 epochs
                                    # for final convergence. R_MAX is reachable.
    LOSS_EMA_ALPHA = 0.99           # Exponential moving average decay for loss
    TOTAL_EPOCHS = 5000             # Total training epochs
    SVD_BASELINE_RANK = 20          # SVD rank for baseline comparison

    # ==================== BASELINE CONFIGURATION (FROZEN) ====================
    # These parameters are identical to the validated baseline run_figure9.py
    expname = 'icvl'
    n_inputs = R_MAX          # Input channels match max capacity
    init_nconv = 128
    num_channels_up = 5
    nettype = 'dip'

    scaling = 1
    tau = 100
    noise_snr = 2

    # Learning constants (identical to baseline)
    learning_rate = 1e-3
    epochs = TOTAL_EPOCHS
    reg_noise_std = 1.0/30.0
    exp_weight = 0.99

    # ==================== ADAPTIVE RANK STATE ====================
    current_rank = R_START
    last_increase_epoch = 0         # Treat initial rank as epoch-0 event
    rank_history = []               # List of (epoch, old_rank, new_rank) tuples
    loss_ema_history = []           # Smoothed loss value per epoch

    # ==================== PRINT CONFIGURATION ====================
    print("=" * 70)
    print("ADAPTIVE RANK SELECTION — PLATEAU-BASED")
    print("=" * 70)
    print(f"R_START                  = {R_START}")
    print(f"R_MAX                    = {R_MAX}")
    print(f"R_STEP                   = {R_STEP}")
    print(f"PLATEAU_WINDOW           = {PLATEAU_WINDOW}")
    print(f"PLATEAU_THRESHOLD        = {PLATEAU_THRESHOLD}")
    print(f"MIN_EPOCHS_BEFORE_GROWTH = {MIN_EPOCHS_BEFORE_GROWTH}")
    print(f"LOSS_EMA_ALPHA           = {LOSS_EMA_ALPHA}")
    print(f"TOTAL_EPOCHS             = {TOTAL_EPOCHS}")
    print(f"SVD_BASELINE_RANK        = {SVD_BASELINE_RANK}")
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

    # Report noisy input quality
    noisy_psnr = utils.psnr(cube, cube_noisy)
    print(f"Input noisy PSNR: {noisy_psnr:.2f} dB")

    hsten = torch.tensor(hsmat)[None, ...].cuda()
    hsten_noisy = torch.tensor(cube_noisy.reshape(H * W, nwvl))[None, ...].cuda()

    criterion_l1 = losses.L1Norm()

    loss_array = np.zeros(epochs)
    mse_array = np.zeros(epochs)
    rank_array = np.zeros(epochs, dtype=int)

    # ==================== NETWORK INITIALIZATION ====================
    # Networks are initialized with R_MAX output channels.
    # During reconstruction, only the first current_rank channels are used.
    print(f"\nInitializing DIP networks with R_MAX={R_MAX} output channels...")

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
        raise ValueError(f"Only 'dip' nettype is supported. Got: {nettype}")

    # ---- Identify final output-layer parameters for Adam state management ----
    # The skip network ends with a 1x1 conv: Conv2d(128, R_MAX, 1) for im_net,
    # Conv1d(128, R_MAX, 1) for spec_net. These are the last learnable params.
    # At rank growth, we reset Adam's running averages for newly activated
    # output channels of these layers to ensure a clean learning start.
    im_net_params_list = list(im_net.parameters())
    spec_net_params_list = list(spec_net.parameters())
    # Final conv weight: [R_MAX, 128, 1, 1] (2D) or [R_MAX, 128, 1] (1D)
    # Final conv bias: [R_MAX]
    im_final_weight = im_net_params_list[-2]
    im_final_bias = im_net_params_list[-1]
    spec_final_weight = spec_net_params_list[-2]
    spec_final_bias = spec_net_params_list[-1]
    assert im_final_weight.shape[0] == R_MAX, \
        f"im_net final weight dim0={im_final_weight.shape[0]}, expected {R_MAX}"
    assert spec_final_weight.shape[0] == R_MAX, \
        f"spec_net final weight dim0={spec_final_weight.shape[0]}, expected {R_MAX}"
    final_conv_params = [im_final_weight, im_final_bias,
                         spec_final_weight, spec_final_bias]

    # Switch to training mode
    im_net.train()
    spec_net.train()

    net_params = list(im_net.parameters()) + list(spec_net.parameters())
    inp_params = [im_inp] + [spec_inp]

    im_inp_per = im_inp.detach().clone()
    spec_inp_per = spec_inp.detach().clone()

    hs_estim_avg = None
    best_loss = float('inf')
    best_epoch = 0
    best_rank_at_best = R_START

    params = net_params + inp_params
    optimizer = torch.optim.Adam(lr=learning_rate, params=params)

    # Reset peak VRAM counter
    torch.cuda.reset_peak_memory_stats()

    smoothed_loss = None  # Initialized on first epoch

    print(f"Starting training with initial active rank = {current_rank}")
    print("-" * 70)

    # ==================== TRAINING LOOP ====================
    tic = time.time()
    tbar = tqdm.tqdm(range(epochs))
    for idx in tbar:
        # Record current rank for this epoch
        rank_array[idx] = current_rank

        # Perturb inputs (identical to baseline)
        im_inp_perturbed = im_inp + im_inp_per.normal_() * reg_noise_std
        spec_inp_perturbed = spec_inp + spec_inp_per.normal_() * reg_noise_std

        # Forward pass: networks produce full R_MAX-channel outputs
        U_img = im_net(im_inp_perturbed)          # (1, R_MAX, H, W)
        V = spec_net(spec_inp_perturbed)           # (1, R_MAX, nwvl)

        # Reshape spatial factor to matrix form
        U = U_img.reshape(-1, R_MAX, H * W).permute(0, 2, 1)  # (1, H*W, R_MAX)

        # ---- ACTIVE RANK MASKING ----
        # Only the first current_rank components participate in reconstruction.
        # The final output-layer weights for channels [current_rank : R_MAX]
        # receive zero gradient since they are excluded from the loss computation.
        # Internal/shared network layers still receive gradients through the
        # active channels and continue to be trained normally.
        U_active = U[:, :, :current_rank]          # (1, H*W, current_rank)
        V_active = V[:, :current_rank, :]          # (1, current_rank, nwvl)

        hs_estim = torch.bmm(U_active, V_active)  # (1, H*W, nwvl)

        # Loss computation (self-supervised against noisy observation)
        # This L1 loss is the ONLY signal used for training and rank decisions.
        loss = criterion_l1(hsten_noisy - hs_estim)
        loss_val = loss.item()
        loss_array[idx] = loss_val
        # Ground-truth MSE: for post-hoc monitoring and plotting ONLY.
        # Does NOT contribute to gradients (.item() detaches from graph).
        # Does NOT influence rank-growth decisions (only L1 loss EMA is used).
        # Does NOT trigger early stopping.
        # The baseline (run_figure9.py) computes this identically.
        mse_array[idx] = ((hs_estim - hsten) ** 2).mean().item()

        # ---- EMA SMOOTHING ----
        if smoothed_loss is None:
            smoothed_loss = loss_val
        else:
            smoothed_loss = LOSS_EMA_ALPHA * smoothed_loss + \
                            (1 - LOSS_EMA_ALPHA) * loss_val
        loss_ema_history.append(smoothed_loss)

        tbar.set_description(
            f'r={current_rank:2d} | L1={loss_val:.4e} | '
            f'EMA={smoothed_loss:.4e} | MSE={mse_array[idx]:.4e}'
        )
        tbar.refresh()

        # Backpropagation and optimizer step (identical to baseline)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Best estimate tracking (same logic as baseline)
        if loss_val < best_loss:
            best_loss = loss_val
            best_epoch = idx
            best_rank_at_best = current_rank
            best_hs_estim = copy.deepcopy(hs_estim.detach().squeeze())

            # Exponential averaging (same as baseline)
            if hs_estim_avg is None:
                hs_estim_avg = hs_estim.detach()
            else:
                hs_estim_avg = exp_weight * hs_estim_avg + \
                               (1 - exp_weight) * hs_estim.detach()

        # ---- PLATEAU DETECTION & RANK GROWTH ----
        # Check conditions:
        #   1. current_rank < R_MAX (room to grow)
        #   2. Enough epochs since last increase (cooldown)
        #   3. Enough total epochs for window comparison
        epochs_since_increase = idx - last_increase_epoch
        if (current_rank < R_MAX
                and epochs_since_increase > MIN_EPOCHS_BEFORE_GROWTH
                and idx >= PLATEAU_WINDOW):

            past_ema = loss_ema_history[idx - PLATEAU_WINDOW]
            current_ema = smoothed_loss

            # Relative improvement over the window
            if past_ema > 0:
                relative_improvement = (past_ema - current_ema) / past_ema
            else:
                relative_improvement = 0.0

            # If improvement is below threshold, the loss has plateaued
            if relative_improvement < PLATEAU_THRESHOLD:
                old_rank = current_rank
                current_rank = min(current_rank + R_STEP, R_MAX)
                last_increase_epoch = idx
                rank_history.append((idx, old_rank, current_rank))

                # ---- RESET ADAM STATE FOR NEWLY ACTIVATED CHANNELS ----
                # The final output-layer parameters for channels [old_rank:current_rank]
                # have received zero gradient while masked. Their Adam first moment
                # (exp_avg) and second moment (exp_avg_sq) are exactly zero.
                # We explicitly reset them to ensure deterministic behavior.
                #
                # Note: Adam's step counter is per-parameter (not per-element),
                # so it cannot be reset for individual output channels. With a
                # high step count and freshly zeroed moments, the bias-corrected
                # effective learning rate for new channels is ~2x the nominal lr
                # for the first few steps. This modest amplification self-corrects
                # as the moments accumulate real gradient statistics, and does not
                # affect the already-active channels.
                for param in final_conv_params:
                    state = optimizer.state.get(param)
                    if state and 'exp_avg' in state:
                        state['exp_avg'][old_rank:current_rank] = 0
                        state['exp_avg_sq'][old_rank:current_rank] = 0

                tbar.write(
                    f"[RANK INCREASE] Epoch {idx}: "
                    f"rank {old_rank} -> {current_rank} | "
                    f"rel_improvement={relative_improvement:.6f} "
                    f"< threshold={PLATEAU_THRESHOLD} | "
                    f"Adam state reset for channels [{old_rank}:{current_rank}]"
                )

    toc = time.time()
    runtime = toc - tic

    # Peak VRAM
    peak_vram_bytes = torch.cuda.max_memory_allocated()
    peak_vram_mb = peak_vram_bytes / (1024 * 1024)

    # ==================== EVALUATION ====================
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Final active rank:   {current_rank}")
    print(f"Best epoch:          {best_epoch}")
    print(f"Rank at best epoch:  {best_rank_at_best}")
    print(f"Best L1 loss:        {best_loss:.6e}")
    print(f"Runtime:             {runtime:.2f} seconds")
    print(f"Peak GPU VRAM:       {peak_vram_mb:.2f} MB")

    cube_estim = best_hs_estim.cpu().numpy().reshape(H, W, nwvl)
    cube_lr = spectral.lr_decompose(cube_noisy, SVD_BASELINE_RANK)

    data_range = cube.max() - cube.min()

    # SVD baseline metrics (rank-20 for direct comparison with baseline)
    svd_psnr = utils.psnr(cube, cube_lr)
    svd_ssim = ssim_func(cube, cube_lr, channel_axis=-1, data_range=data_range)

    # Adaptive DeepTensor metrics
    dt_psnr = utils.psnr(cube, cube_estim)
    dt_ssim = ssim_func(cube, cube_estim, channel_axis=-1, data_range=data_range)

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Input noisy PSNR:          {noisy_psnr:.2f} dB")
    print(f"SVD rank-{SVD_BASELINE_RANK} PSNR:          {svd_psnr:.2f} dB")
    print(f"SVD rank-{SVD_BASELINE_RANK} SSIM:          {svd_ssim:.4f}")
    print(f"Adaptive DeepTensor PSNR:  {dt_psnr:.2f} dB")
    print(f"Adaptive DeepTensor SSIM:  {dt_ssim:.4f}")
    print(f"Final selected rank:       {current_rank}")
    print(f"Rank at best epoch:        {best_rank_at_best}")
    print(f"Total rank increases:      {len(rank_history)}")
    print(f"Runtime:                   {runtime:.2f} seconds")
    print(f"Peak GPU VRAM:             {peak_vram_mb:.2f} MB")
    print(f"Total epochs:              {epochs}")

    if rank_history:
        print("\nRank change log:")
        for epoch_num, old_r, new_r in rank_history:
            print(f"  Epoch {epoch_num:5d}: rank {old_r:2d} -> {new_r:2d}")
    else:
        print("\nNo rank changes occurred during training.")

    print("=" * 70)

    # ==================== SAVE VISUALIZATION ====================
    wvl_idx = nwvl // 2

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(cube[..., wvl_idx], cmap='gray')
    axes[0, 0].set_title('Ground truth')

    axes[0, 1].imshow(cube_lr[..., wvl_idx], cmap='gray')
    axes[0, 1].set_title(f'SVD rank-{SVD_BASELINE_RANK} | '
                         f'{svd_psnr:.1f} dB | {svd_ssim:.2f}')

    axes[0, 2].imshow(cube_estim[..., wvl_idx], cmap='gray')
    axes[0, 2].set_title(f'Adaptive DT (r={best_rank_at_best}) | '
                         f'{dt_psnr:.1f} dB | {dt_ssim:.2f}')

    # Loss curve
    axes[1, 0].semilogy(loss_array)
    axes[1, 0].set_title('L1 Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('L1 Loss')

    # Rank evolution
    axes[1, 1].plot(rank_array, linewidth=2)
    axes[1, 1].set_title('Active Rank Evolution')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Active Rank')
    axes[1, 1].set_ylim([0, R_MAX + 2])

    # MSE curve (vs ground truth, for reference only)
    axes[1, 2].semilogy(mse_array)
    axes[1, 2].set_title('MSE (vs ground truth)')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('MSE')

    plt.suptitle(f'Adaptive Rank Selection — Final rank: {current_rank}, '
                 f'PSNR: {dt_psnr:.2f} dB, SSIM: {dt_ssim:.4f}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/adaptive_rank_results.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to results/adaptive_rank_results.png")
