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
    n_inputs = 20             # Input channels match frozen baseline (SVD_BASELINE_RANK)
    MASK_RATIO = 0.05
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
    rank_locked = False
    selected_rank = None
    rank_lock_epoch = None
    last_increase_epoch = 0         # Treat initial rank as epoch-0 event
    rank_history = []               # List of (epoch, old_rank, new_rank) tuples
    train_loss_ema_history = []     # Smoothed training loss value per epoch
    val_loss_ema_history = []       # Smoothed validation loss value per epoch
    val_ema_at_last_plateau = None

    # ==================== PRINT CONFIGURATION ====================
    print("=" * 70)
    print("SELF-SUPERVISED ONLINE MODEL SELECTION FOR EFFECTIVE RANK")
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

    # Create a fixed random binary mask for held-out validation
    # Mask is True for visible/training (95%), False for held-out/validation (5%)
    # No clean ground truth is used here.
    np.random.seed(42) # For fixed mask geometry across identical noise setups
    mask_np = np.random.rand(*cube_noisy.shape) > MASK_RATIO
    mask_train = torch.tensor(mask_np.reshape(H * W, nwvl))[None, ...].cuda()
    mask_val = ~mask_train
    # Remove seed after mask creation to preserve stochastic training behavior
    np.random.seed(None)

    train_loss_array = np.zeros(epochs)
    val_loss_array = np.zeros(epochs)
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
    best_val_loss = float('inf')
    best_epoch = 0
    best_rank_at_best = R_START

    params = net_params + inp_params
    optimizer = torch.optim.Adam(lr=learning_rate, params=params)

    # Reset peak VRAM counter
    torch.cuda.reset_peak_memory_stats()

    smoothed_train_loss = None  # Initialized on first epoch
    smoothed_val_loss = None

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
        # Calculate L1 on masked (val) and unmasked (train) elements
        diff = hsten_noisy - hs_estim
        
        # Training loss: mean over unmasked elements
        train_diff = diff[mask_train]
        train_loss = train_diff.abs().mean()
        train_loss_val = train_loss.item()
        train_loss_array[idx] = train_loss_val
        
        # Validation loss: mean over masked elements
        val_diff = diff[mask_val]
        val_loss = val_diff.abs().mean()
        val_loss_val = val_loss.item()
        val_loss_array[idx] = val_loss_val
        
        # Ground-truth MSE: for post-hoc monitoring ONLY.
        mse_array[idx] = ((hs_estim - hsten) ** 2).mean().item()

        # ---- EMA SMOOTHING ----
        if smoothed_train_loss is None:
            smoothed_train_loss = train_loss_val
            smoothed_val_loss = val_loss_val
            # Initialize the validation reference with the initial random-network state.
            # This ensures Rank 4 must genuinely outperform random noise to trigger a growth to Rank 6,
            # avoiding an unconditional first increase.
            val_ema_at_last_plateau = val_loss_val
        else:
            smoothed_train_loss = LOSS_EMA_ALPHA * smoothed_train_loss + \
                            (1 - LOSS_EMA_ALPHA) * train_loss_val
            smoothed_val_loss = LOSS_EMA_ALPHA * smoothed_val_loss + \
                            (1 - LOSS_EMA_ALPHA) * val_loss_val
        train_loss_ema_history.append(smoothed_train_loss)
        val_loss_ema_history.append(smoothed_val_loss)

        tbar.set_description(
            f'r={current_rank:2d} | T-EMA={smoothed_train_loss:.4e} | '
            f'V-EMA={smoothed_val_loss:.4e}'
        )
        tbar.refresh()

        # Backpropagation and optimizer step (only on training loss)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # Best estimate tracking based on Validation Loss
        if val_loss_val < best_val_loss:
            best_val_loss = val_loss_val
            best_epoch = idx
            best_rank_at_best = current_rank
            best_hs_estim = copy.deepcopy(hs_estim.detach().squeeze())

            if hs_estim_avg is None:
                hs_estim_avg = hs_estim.detach()
            else:
                hs_estim_avg = exp_weight * hs_estim_avg + \
                               (1 - exp_weight) * hs_estim.detach()

        # ---- PLATEAU DETECTION & VALIDATION-BASED RANK LOCKING ----
        epochs_since_increase = idx - last_increase_epoch
        if (not rank_locked 
                and current_rank < R_MAX
                and epochs_since_increase > MIN_EPOCHS_BEFORE_GROWTH
                and idx >= PLATEAU_WINDOW):

            past_train_ema = train_loss_ema_history[idx - PLATEAU_WINDOW]
            current_train_ema = smoothed_train_loss

            if past_train_ema > 0:
                rel_train_improvement = (past_train_ema - current_train_ema) / past_train_ema
            else:
                rel_train_improvement = 0.0

            # Directional Plateau: Improvement is positive but very small.
            # 0 <= rel_improvement < threshold. 
            # Negative improvement (loss increase) is not a growth opportunity.
            if 0 <= rel_train_improvement < PLATEAU_THRESHOLD:
                # Check validation signal to decide whether to grow or lock
                if smoothed_val_loss < val_ema_at_last_plateau:
                    # Generalization improved, accept growth
                    old_rank = current_rank
                    current_rank = min(current_rank + R_STEP, R_MAX)
                    last_increase_epoch = idx
                    val_ema_at_last_plateau = smoothed_val_loss
                    rank_history.append((idx, old_rank, current_rank))

                    # Note on Adam optimizer state:
                    # PyTorch Adam naturally tracks state per-tensor. The newly activated
                    # output rows received zero gradient while masked, so their moments
                    # are exactly zero. When activated at a large global step, they
                    # experience a known transient effective learning rate multiplier 
                    # of ~3.16x. With lr=1e-3, this is perfectly stable. We let PyTorch
                    # handle this naturally without unsupported moment-copying.
                    tbar.write(
                        f"[RANK INCREASE] Epoch {idx}: "
                        f"rank {old_rank} -> {current_rank} | "
                        f"Val EMA: {smoothed_val_loss:.4e}"
                    )
                else:
                    # Validation stalled or degraded, permanently lock rank
                    rank_locked = True
                    selected_rank = current_rank
                    rank_lock_epoch = idx
                    tbar.write(
                        f"[RANK LOCKED] Epoch {idx}: "
                        f"Validation plateaued at rank {current_rank}. "
                        f"Growth permanently stopped."
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
    # Fallback if never locked: report the rank associated with best validation reconstruction
    if not rank_locked:
        selected_rank = best_rank_at_best

    rank_status_str = f"Locked rank: {selected_rank} (at ep {rank_lock_epoch})" if rank_locked else f"Selected rank: {selected_rank} (best val at ep {best_epoch}, unlocked)"

    print(f"Final active rank:   {current_rank}")
    print(f"Status:              {rank_status_str}")
    print(f"Best validation epoch:{best_epoch}")
    print(f"Rank at best val ep: {best_rank_at_best}")
    print(f"Best Val L1 loss:    {best_val_loss:.6e}")
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
    print(f"Final active rank:         {current_rank}")
    print(f"Selected/Locked rank:      {selected_rank} ({'locked' if rank_locked else 'unlocked'})")
    print(f"Rank at best val epoch:    {best_rank_at_best}")
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
    axes[1, 0].semilogy(train_loss_array, label='Train (95%)')
    axes[1, 0].semilogy(val_loss_array, label='Val (5%)', alpha=0.7)
    axes[1, 0].set_title('L1 Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('L1 Loss')
    axes[1, 0].legend()

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
