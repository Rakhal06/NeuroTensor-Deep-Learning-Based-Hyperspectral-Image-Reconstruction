#!/usr/bin/env python
'''
    Extension 3: Missing-Data / Tensor Completion Capability for DeepTensor

    FROZEN DESIGN:
    - Adds mask-aware reconstruction loss to evaluate DeepTensor on tensor completion.
    - DeepTensor architecture is unchanged (Fixed R = 20, DIP generators).
    - Data: ICVL hyperspectral image cube (1000x1000x31 or downsampled).
    - Network input: Latent noise grids (im_inp, spec_inp) ONLY. The observation tensor
      is used solely as the target in loss computation.
    - Missingness Masking:
      - Uses scoped NumPy RNG (default_rng(seed)) to generate nested missing masks:
        mask_missing = u < missing_ratio (80% missing superset of 50%, superset of 20%).
      - Observed entries (1 - mask_missing) are split into 95% training and 5% validation.
      - Missing entries contribute ZERO to training and validation losses.
    - Checkpoint Selection: Strictly based on self-supervised validation L1 loss (mask_val).
    - Final Evaluation: Restores best checkpoint in train mode (im_net.train(), spec_net.train())
      and computes Full, Missing-only, and Observed-only PSNR/RMSE against clean ground truth.
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
        description='Extension 3: DeepTensor Missing-Data / Tensor Completion'
    )
    parser.add_argument(
        '--missing_ratio',
        type=float,
        default=0.20,
        help='Missing entry fraction p_missing in [0.0, 1.0) (default: 0.20)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for mask generation and model initialization (default: 42)'
    )
    parser.add_argument(
        '--clean',
        action='store_true',
        default=False,
        help='If set, evaluate tensor completion on clean data without measurement noise'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=5000,
        help='Total training epochs (default: 5000)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results_completion',
        help='Output directory for results and summary files'
    )
    return parser.parse_args()


def generate_completion_masks(cube_shape, missing_ratio, seed=42, val_ratio=0.05):
    """
    Generates nested missing-data masks and observed train/validation splits.

    RNG Specification:
    - Uses a scoped NumPy generator `rng = np.random.default_rng(seed)`.
    - First draw `u = rng.random(cube_shape)` defines missing entries: `u < missing_ratio`.
      This guarantees NESTED missing masks (80% missing is a superset of 50%, superset of 20%).
    - Second draw `val_draw = rng.random(cube_shape)` from the SAME `rng` splits observed
      entries into 95% training and 5% validation (`val_draw < val_ratio`).

    Returns:
        mask_train_np: boolean ndarray (True for observed training pixels)
        mask_val_np:   boolean ndarray (True for observed validation pixels)
        mask_missing_np: boolean ndarray (True for missing pixels)
    """
    rng = np.random.default_rng(seed)

    # First draw: missing entry mask (nested property across missing_ratio)
    u = rng.random(cube_shape)
    mask_missing_np = (u < missing_ratio)
    mask_obs_np = ~mask_missing_np

    # Second draw: split observed entries into 95% train and 5% val
    val_draw = rng.random(cube_shape)
    val_cond = (val_draw < val_ratio)

    mask_val_np = mask_obs_np & val_cond
    mask_train_np = mask_obs_np & ~val_cond

    # Runtime assertion checks for strict disjointness and full coverage
    assert (mask_train_np & mask_missing_np).sum() == 0, "Leakage: Train mask contains missing entries!"
    assert (mask_val_np & mask_missing_np).sum() == 0, "Leakage: Val mask contains missing entries!"
    assert (mask_train_np & mask_val_np).sum() == 0, "Leakage: Train and Val masks overlap!"
    assert (mask_train_np | mask_val_np | mask_missing_np).sum() == mask_train_np.size, \
        "Coverage Error: Union of masks does not cover the full tensor exactly once."

    return mask_train_np, mask_val_np, mask_missing_np


def compute_subset_metrics(cube_gt, cube_estim, mask_np):
    """
    Computes PSNR and RMSE on a specific spatial/spectral boolean subset mask_np.
    Note: SSIM is NOT computed on arbitrary boolean subsets because SSIM relies
    on contiguous 2D spatial windowing, which is ill-defined for scattered subsets.
    """
    count = mask_np.sum()
    if count == 0:
        return float('nan'), float('nan')

    diff = cube_gt[mask_np] - cube_estim[mask_np]
    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)

    gt_max = cube_gt.max()
    if mse < 1e-12:
        psnr_val = 100.0
    else:
        psnr_val = 10.0 * np.log10((gt_max ** 2) / mse)

    return psnr_val, rmse


if __name__ == '__main__':
    args = parse_args()
    missing_ratio = args.missing_ratio
    seed = args.seed
    clean_mode = args.clean
    epochs = args.epochs
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    # DeepTensor Baseline Parameters (Frozen)
    expname = 'icvl'
    rank = 20
    n_inputs = rank
    init_nconv = 128
    nettype = 'dip'
    scaling = 1
    tau = 100
    noise_snr = 2
    learning_rate = 1e-3
    reg_noise_std = 1.0 / 30.0
    exp_weight = 0.99
    SVD_BASELINE_RANK = 20

    print("=" * 70)
    print("EXTENSION 3: MISSING-DATA / TENSOR COMPLETION CAPABILITY")
    print("=" * 70)
    print(f"Missing Ratio (--missing_ratio) : {missing_ratio:.2f}")
    print(f"Seed (--seed)                   : {seed}")
    print(f"Clean Mode (--clean)            : {clean_mode}")
    print(f"Epochs (--epochs)               : {epochs}")
    print(f"Rank R                          : {rank}")
    print(f"Learning rate                   : {learning_rate}")
    print(f"Output Directory                : {output_dir}")
    print("=" * 70)

    # ==================== DATA LOADING & PREPROCESSING ====================
    data = io.loadmat('data/%s.mat' % expname)
    cube = data['hypercube'].astype(np.float32)
    wavelengths = data['wavelengths'].astype(np.float32).ravel()
    cube = utils.resize(cube / cube.max(), scaling)

    H, W, nwvl = cube.shape
    print(f"Dataset: {expname} | Shape: ({H}, {W}, {nwvl}) | Total elements: {cube.size}")

    hsmat = cube.reshape(H * W, nwvl)

    # Measurement noise condition
    if clean_mode:
        print("Condition: CLEAN (No measurement noise added)")
        cube_target = cube.copy()
    else:
        print(f"Condition: NOISY (SNR={noise_snr}, tau={tau})")
        cube_target = utils.measure(cube, noise_snr, tau)

    noisy_psnr = utils.psnr(cube, cube_target)
    print(f"Target input PSNR (vs Clean GT): {noisy_psnr:.2f} dB")

    hsten = torch.tensor(hsmat)[None, ...].cuda()
    hsten_target = torch.tensor(cube_target.reshape(H * W, nwvl))[None, ...].cuda()

    criterion_l1 = losses.L1Norm()

    # ==================== MASK GENERATION ====================
    mask_train_np, mask_val_np, mask_missing_np = generate_completion_masks(
        cube.shape, missing_ratio, seed=seed, val_ratio=0.05
    )

    n_total = mask_train_np.size
    n_missing = mask_missing_np.sum()
    n_obs = n_total - n_missing
    n_train = mask_train_np.sum()
    n_val = mask_val_np.sum()

    print("\nMask Statistics:")
    print(f"  Total elements   : {n_total:10d}")
    print(f"  Observed elements: {n_obs:10d} ({n_obs / n_total * 100:.2f}%)")
    print(f"  Train elements   : {n_train:10d} ({n_train / n_total * 100:.2f}%)")
    print(f"  Val elements     : {n_val:10d} ({n_val / n_total * 100:.2f}%)")
    print(f"  Missing elements : {n_missing:10d} ({n_missing / n_total * 100:.2f}%)")

    # Reshape masks to 2D matrix shape (1, H*W, nwvl) for PyTorch indexing
    mask_train = torch.tensor(mask_train_np.reshape(H * W, nwvl))[None, ...].cuda()
    mask_val = torch.tensor(mask_val_np.reshape(H * W, nwvl))[None, ...].cuda()

    # ==================== MODEL SEED INITIALIZATION ====================
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # ==================== NETWORK INITIALIZATION ====================
    print(f"\nInitializing DIP networks with rank R={rank}...")
    if nettype == 'dip':
        im_net = deep_prior.get_net(n_inputs, 'skip', 'reflection',
                                    upsample_mode='bilinear',
                                    skip_n33d=128,
                                    skip_n33u=128,
                                    num_scales=5,
                                    n_channels=rank).cuda()
        spec_net = deep_prior.get_net(n_inputs, 'skip1d', 'reflection',
                                      upsample_mode='linear',
                                      skip_n33d=128,
                                      skip_n33u=128,
                                      num_scales=5,
                                      n_channels=rank).cuda()
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

    best_val_loss = float('inf')
    best_epoch = 0
    best_im_net_state = None
    best_spec_net_state = None
    best_im_inp = None
    best_spec_inp = None

    torch.cuda.reset_peak_memory_stats()

    print(f"\nStarting training for {epochs} epochs...")
    print("-" * 70)

    tic = time.time()
    tbar = tqdm.tqdm(range(epochs))
    for idx in tbar:
        # Perturb inputs (DeepTensor input regularization)
        im_inp_perturbed = im_inp + im_inp_per.normal_() * reg_noise_std
        spec_inp_perturbed = spec_inp + spec_inp_per.normal_() * reg_noise_std

        # Network forward pass (takes ONLY latent noise vectors, NEVER observed image data)
        U_img = im_net(im_inp_perturbed)                      # (1, rank, H, W)
        V = spec_net(spec_inp_perturbed)                       # (1, rank, nwvl)
        U = U_img.reshape(-1, rank, H * W).permute(0, 2, 1)    # (1, H*W, rank)

        hs_estim = torch.bmm(U, V)                             # (1, H*W, nwvl)

        # Difference against observed target tensor
        diff = hsten_target - hs_estim

        # Masked L1 Training Loss (95% observed entries)
        train_diff = diff[mask_train]
        train_l1_loss = train_diff.abs().mean()

        # Masked L1 Validation Loss (5% observed entries)
        if n_val > 0:
            val_diff = diff[mask_val]
            val_l1_loss = val_diff.abs().mean()
        else:
            val_l1_loss = train_l1_loss

        # Log metrics
        train_loss_array[idx] = train_l1_loss.item()
        val_loss_array[idx] = val_l1_loss.item()

        tbar.set_description(
            f'Train-L1={train_l1_loss.item():.4e} | Val-L1={val_l1_loss.item():.4e}'
        )

        # Backpropagation strictly on observed training loss
        optimizer.zero_grad()
        train_l1_loss.backward()
        optimizer.step()

        # Checkpoint selection strictly based on self-supervised validation L1 loss
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

    # IMPORTANT: BatchNorm layers MUST use train mode for DIP stability
    im_net.train()
    spec_net.train()

    with torch.no_grad():
        U_img_best = im_net(im_inp)
        V_best = spec_net(spec_inp)
        U_best = U_img_best.reshape(-1, rank, H * W).permute(0, 2, 1)
        hs_estim_best = torch.bmm(U_best, V_best)

    best_hs_estim_np = hs_estim_best.squeeze().cpu().numpy()
    cube_estim = best_hs_estim_np.reshape(H, W, nwvl)

    data_range = cube.max() - cube.min()

    # --- Metrics Computation against Clean Ground Truth (Post-hoc ONLY) ---
    # 1. Full Tensor
    psnr_full = utils.psnr(cube, cube_estim)
    ssim_full = ssim_func(cube, cube_estim, channel_axis=-1, data_range=data_range)
    rmse_full = np.sqrt(((cube - cube_estim) ** 2).mean())

    # 2. Missing Entries Only
    psnr_missing, rmse_missing = compute_subset_metrics(cube, cube_estim, mask_missing_np)

    # 3. Observed Entries Only
    psnr_obs, rmse_obs = compute_subset_metrics(cube, cube_estim, ~mask_missing_np)

    # Baseline SVD comparison on observed target tensor
    cube_lr = spectral.lr_decompose(cube_target, SVD_BASELINE_RANK)
    svd_psnr = utils.psnr(cube, cube_lr)
    svd_ssim = ssim_func(cube, cube_lr, channel_axis=-1, data_range=data_range)

    print(f"Condition:                 {'CLEAN' if clean_mode else 'NOISY'}")
    print(f"Missing Ratio:             {missing_ratio:.2f}")
    print(f"Model Seed:                {seed}")
    print(f"Best Validation Epoch:     {best_epoch}")
    print(f"Best Validation L1 Loss:   {best_val_loss:.6e}")
    print("-" * 70)
    print("METRICS AGAINST CLEAN GROUND TRUTH:")
    print(f"  Full Tensor PSNR:        {psnr_full:.4f} dB")
    print(f"  Full Tensor SSIM:        {ssim_full:.6f}")
    print(f"  Full Tensor RMSE:        {rmse_full:.6f}")
    print(f"  Missing Entries PSNR:    {psnr_missing:.4f} dB")
    print(f"  Missing Entries RMSE:    {rmse_missing:.6f}")
    print(f"  Observed Entries PSNR:   {psnr_obs:.4f} dB")
    print(f"  Observed Entries RMSE:   {rmse_obs:.6f}")
    print("-" * 70)
    print(f"Runtime:                   {runtime:.2f} seconds")
    print(f"Peak GPU VRAM:             {peak_vram_mb:.2f} MB")
    print("=" * 70)

    # Save detailed text summary
    cond_str = "clean" if clean_mode else "noisy"
    summary_filename = os.path.join(
        output_dir, f"summary_p_{missing_ratio}_cond_{cond_str}_seed_{seed}.txt"
    )
    with open(summary_filename, 'w') as f:
        f.write("EXTENSION 3: MISSING-DATA / TENSOR COMPLETION RESULTS\n")
        f.write("=" * 70 + "\n")
        f.write(f"condition:               {cond_str}\n")
        f.write(f"missing_ratio:           {missing_ratio:.2f}\n")
        f.write(f"seed:                    {seed}\n")
        f.write(f"epochs:                  {epochs}\n")
        f.write(f"rank:                    {rank}\n")
        f.write(f"runtime_sec:             {runtime:.2f}\n")
        f.write(f"peak_vram_mb:           {peak_vram_mb:.2f}\n")
        f.write(f"best_val_epoch:         {best_epoch}\n")
        f.write(f"best_val_l1_loss:       {best_val_loss:.6e}\n")
        f.write("-" * 70 + "\n")
        f.write("ELEMENT COUNTS:\n")
        f.write(f"n_total:                 {n_total}\n")
        f.write(f"n_obs:                   {n_obs}\n")
        f.write(f"n_train:                 {n_train}\n")
        f.write(f"n_val:                   {n_val}\n")
        f.write(f"n_missing:               {n_missing}\n")
        f.write("-" * 70 + "\n")
        f.write("METRICS (VS CLEAN GROUND TRUTH):\n")
        f.write(f"psnr_full_db:            {psnr_full:.4f}\n")
        f.write(f"ssim_full:               {ssim_full:.6f}\n")
        f.write(f"rmse_full:               {rmse_full:.6f}\n")
        f.write(f"psnr_missing_db:         {psnr_missing:.4f}\n")
        f.write(f"rmse_missing:            {rmse_missing:.6f}\n")
        f.write(f"psnr_obs_db:             {psnr_obs:.4f}\n")
        f.write(f"rmse_obs:                {rmse_obs:.6f}\n")
        f.write(f"input_target_psnr:       {noisy_psnr:.4f}\n")
        f.write(f"svd_rank20_psnr:        {svd_psnr:.4f}\n")
        f.write(f"svd_rank20_ssim:        {svd_ssim:.6f}\n")

    print(f"Summary written to {summary_filename}")
