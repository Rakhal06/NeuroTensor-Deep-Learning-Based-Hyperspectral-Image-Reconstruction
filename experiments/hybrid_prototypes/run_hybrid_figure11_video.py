#!/usr/bin/env python
"""
Hybrid CNN-Transformer Suite: Figure 11 Video Compressive Sensing (CACTI)
========================================================================
Comparative evaluation across five model variants:
  1. baseline:           Standard 2D Skip (U) + 1D Skip CNN (V)
  2. hybrid:             2D Skip (U) + 1D Skip CNN + Multi-Head Self-Attention + FFN (V)
  3. cnn_control:        Parameter-matched pure 1D CNN with Conv1d residual block (V)
  4. pointwise_control:  Parameter-matched pointwise linear MLP mixing + FFN (V)
  5. transformer:        2D Skip (U) + Pure Pre-LN Spectral Transformer with ALiBi (V)

Dataset: Cat Video (data/cat_video.mat), shape: (128, 128, 64)
Forward Model: 8-frame temporal multiplexing with random binary coded aperture (Hitomi et al., ICCV 2011)
Decomposition: Bilinear matrix decomposition U(H, W) x V(T), Rank R=32
Loss: Unsupervised L2 measurement loss ||Y_meas - Phi(X_hat)||^2_2
"""

import sys
import os
import argparse
import time
import copy
import random
import json
from typing import Dict, Any, List

import numpy as np
from scipy import io
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, os.path.join(project_root, 'src'))
    sys.path.insert(0, project_root)

from deeptensor import utils, losses
from deeptensor.physics import operators as lin_inverse
from deeptensor.models.factory import (
    build_deeptensor_model,
    count_parameters,
)
from deeptensor.utils.comparison import SideBySideComparator


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_single_variant(
    variant: str,
    epochs: int,
    cube: np.ndarray,
    masks_ten: torch.Tensor,
    measurements_ten: torch.Tensor,
    nframes: int,
    device: torch.device,
    out_dir: str,
    log_interval: int = 100,
    log_file: str = None,
) -> Dict[str, Any]:
    H, W, totalframes = cube.shape
    rank = 32
    n_inputs = rank
    lr = 1e-3
    reg_noise_std = 1.0 / 30.0

    criterion = losses.L2Norm()

    if variant == 'baseline':
        spectral_type = 'skip1d'
    elif variant == 'hybrid':
        spectral_type = 'hybrid_attention'
    elif variant == 'cnn_control':
        spectral_type = 'cnn_control'
    elif variant == 'pointwise_control':
        spectral_type = 'hybrid_control'
    elif variant == 'transformer':
        spectral_type = 'spectral_transformer'
    else:
        raise ValueError(f"Unknown variant: {variant}")

    set_seed(100)
    model = build_deeptensor_model(
        rank=rank,
        seq_len=totalframes,
        spatial_type='skip',
        spectral_type=spectral_type,
        n_inputs=n_inputs,
        skip_n33d=64,
        skip_n33u=64,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ffn=512,
    ).to(device)

    set_seed(200)
    im_inp = utils.get_inp([1, n_inputs, H, W], device=device)
    spec_inp = utils.get_inp([1, n_inputs, totalframes], device=device)

    model.train()
    optimizer = torch.optim.Adam(
        list(model.parameters()) + [im_inp, spec_inp],
        lr=lr
    )

    im_inp_per = im_inp.detach().clone()
    spec_inp_per = spec_inp.detach().clone()

    best_loss = float('inf')
    best_epoch = 0
    best_recon = None
    best_state_dict = None
    loss_history = []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    start_time = time.time()
    for idx in range(epochs):
        im_pert = im_inp + im_inp_per.normal_() * reg_noise_std
        spec_pert = spec_inp + spec_inp_per.normal_() * reg_noise_std

        mat_estim = model(im_pert, spec_pert)  # (1, H*W, totalframes)
        cube_estim = mat_estim.reshape(1, H, W, totalframes).permute(0, 3, 1, 2)  # (1, totalframes, H, W)
        meas_estim = lin_inverse.video2codedvideo(cube_estim, masks_ten, nframes)  # (1, 8, H, W)

        loss = criterion(measurements_ten - meas_estim)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cur_loss = loss.item()
        loss_history.append(cur_loss)

        if cur_loss < best_loss:
            best_loss = cur_loss
            best_epoch = idx
            best_recon = cube_estim.permute(0, 2, 3, 1)[0].detach().cpu().numpy()
            best_state_dict = copy.deepcopy(model.state_dict())

        if (idx + 1) % log_interval == 0 or idx == epochs - 1:
            elapsed = time.time() - start_time
            msg = f"  [{variant}] Epoch {idx + 1:5d}/{epochs} | Meas L2 Loss: {cur_loss:.5e} | Best: {best_loss:.5e} (ep {best_epoch + 1}) | Elapsed: {elapsed:.1f}s"
            print(msg)
            if log_file:
                with open(log_file, 'a') as f_log:
                    f_log.write(msg + '\n')

    runtime = time.time() - start_time
    peak_vram = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if torch.cuda.is_available() else 0.0
    total_params = count_parameters(model)

    ckpt_path = os.path.join(out_dir, f"{variant}_checkpoint.pth")
    checkpoint_dict = {
        'model_state_dict': best_state_dict if best_state_dict is not None else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'im_inp': im_inp.detach().cpu(),
        'spec_inp': spec_inp.detach().cpu(),
        'best_recon': best_recon if best_recon is not None else cube_estim.permute(0, 2, 3, 1)[0].detach().cpu().numpy(),
        'epoch': best_epoch,
        'total_epochs': epochs,
        'best_loss': best_loss,
        'final_loss': cur_loss,
        'variant': variant,
        'total_params': total_params,
        'rank': rank,
        'nframes': nframes,
        'H': H,
        'W': W,
        'totalframes': totalframes,
    }
    torch.save(checkpoint_dict, ckpt_path)

    return {
        'recon': best_recon if best_recon is not None else cube_estim.permute(0, 2, 3, 1)[0].detach().cpu().numpy(),
        'params': total_params,
        'runtime_sec': runtime,
        'peak_vram_mb': peak_vram,
        'loss_history': loss_history,
        'best_epoch': best_epoch,
        'best_loss': best_loss,
        'checkpoint_path': ckpt_path,
    }


def generate_temporal_strip(
    gt: np.ndarray,
    recons: Dict[str, np.ndarray],
    frames: List[int],
    out_path: str,
):
    """
    Generate a publication-grade temporal frame strip comparison across models.
    Rows: Ground Truth and each reconstructed model variant.
    Columns: Selected temporal frames across time.
    """
    variants = list(recons.keys())
    rows = ['Ground Truth'] + variants
    n_rows = len(rows)
    n_cols = len(frames)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.8 * n_cols, 2.5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for col_idx, f in enumerate(frames):
        # Row 0: Ground Truth
        axes[0, col_idx].imshow(gt[:, :, f], cmap='gray', vmin=0, vmax=1)
        axes[0, col_idx].set_title(f"Frame {f}", fontsize=11, fontweight='bold')
        axes[0, col_idx].axis('off')

        # Model rows
        for row_idx, v in enumerate(variants, start=1):
            recon_f = recons[v][:, :, f]
            axes[row_idx, col_idx].imshow(recon_f, cmap='gray', vmin=0, vmax=1)
            axes[row_idx, col_idx].axis('off')

    # Label rows on left-most column
    axes[0, 0].text(-0.15, 0.5, "Ground Truth", transform=axes[0, 0].transAxes,
                    fontsize=11, fontweight='bold', va='center', ha='right', rotation=90)
    for row_idx, v in enumerate(variants, start=1):
        axes[row_idx, 0].text(-0.15, 0.5, v, transform=axes[row_idx, 0].transAxes,
                              fontsize=11, fontweight='bold', va='center', ha='right', rotation=90)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()


def generate_diff_map(
    gt: np.ndarray,
    recons: Dict[str, np.ndarray],
    slice_idx: int,
    out_path: str,
):
    """Generate high-resolution difference maps (|Recon - GT|) across all models."""
    gt_slice = gt[:, :, slice_idx]
    variants = list(recons.keys())
    n = len(variants)

    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)

    max_err = 0.0
    for v in variants:
        err = np.abs(recons[v][:, :, slice_idx] - gt_slice)
        max_err = max(max_err, float(np.percentile(err, 99.5)))
    max_err = max(max_err, 1e-4)

    for i, v in enumerate(variants):
        recon_slice = recons[v][:, :, slice_idx]
        err_slice = np.abs(recon_slice - gt_slice)
        mae = float(np.mean(err_slice))

        axes[0, i].imshow(recon_slice, cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f"{v}\nFrame {slice_idx}", fontsize=11, fontweight='bold')
        axes[0, i].axis('off')

        im = axes[1, i].imshow(err_slice, cmap='inferno', vmin=0, vmax=max_err)
        axes[1, i].set_title(f"|Error| (MAE: {mae:.4f})", fontsize=10)
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Hybrid CNN-Transformer Prototype for Figure 11 Video CS")
    parser.add_argument('--variant', type=str, default='all',
                        choices=['baseline', 'hybrid', 'cnn_control', 'pointwise_control', 'transformer', 'all'])
    parser.add_argument('--smoke_test', action='store_true', help="Run 1 epoch for verification")
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    out_dir = os.path.join(project_root, 'results', 'hybrid', 'figure11_video_cs')
    os.makedirs(out_dir, exist_ok=True)

    epochs = 1 if args.smoke_test else (args.epochs or 3000)
    device_str = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_str)

    print(f"=== Figure 11 Video Compressive Sensing Suite on {device} (Epochs: {epochs}) ===")

    # Load video data
    data_path = os.path.join(project_root, 'data', 'cat_video.mat')
    data = io.loadmat(data_path)
    cube = data['hypercube'].astype(np.float32)
    cube = utils.resize(cube / cube.max(), 1)
    H, W, totalframes = cube.shape

    set_seed(42)
    nframes = 8
    cube_noisy = utils.measure(cube, 2, 100)
    masks = lin_inverse.get_video_coding_frames(cube.shape, nframes)

    measurements = np.zeros((H, W, totalframes // nframes), dtype=np.float32)
    for idx in range(0, totalframes, nframes):
        cube_chunk = cube_noisy[..., idx:idx + nframes]
        masks_chunk = masks[..., idx:idx + nframes]
        measurements[..., idx // nframes] = (cube_chunk * masks_chunk).sum(2)

    measurements_ten = torch.tensor(measurements, device=device)[None, ...].permute(0, 3, 1, 2)
    masks_ten = torch.tensor(masks, device=device)[None, ...].permute(0, 3, 1, 2)

    comparator = SideBySideComparator("figure11_video_cs", out_dir)
    variants = ['baseline', 'hybrid', 'cnn_control', 'pointwise_control', 'transformer'] if args.variant == 'all' else [args.variant]

    log_file = os.path.join(out_dir, f"training_{epochs}epochs.log")
    with open(log_file, 'w') as f_log:
        f_log.write(f"=== Figure 11 Video CS Training Log ({epochs} epochs on {device}) ===\n")

    recons = {}
    results = {}
    log_interval = 1 if args.smoke_test else 200

    for v in variants:
        msg = f"\n--> Training variant: {v}..."
        print(msg)
        with open(log_file, 'a') as f_log:
            f_log.write(msg + '\n')
        res = run_single_variant(
            v, epochs, cube, masks_ten, measurements_ten, nframes, device, out_dir,
            log_interval=log_interval, log_file=log_file
        )
        recons[v] = res['recon']
        results[v] = res
        comparator.add_model_result(
            name=v,
            recon=res['recon'],
            params=res['params'],
            runtime_sec=res['runtime_sec'],
            peak_vram_mb=res['peak_vram_mb'],
            loss_history=res['loss_history'],
        )
        print(f"    Completed {v}: params={res['params']:,}, time={res['runtime_sec']:.2f}s, VRAM={res['peak_vram_mb']:.1f}MB")

    slice_idx = totalframes // 2  # Frame 32
    summary = comparator.generate_comparison(
        gt=cube,
        observation=cube_noisy,
        slice_idx=slice_idx,
        save_figures=True,
    )

    diff_map_path = os.path.join(out_dir, 'figure11_video_cs_diff_map.png')
    generate_diff_map(cube, recons, slice_idx, diff_map_path)

    temporal_strip_path = os.path.join(out_dir, 'figure11_video_cs_temporal_strip.png')
    sample_frames = [8, 16, 24, 32, 40, 48, 56]
    generate_temporal_strip(cube, recons, sample_frames, temporal_strip_path)

    # Save experiment metadata
    metadata = {
        'experiment': 'Figure 11 Video Compressive Sensing (CACTI)',
        'dataset': 'data/cat_video.mat',
        'tensor_shape': [H, W, totalframes],
        'multiplexed_frames': nframes,
        'coded_snapshots': totalframes // nframes,
        'rank': 32,
        'epochs': epochs,
        'device': str(device),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'torch_version': torch.__version__,
        'loss_criterion': 'losses.L2Norm',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'variants': variants,
        'parameter_counts': {v: results[v]['params'] for v in variants},
        'runtimes_sec': {v: results[v]['runtime_sec'] for v in variants},
        'peak_vram_mb': {v: results[v]['peak_vram_mb'] for v in variants},
    }
    with open(os.path.join(out_dir, 'experiment_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n=== Figure 11 Video CS Suite Complete ===")
    print(f"Summary saved to:  {summary['json_path']}")
    print(f"CSV saved to:      {summary['csv_path']}")
    print(f"Panel saved to:    {summary.get('figure_path', 'N/A')}")
    print(f"Diff map saved to: {diff_map_path}")
    print(f"Strip saved to:    {temporal_strip_path}")


if __name__ == '__main__':
    main()
