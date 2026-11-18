#!/usr/bin/env python
"""
Hybrid CNN-Transformer Suite: Figure 12 3D PET/CT Radon Tomographic Reconstruction
==================================================================================
Comparative evaluation across five model variants:
  1. baseline:           Rank-1000 CP: U(128) 1D Skip + V(128) 1D Skip + W(56) 1D Skip
  2. hybrid:             Rank-1000 CP: U(128) 1D Skip + V(128) 1D Skip + W(56) Hybrid Attention
  3. cnn_control:        Parameter-matched pure 1D CNN with Conv1d residual block (W)
  4. pointwise_control:  Parameter-matched pointwise linear MLP mixing + FFN (W)
  5. transformer:        Rank-1000 CP: U(128) 1D Skip + V(128) 1D Skip + W(56) Pure Transformer

Dataset: 3D PET/CT Volume (data/pet.mat), shape: (128, 128, 56)
Forward Model: 40-angle Radon tomographic projection with Poisson (tau=100) + Gaussian (SNR=2.0) noise
Decomposition: 3-Way CP (PARAFAC) tensor factorization, Rank R=1000
Loss: Unsupervised L2 Radon measurement sinogram loss ||Y_meas - R(X_hat)||^2_2
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
    build_deeptensor_cp_model,
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
    meas_ten: torch.Tensor,
    angles: torch.Tensor,
    device: torch.device,
    out_dir: str,
    log_interval: int = 100,
    log_file: str = None,
) -> Dict[str, Any]:
    H, W, T = cube.shape
    rank = 1000
    n_inputs = rank
    lr = 1e-3
    reg_noise_std = 1.0 / 30.0

    criterion_l2 = losses.L2Norm()

    # Architecture for Mode 3 (axial slice dimension T=56)
    if variant == 'baseline':
        w_type = 'skip1d'
    elif variant == 'hybrid':
        w_type = 'hybrid_attention'
    elif variant == 'cnn_control':
        w_type = 'cnn_control'
    elif variant == 'pointwise_control':
        w_type = 'hybrid_control'
    elif variant == 'transformer':
        w_type = 'spectral_transformer'
    else:
        raise ValueError(f"Unknown variant: {variant}")

    set_seed(100)
    model = build_deeptensor_cp_model(
        rank=rank,
        lengths=(H, W, T),
        u_type='skip1d',
        v_type='skip1d',
        w_type=w_type,
        n_inputs=n_inputs,
        learnable_core=True,
        normalize_core=True,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ffn=512,
    ).to(device)

    set_seed(200)
    u_inp = utils.get_1d_posencode_inp(H, rank // 2, device=device)
    v_inp = utils.get_1d_posencode_inp(W, rank // 2, device=device)
    w_inp = utils.get_1d_posencode_inp(T, rank // 2, device=device)

    model.train()
    optimizer = torch.optim.Adam(
        list(model.parameters()) + [u_inp, v_inp, w_inp],
        lr=lr
    )

    u_inp_per = u_inp.detach().clone()
    v_inp_per = v_inp.detach().clone()
    w_inp_per = w_inp.detach().clone()

    best_loss = float('inf')
    best_epoch = 0
    best_recon = None
    best_state_dict = None
    loss_history = []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    start_time = time.time()
    for idx in range(epochs):
        u_pert = u_inp + u_inp_per.normal_() * reg_noise_std
        v_pert = v_inp + v_inp_per.normal_() * reg_noise_std
        w_pert = w_inp + w_inp_per.normal_() * reg_noise_std

        tensor_estim = model(u_pert, v_pert, w_pert)
        m_est = lin_inverse.radon(tensor_estim[None, ...].permute(0, 3, 1, 2), angles)

        loss = criterion_l2(meas_ten - m_est)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cur_loss = loss.item()
        loss_history.append(cur_loss)

        if cur_loss < best_loss:
            best_loss = cur_loss
            best_epoch = idx
            best_recon = tensor_estim.detach().cpu().numpy()
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
        'u_inp': u_inp.detach().cpu(),
        'v_inp': v_inp.detach().cpu(),
        'w_inp': w_inp.detach().cpu(),
        'best_recon': best_recon if best_recon is not None else tensor_estim.detach().cpu().numpy(),
        'epoch': best_epoch,
        'total_epochs': epochs,
        'best_loss': best_loss,
        'final_loss': cur_loss,
        'variant': variant,
        'total_params': total_params,
        'rank': rank,
        'nangles': len(angles),
        'H': H,
        'W': W,
        'T': T,
    }
    torch.save(checkpoint_dict, ckpt_path)

    return {
        'recon': best_recon if best_recon is not None else tensor_estim.detach().cpu().numpy(),
        'params': total_params,
        'runtime_sec': runtime,
        'peak_vram_mb': peak_vram,
        'loss_history': loss_history,
        'best_epoch': best_epoch,
        'best_loss': best_loss,
        'checkpoint_path': ckpt_path,
    }


def generate_slice_strip(
    gt: np.ndarray,
    recons: Dict[str, np.ndarray],
    slices: List[int],
    out_path: str,
):
    """
    Generate a publication-grade axial slice progression strip across models.
    Rows: Ground Truth and each reconstructed model variant.
    Columns: Selected axial slices along z-axis.
    """
    variants = list(recons.keys())
    rows = ['Ground Truth'] + variants
    n_rows = len(rows)
    n_cols = len(slices)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.8 * n_cols, 2.5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for col_idx, s in enumerate(slices):
        axes[0, col_idx].imshow(gt[:, :, s], cmap='gray', vmin=0, vmax=1)
        axes[0, col_idx].set_title(f"Slice {s}", fontsize=11, fontweight='bold')
        axes[0, col_idx].axis('off')

        for row_idx, v in enumerate(variants, start=1):
            recon_s = recons[v][:, :, s]
            axes[row_idx, col_idx].imshow(recon_s, cmap='gray', vmin=0, vmax=1)
            axes[row_idx, col_idx].axis('off')

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
        axes[0, i].set_title(f"{v}\nSlice {slice_idx}", fontsize=11, fontweight='bold')
        axes[0, i].axis('off')

        im = axes[1, i].imshow(err_slice, cmap='inferno', vmin=0, vmax=max_err)
        axes[1, i].set_title(f"|Error| (MAE: {mae:.4f})", fontsize=10)
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Hybrid CNN-Transformer Prototype for Figure 12 PET/CT")
    parser.add_argument('--variant', type=str, default='all',
                        choices=['baseline', 'hybrid', 'cnn_control', 'pointwise_control', 'transformer', 'all'])
    parser.add_argument('--smoke_test', action='store_true', help="Run 1 epoch for verification")
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    out_dir = os.path.join(project_root, 'results', 'hybrid', 'figure12_pet_ct')
    os.makedirs(out_dir, exist_ok=True)

    epochs = 1 if args.smoke_test else (args.epochs or 4000)
    device_str = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_str)

    print(f"=== Figure 12 3D PET/CT Radon CS Suite on {device} (Epochs: {epochs}) ===")

    # Load PET data
    data_path = os.path.join(project_root, 'data', 'pet.mat')
    if not os.path.exists(data_path):
        data_path = os.path.join(project_root, 'data', 'PET.mat')
    data = io.loadmat(data_path)
    cube = data.get('hypercube', data.get('PET', data.get('data', None)))
    if cube is None:
        for k in data:
            if not k.startswith('__') and isinstance(data[k], np.ndarray) and data[k].ndim == 3:
                cube = data[k]
                break

    cube = cube.astype(np.float32)
    cube = cube / cube.max()
    H, W, T = cube.shape

    set_seed(42)
    nangles = 40
    angles = torch.tensor(np.linspace(0, 180, nangles).astype(np.float32), device=device)

    # Physical forward model: Radon projection + Poisson/Gaussian detector noise
    cube_ten = torch.tensor(cube, dtype=torch.float32, device=device).permute(2, 0, 1)[None, ...]  # (1, 56, 128, 128)
    clean_measurements = lin_inverse.radon(cube_ten, angles).detach().cpu().numpy()  # (40, 56, 128)
    noisy_measurements = utils.measure(clean_measurements, 2.0, 100.0)
    meas_ten = torch.tensor(noisy_measurements, dtype=torch.float32, device=device)

    # Generate unregularized algorithmic inversion for degraded baseline observation in comparator
    print("--> Computing algorithmic initial inversion for baseline observation...")
    rec_alg = utils.get_inp([H, W, T], device=device)
    opt_alg = torch.optim.Adam([rec_alg], lr=0.1)
    crit_l2 = losses.L2Norm()
    for _ in range(30 if args.smoke_test else 100):
        m_est = lin_inverse.radon(rec_alg[None, ...].permute(0, 3, 1, 2), angles)
        l_alg = crit_l2(meas_ten - m_est)
        opt_alg.zero_grad()
        l_alg.backward()
        opt_alg.step()
    cube_obs = rec_alg.detach().cpu().numpy()

    comparator = SideBySideComparator("figure12_pet_ct", out_dir)
    variants = ['baseline', 'hybrid', 'cnn_control', 'pointwise_control', 'transformer'] if args.variant == 'all' else [args.variant]

    log_file = os.path.join(out_dir, f"training_{epochs}epochs.log")
    with open(log_file, 'w') as f_log:
        f_log.write(f"=== Figure 12 3D PET/CT Training Log ({epochs} epochs on {device}) ===\n")

    recons = {}
    results = {}
    log_interval = 1 if args.smoke_test else 200

    for v in variants:
        msg = f"\n--> Training variant: {v}..."
        print(msg)
        with open(log_file, 'a') as f_log:
            f_log.write(msg + '\n')
        res = run_single_variant(
            v, epochs, cube, meas_ten, angles, device, out_dir,
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

    slice_idx = T // 2  # Slice 28
    summary = comparator.generate_comparison(
        gt=cube,
        observation=cube_obs,
        slice_idx=slice_idx,
        save_figures=True,
    )

    diff_map_path = os.path.join(out_dir, 'figure12_pet_ct_diff_map.png')
    generate_diff_map(cube, recons, slice_idx, diff_map_path)

    slice_strip_path = os.path.join(out_dir, 'figure12_pet_ct_slice_strip.png')
    sample_slices = [7, 14, 21, 28, 35, 42, 49]
    generate_slice_strip(cube, recons, sample_slices, slice_strip_path)

    # Save experiment metadata
    metadata = {
        'experiment': 'Figure 12 3D PET/CT Radon CS',
        'dataset': 'data/pet.mat',
        'tensor_shape': [H, W, T],
        'tomography_angles': nangles,
        'rank': 1000,
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

    print(f"\n=== Figure 12 PET/CT Suite Complete ===")
    print(f"Summary saved to:  {summary['json_path']}")
    print(f"CSV saved to:      {summary['csv_path']}")
    print(f"Panel saved to:    {summary.get('figure_path', 'N/A')}")
    print(f"Diff map saved to: {diff_map_path}")
    print(f"Strip saved to:    {slice_strip_path}")


if __name__ == '__main__':
    main()
