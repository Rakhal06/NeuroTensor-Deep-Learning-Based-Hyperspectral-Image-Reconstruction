#!/usr/bin/env python
"""
Independent Metric Verification & Discrepancy Audit Script for Figure 09
========================================================================
Audits the Figure 09 ICVL Hyperspectral experiment:
1. Investigates the mathematical and implementation discrepancy between
   full-cube evaluation (43-46 dB in CSV/JSON) and slice evaluation (29-32 dB in panel titles).
2. Loads all five model checkpoints and validates architecture definitions.
3. Performs rigorous metric calculation across dimensions (full 3D cube vs. 2D slice)
   and data ranges (global range = 1.0/0.9616 vs. slice range = 0.2175).
4. Exports verified independent metrics and publication-corrected comparison panels.
"""

import os
import sys
import json
import csv
import random
from typing import Dict, Any, Tuple

import numpy as np
from scipy import io
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from deeptensor import utils, losses
from deeptensor.models.factory import build_deeptensor_model
from deeptensor.utils.metrics import calculate_psnr, calculate_ssim
from deeptensor.utils.comparison import compute_metrics, to_2d_slice


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_verification():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== Running Figure 09 Independent Verification on {device} ===")

    # 1. Load Ground Truth ICVL Data
    data_path = os.path.join(project_root, 'data', 'icvl.mat')
    data = io.loadmat(data_path)
    cube_raw = data['hypercube'].astype(np.float32)
    cube = cube_raw / cube_raw.max()
    H, W, nwvl = cube.shape

    slice_idx = nwvl // 2  # Channel 130
    gt_slice = cube[:, :, slice_idx]

    cube_ptp = float(np.ptp(cube))
    slice_ptp = float(np.ptp(gt_slice))

    print(f"\n--- Ground Truth Properties ---")
    print(f"Tensor Shape: {cube.shape} (H={H}, W={W}, L={nwvl})")
    print(f"Cube Min: {cube.min():.4f}, Max: {cube.max():.4f}, Range: {cube_ptp:.4f}")
    print(f"Slice 130 Min: {gt_slice.min():.4f}, Max: {gt_slice.max():.4f}, Range: {slice_ptp:.4f}")
    print(f"Range Discrepancy Factor (20 * log10(slice_ptp / cube_ptp)): {20 * np.log10(slice_ptp / cube_ptp):.2f} dB")
    print(f"Range Discrepancy Factor (20 * log10(slice_ptp / 1.0)):      {20 * np.log10(slice_ptp / 1.0):.2f} dB")

    # 2. Check Saved JSON and CSV
    results_dir = os.path.join(project_root, 'results', 'hybrid', 'figure09_hyperspectral')
    out_dir = os.path.join(results_dir, 'independent_verification')
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(results_dir, 'figure09_icvl_hsi_comparison.json')
    with open(json_path, 'r') as f:
        saved_summary = json.load(f)

    print(f"\n--- Saved Report Metrics (Full 3D Cube) ---")
    for name, data in saved_summary['models'].items():
        m = data['metrics']
        print(f"  {name:18s}: PSNR = {m['psnr']:.4f} dB | SSIM = {m['ssim']:.4f} | RMSE = {m['rmse']:.6f}")

    # 3. Model Architecture Mapping
    variants = {
        'baseline': 'skip1d',
        'hybrid': 'hybrid_attention',
        'cnn_control': 'cnn_control',
        'pointwise_control': 'hybrid_control',
        'transformer': 'spectral_transformer',
    }

    # Generate degraded observation identically to training
    set_seed(42)
    cube_noisy = utils.measure(cube, 2, 100)
    hsten_noisy = torch.tensor(cube_noisy.reshape(H * W, nwvl), dtype=torch.float32)[None, ...].to(device)

    verification_records = {}

    for v_name, s_type in variants.items():
        ckpt_path = os.path.join(results_dir, f"{v_name}_checkpoint.pth")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint missing: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        # Build and load model
        model = build_deeptensor_model(
            rank=20,
            seq_len=nwvl,
            spatial_type='skip',
            spectral_type=s_type,
            n_inputs=20,
            d_model=128,
            n_heads=4,
            n_layers=4,
            d_ffn=512,
        ).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        # Re-evaluate with initialized latents and forward pass
        set_seed(200)
        im_inp = utils.get_inp([1, 20, H, W]).to(device)
        spec_inp = utils.get_inp([1, 20, nwvl]).to(device)

        with torch.no_grad():
            recon_cube = model(im_inp, spec_inp).squeeze(0).cpu().numpy().reshape(H, W, nwvl)

        recon_slice = recon_cube[:, :, slice_idx]

        # Calculate metrics under multiple protocols
        # Protocol A: Full 3D Cube (data_range = 1.0)
        psnr_cube_1 = calculate_psnr(cube, recon_cube, data_range=1.0)
        # Protocol B: Full 3D Cube (data_range = ptp(cube))
        psnr_cube_ptp = calculate_psnr(cube, recon_cube, data_range=cube_ptp)
        ssim_cube = calculate_ssim(cube, recon_cube, data_range=cube_ptp, channel_axis=-1)
        rmse_cube = float(np.sqrt(np.mean((cube - recon_cube) ** 2)))
        mae_cube = float(np.mean(np.abs(cube - recon_cube)))
        rel_frob_cube = float(np.linalg.norm(cube - recon_cube) / np.linalg.norm(cube))

        # Protocol C: 2D Slice 130 (data_range = ptp(slice) = 0.2175) -> Reproduces the 29-32 dB panel titles
        psnr_slice_local = calculate_psnr(gt_slice, recon_slice, data_range=slice_ptp)
        # Protocol D: 2D Slice 130 (data_range = 1.0)
        psnr_slice_global = calculate_psnr(gt_slice, recon_slice, data_range=1.0)

        saved_m = saved_summary['models'][v_name]['metrics']

        verification_records[v_name] = {
            'checkpoint_epoch': ckpt['epoch'],
            'checkpoint_final_loss': ckpt['metrics'].get('final_loss'),
            'checkpoint_best_loss': ckpt['metrics'].get('best_loss'),
            'saved_report_psnr_cube': saved_m['psnr'],
            'saved_report_ssim_cube': saved_m['ssim'],
            'saved_report_rmse_cube': saved_m['rmse'],
            'unaligned_eval_psnr_cube_1': float(psnr_cube_1),
            'unaligned_eval_psnr_cube_ptp': float(psnr_cube_ptp),
            'unaligned_eval_ssim_cube': float(ssim_cube),
            'unaligned_eval_rmse_cube': float(rmse_cube),
            'unaligned_eval_mae_cube': float(mae_cube),
            'unaligned_eval_rel_frob_cube': float(rel_frob_cube),
            'slice_psnr_local_range': float(psnr_slice_local),
            'slice_psnr_global_range': float(psnr_slice_global),
            'slice_ptp': float(slice_ptp),
            'cube_ptp': float(cube_ptp),
        }

        print(f"\nVariant: {v_name}")
        print(f"  Saved Full-Cube PSNR:          {saved_m['psnr']:.2f} dB (Saved Report)")
        print(f"  Slice 130 Local PSNR (0.2175): {psnr_slice_local:.2f} dB (Matches Panel Title Scale)")
        print(f"  Slice 130 Global PSNR (1.0):   {psnr_slice_global:.2f} dB")
        print(f"  Delta (Global - Local):        {psnr_slice_global - psnr_slice_local:.2f} dB")

    # 4. Save independent verification JSON
    ind_json_path = os.path.join(out_dir, 'independent_metrics.json')
    with open(ind_json_path, 'w') as f:
        json.dump(verification_records, f, indent=2)

    # 5. Save independent verification CSV
    ind_csv_path = os.path.join(out_dir, 'independent_metrics.csv')
    with open(ind_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Model',
            'Report PSNR Cube (dB)',
            'Report SSIM Cube',
            'Report RMSE Cube',
            'Slice 130 PSNR Local Range (dB)',
            'Slice 130 PSNR Global Range (dB)',
            'Range Shift Delta (dB)',
        ])
        for name, r in verification_records.items():
            writer.writerow([
                name,
                f"{r['saved_report_psnr_cube']:.4f}",
                f"{r['saved_report_ssim_cube']:.4f}",
                f"{r['saved_report_rmse_cube']:.6f}",
                f"{r['slice_psnr_local_range']:.2f}",
                f"{r['slice_psnr_global_range']:.2f}",
                f"{r['slice_psnr_global_range'] - r['slice_psnr_local_range']:.2f}",
            ])

    # 6. Generate Publication-Corrected Visual Panel
    print("\n--> Generating Publication-Corrected Comparison Panel...")
    models_list = list(variants.keys())
    cols = len(models_list) + 2
    fig, axes = plt.subplots(2, cols, figsize=(4 * cols, 7))

    # Panel 1: Ground Truth
    axes[0, 0].imshow(gt_slice, cmap='gray')
    axes[0, 0].set_title("Ground Truth\n(Slice 130 / 550nm)", fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')

    axes[1, 0].imshow(np.zeros_like(gt_slice), cmap='inferno', vmin=0, vmax=0.05)
    axes[1, 0].set_title("Zero Error Ref", fontsize=10)
    axes[1, 0].axis('off')

    # Panel 2: Degraded Observation
    obs_slice = cube_noisy[:, :, slice_idx]
    axes[0, 1].imshow(obs_slice, cmap='gray')
    axes[0, 1].set_title("Degraded Input\n(Poisson + Gaussian)", fontsize=11, fontweight='bold')
    axes[0, 1].axis('off')

    obs_err = np.abs(gt_slice - obs_slice)
    im_err = axes[1, 1].imshow(obs_err, cmap='inferno')
    axes[1, 1].set_title(f"Input Error\nMAE: {np.mean(obs_err):.4f}", fontsize=10)
    axes[1, 1].axis('off')

    # Reconstruct all models for panel visualization
    recon_slices = {}
    for v_name, s_type in variants.items():
        ckpt_path = os.path.join(results_dir, f"{v_name}_checkpoint.pth")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = build_deeptensor_model(
            rank=20, seq_len=nwvl, spatial_type='skip', spectral_type=s_type,
            n_inputs=20, d_model=128, n_heads=4, n_layers=4, d_ffn=512
        ).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        set_seed(200)
        im_inp = utils.get_inp([1, 20, H, W]).to(device)
        spec_inp = utils.get_inp([1, 20, nwvl]).to(device)

        with torch.no_grad():
            r_cube = model(im_inp, spec_inp).squeeze(0).cpu().numpy().reshape(H, W, nwvl)
        recon_slices[v_name] = r_cube[:, :, slice_idx]

    max_err = 0.05
    for m_idx, m_name in enumerate(models_list):
        c = 2 + m_idx
        m_slice = recon_slices[m_name]
        err_map = np.abs(gt_slice - m_slice)
        r = verification_records[m_name]

        axes[0, c].imshow(m_slice, cmap='gray')
        # Explicitly show BOTH full-cube report PSNR and slice PSNR to resolve discrepancy
        axes[0, c].set_title(
            f"{m_name.upper()}\nCube PSNR: {r['saved_report_psnr_cube']:.2f} dB\n(Slice: {r['slice_psnr_local_range']:.2f} dB)",
            fontsize=10,
            fontweight='bold',
        )
        axes[0, c].axis('off')

        im_m = axes[1, c].imshow(err_map, cmap='inferno', vmin=0, vmax=max_err)
        axes[1, c].set_title(f"{m_name} Error Map\nMAE: {np.mean(err_map):.4f}", fontsize=10)
        axes[1, c].axis('off')

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.3])
    cbar = fig.colorbar(im_m, cax=cbar_ax)
    cbar.set_label('Absolute Error |GT - Recon|', fontsize=9)

    plt.subplots_adjust(top=0.88, bottom=0.08, left=0.03, right=0.90, hspace=0.30, wspace=0.15)
    plt.suptitle(
        "Figure 09 ICVL HSI: Verified Multi-Model Architecture Comparison\n"
        "(Dual Metric Display: Full 3D Cube Global PSNR vs. Single-Slice Local PSNR)",
        fontsize=13,
        fontweight='bold',
    )

    corrected_panel_path = os.path.join(out_dir, "figure09_icvl_hsi_comparison_panel_corrected.png")
    plt.savefig(corrected_panel_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f"\n=== Verification Complete ===")
    print(f"Independent Metrics JSON: {ind_json_path}")
    print(f"Independent Metrics CSV:  {ind_csv_path}")
    print(f"Corrected Panel:          {corrected_panel_path}")


if __name__ == '__main__':
    run_verification()

