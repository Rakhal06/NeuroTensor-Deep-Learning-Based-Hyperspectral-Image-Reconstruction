#!/usr/bin/env python
"""
Independent Metric Verification Script for Figure 10 (Yale Face B)
==================================================================
Audits the Figure 10 Yale Face B 3-Way PARAFAC Denoising experiment:
1. Loads all five model checkpoints and validates saved state dicts and latent codes.
2. Reconstructs the 3D tensor from each model using saved checkpoint configurations.
3. Calculates PSNR, SSIM, RMSE, MAE, and Relative Frobenius Error independently.
4. Compares independently reproduced metrics against saved CSV and JSON.
5. Exports independent verification artifacts to results/hybrid/figure10_yale_denoising/independent_verification/.
"""

import os
import sys
import json
import csv
from typing import Dict, Any

import numpy as np
from scipy import io
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from deeptensor.models.factory import build_deeptensor_cp_model
from deeptensor.utils.metrics import calculate_psnr, calculate_ssim


def verify_figure10():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== Running Figure 10 Independent Verification on {device} ===")

    # 1. Load Ground Truth Yale Face B Data
    data_path = os.path.join(project_root, 'data', 'yale_b.mat')
    mat = io.loadmat(data_path)
    clean_tensor = mat['faces'].astype(np.float32) / 255.0
    H, W, T = clean_tensor.shape
    rank = 2000

    print(f"Ground Truth Shape: {clean_tensor.shape} (H={H}, W={W}, T={T})")
    print(f"Dynamic Range: Min={clean_tensor.min():.4f}, Max={clean_tensor.max():.4f}, PTP={np.ptp(clean_tensor):.4f}")

    results_dir = os.path.join(project_root, 'results', 'hybrid', 'figure10_yale_denoising')
    out_dir = os.path.join(results_dir, 'independent_verification')
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(results_dir, 'figure10_yale_b_comparison.json')
    with open(json_path, 'r') as f:
        saved_summary = json.load(f)

    type_map = {
        'baseline': 'skip1d',
        'hybrid': 'hybrid_attention',
        'cnn_control': 'cnn_control',
        'pointwise_control': 'hybrid_control',
        'transformer': 'spectral_transformer',
    }

    verification_records = {}
    print("\n--- Independent Checkpoint Evaluation ---")

    for v_name, w_type in type_map.items():
        ckpt_path = os.path.join(results_dir, f"{v_name}_checkpoint.pth")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        model = build_deeptensor_cp_model(
            rank=rank,
            lengths=(H, W, T),
            u_type='skip1d',
            v_type='skip1d',
            w_type=w_type,
            n_inputs=rank,
            learnable_core=True,
            normalize_core=True,
            d_model=128,
            n_heads=4,
            n_layers=4,
            d_ffn=512,
        ).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        # In single-instance Deep Image Prior (DIP) with batch size 1,
        # BatchNorm layers must remain in train mode to compute single-instance statistics
        # rather than uncalibrated exponential running averages.
        model.train()

        # Retrieve saved latents from checkpoint config
        cfg = ckpt.get('config', {})
        u_inp = cfg['u_inp'].to(device)
        v_inp = cfg['v_inp'].to(device)
        w_inp = cfg['w_inp'].to(device)

        with torch.no_grad():
            recon = model(u_inp, v_inp, w_inp).detach().cpu().numpy()

        # Compute independent metrics (Full 3D Tensor, data_range = 1.0)
        psnr_ind = float(calculate_psnr(clean_tensor, recon, data_range=1.0))
        ssim_ind = float(calculate_ssim(clean_tensor, recon, data_range=1.0, channel_axis=-1))
        diff = clean_tensor - recon
        rmse_ind = float(np.sqrt(np.mean(diff ** 2)))
        mae_ind = float(np.mean(np.abs(diff)))
        rel_frob_ind = float(np.linalg.norm(diff) / np.linalg.norm(clean_tensor))

        saved_m = saved_summary['models'][v_name]['metrics']
        delta_psnr = psnr_ind - saved_m['psnr']
        delta_ssim = ssim_ind - saved_m['ssim']

        match_status = "EXACT MATCH (< 1e-4 dB)" if abs(delta_psnr) < 1e-4 else f"DIFF ({delta_psnr:+.4f} dB)"

        verification_records[v_name] = {
            'checkpoint_epoch': ckpt['epoch'],
            'checkpoint_best_loss': ckpt['metrics'].get('best_loss'),
            'saved_report_psnr': saved_m['psnr'],
            'ind_verified_psnr': psnr_ind,
            'saved_report_ssim': saved_m['ssim'],
            'ind_verified_ssim': ssim_ind,
            'ind_verified_rmse': rmse_ind,
            'ind_verified_mae': mae_ind,
            'ind_verified_rel_frob': rel_frob_ind,
            'delta_psnr': delta_psnr,
            'delta_ssim': delta_ssim,
            'match_status': match_status,
        }

        print(f"Variant: {v_name:18s} | Saved: {saved_m['psnr']:.4f} dB | Verified: {psnr_ind:.4f} dB | {match_status}")

    # Export independent JSON
    ind_json_path = os.path.join(out_dir, 'independent_metrics.json')
    with open(ind_json_path, 'w') as f:
        json.dump(verification_records, f, indent=2)

    # Export independent CSV
    ind_csv_path = os.path.join(out_dir, 'independent_metrics.csv')
    with open(ind_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Model',
            'Report PSNR (dB)',
            'Verified PSNR (dB)',
            'Report SSIM',
            'Verified SSIM',
            'Verified RMSE',
            'Verified MAE',
            'Verified Rel Frob',
            'Status'
        ])
        for name, r in verification_records.items():
            writer.writerow([
                name,
                f"{r['saved_report_psnr']:.4f}",
                f"{r['ind_verified_psnr']:.4f}",
                f"{r['saved_report_ssim']:.4f}",
                f"{r['ind_verified_ssim']:.4f}",
                f"{r['ind_verified_rmse']:.6f}",
                f"{r['ind_verified_mae']:.6f}",
                f"{r['ind_verified_rel_frob']:.6f}",
                r['match_status']
            ])

    print(f"\n=== Independent Verification Complete ===")
    print(f"JSON Output: {ind_json_path}")
    print(f"CSV Output:  {ind_csv_path}")


if __name__ == '__main__':
    verify_figure10()
