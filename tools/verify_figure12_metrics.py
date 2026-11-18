#!/usr/bin/env python
"""
DeepTensor++ Independent Verification Tool: Figure 12 3D PET/CT Radon CS
========================================================================
Performs 100% independent audit of Figure 12 checkpoints directly from disk.
Recomputes PSNR, SSIM, RMSE, and MAE from saved checkpoints and validates
against reported run metrics.
"""

import sys
import os
import json
import csv
from typing import Dict, Any

import numpy as np
from scipy import io
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, os.path.join(project_root, 'src'))
    sys.path.insert(0, project_root)

from deeptensor.models.factory import build_deeptensor_cp_model, count_parameters
from deeptensor.utils.metrics import calculate_psnr, calculate_ssim


def main():
    print("=" * 78)
    print("DeepTensor++: Figure 12 PET/CT Independent Verification Audit")
    print("=" * 78)

    data_path = os.path.join(project_root, 'data', 'pet.mat')
    if not os.path.exists(data_path):
        data_path = os.path.join(project_root, 'data', 'PET.mat')
    data = io.loadmat(data_path)
    cube_raw = data.get('hypercube', data.get('PET', data.get('data', None)))
    if cube_raw is None:
        for k in data:
            if not k.startswith('__') and isinstance(data[k], np.ndarray) and data[k].ndim == 3:
                cube_raw = data[k]
                break

    cube_gt = cube_raw.astype(np.float32) / cube_raw.max()
    H, W, T = cube_gt.shape
    data_range = 1.0

    print(f"Dataset: pet.mat -> Shape: ({H}, {W}, {T}) | Dynamic Range: {data_range:.4f}")

    results_dir = os.path.join(project_root, 'results', 'hybrid', 'figure12_pet_ct')
    comp_json_path = os.path.join(results_dir, 'figure12_pet_ct_comparison.json')
    if os.path.exists(comp_json_path):
        with open(comp_json_path, 'r') as f:
            saved_comparison = json.load(f)
    else:
        saved_comparison = {}

    variants = ['baseline', 'hybrid', 'cnn_control', 'pointwise_control', 'transformer']
    w_types = {
        'baseline': 'skip1d',
        'hybrid': 'hybrid_attention',
        'cnn_control': 'cnn_control',
        'pointwise_control': 'hybrid_control',
        'transformer': 'spectral_transformer',
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Audit device: {device}\n")

    audit_records = {}

    for v in variants:
        ckpt_path = os.path.join(results_dir, f"{v}_checkpoint.pth")
        if not os.path.exists(ckpt_path):
            print(f"[-] Checkpoint missing for {v}: {ckpt_path}")
            continue

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        # 1. Evaluate from stored best_recon
        if 'best_recon' in ckpt:
            stored_recon = ckpt['best_recon']
            psnr_stored = float(calculate_psnr(cube_gt, stored_recon, data_range=data_range))
            ssim_stored = float(calculate_ssim(cube_gt, stored_recon, data_range=data_range, channel_axis=-1))
            rmse_stored = float(np.sqrt(np.mean((cube_gt - stored_recon) ** 2)))
            mae_stored = float(np.mean(np.abs(cube_gt - stored_recon)))
        else:
            psnr_stored = None
            ssim_stored = None
            rmse_stored = None
            mae_stored = None

        # 2. Evaluate from network weights + forward pass
        model = build_deeptensor_cp_model(
            rank=1000,
            lengths=(H, W, T),
            u_type='skip1d',
            v_type='skip1d',
            w_type=w_types[v],
            n_inputs=1000,
            learnable_core=True,
            normalize_core=True,
            d_model=128,
            n_heads=4,
            n_layers=4,
            d_ffn=512,
        ).to(device)

        model.load_state_dict(ckpt['model_state_dict'])
        model.train()  # Maintain instance-level BatchNorm statistics for single-instance DIP

        u_inp = ckpt['u_inp'].to(device)
        v_inp = ckpt['v_inp'].to(device)
        w_inp = ckpt['w_inp'].to(device)

        with torch.no_grad():
            tensor_estim = model(u_inp, v_inp, w_inp)
            forward_recon = tensor_estim.cpu().numpy()

        psnr_fwd = float(calculate_psnr(cube_gt, forward_recon, data_range=data_range))
        ssim_fwd = float(calculate_ssim(cube_gt, forward_recon, data_range=data_range, channel_axis=-1))
        rmse_fwd = float(np.sqrt(np.mean((cube_gt - forward_recon) ** 2)))
        mae_fwd = float(np.mean(np.abs(cube_gt - forward_recon)))

        reported_m = saved_comparison.get('models', {}).get(v, {}).get('metrics', {})
        reported_psnr = reported_m.get('psnr')
        reported_ssim = reported_m.get('ssim')

        audit_records[v] = {
            'variant': v,
            'total_params': ckpt.get('total_params', count_parameters(model)),
            'best_epoch': ckpt.get('epoch'),
            'best_loss': ckpt.get('best_loss'),
            'reported_psnr': reported_psnr,
            'stored_recon_psnr': psnr_stored,
            'forward_pass_psnr': psnr_fwd,
            'reported_ssim': reported_ssim,
            'stored_recon_ssim': ssim_stored,
            'forward_pass_ssim': ssim_fwd,
            'rmse': rmse_stored if rmse_stored is not None else rmse_fwd,
            'mae': mae_stored if mae_stored is not None else mae_fwd,
        }

        print(f"[{v.upper():18s}]")
        print(f"  Params:              {audit_records[v]['total_params']:,}")
        print(f"  Best Epoch:          {audit_records[v]['best_epoch']}")
        print(f"  Best Meas Loss:      {audit_records[v]['best_loss']:.5e}")
        print(f"  Reported PSNR:       {reported_psnr:.4f} dB" if reported_psnr is not None else "  Reported PSNR:       N/A")
        print(f"  Stored Recon PSNR:   {psnr_stored:.4f} dB" if psnr_stored is not None else "  Stored Recon PSNR:   N/A")
        print(f"  Forward Pass PSNR:   {psnr_fwd:.4f} dB")
        print(f"  Reported SSIM:       {reported_ssim:.4f}" if reported_ssim is not None else "  Reported SSIM:       N/A")
        print(f"  Stored Recon SSIM:   {ssim_stored:.4f}" if ssim_stored is not None else "  Stored Recon SSIM:   N/A")
        print(f"  Forward Pass SSIM:   {ssim_fwd:.4f}")
        print()

    # Save verification outputs
    audit_out_dir = os.path.join(results_dir, 'independent_verification')
    os.makedirs(audit_out_dir, exist_ok=True)

    json_path = os.path.join(audit_out_dir, 'independent_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(audit_records, f, indent=2)

    csv_path = os.path.join(audit_out_dir, 'independent_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Model', 'Parameters', 'Best Epoch', 'Best Meas Loss',
            'Reported PSNR (dB)', 'Verified Stored PSNR (dB)', 'Forward Pass PSNR (dB)',
            'Reported SSIM', 'Verified Stored SSIM', 'Forward Pass SSIM'
        ])
        for v, rec in audit_records.items():
            writer.writerow([
                v, rec['total_params'], rec['best_epoch'], f"{rec['best_loss']:.5e}",
                f"{rec['reported_psnr']:.4f}" if rec['reported_psnr'] is not None else 'N/A',
                f"{rec['stored_recon_psnr']:.4f}" if rec['stored_recon_psnr'] is not None else 'N/A',
                f"{rec['forward_pass_psnr']:.4f}",
                f"{rec['reported_ssim']:.4f}" if rec['reported_ssim'] is not None else 'N/A',
                f"{rec['stored_recon_ssim']:.4f}" if rec['stored_recon_ssim'] is not None else 'N/A',
                f"{rec['forward_pass_ssim']:.4f}",
            ])

    print("=" * 78)
    print(f"Audit Complete. Files written to:")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")
    print("=" * 78)


if __name__ == '__main__':
    main()

