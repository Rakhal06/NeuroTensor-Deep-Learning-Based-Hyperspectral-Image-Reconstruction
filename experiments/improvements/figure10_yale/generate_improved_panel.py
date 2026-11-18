#!/usr/bin/env python
'''
Generates the comprehensive Figure 10 visual comparison panel and summary tables
combining both 30% and 60% Salt & Pepper noise improved runs.
'''

import os
import sys
import json
import numpy as np
import scipy.io as io
from skimage.metrics import structural_similarity as ssim_func
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from deeptensor import utils

def main():
    improved_dir = os.path.join(project_root, 'results', 'improved', 'figure10_yale_denoising')
    orig_dir = os.path.join(project_root, 'results', 'baseline', 'figure10_yale_denoising')
    images_dir = os.path.join(improved_dir, 'images')
    plots_dir = os.path.join(improved_dir, 'plots')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Load Clean Dataset
    data = io.loadmat(os.path.join(project_root, 'data', 'yale_b.mat'))
    tensor_clean = data['faces'].astype(np.float32) / 255.0
    H, W, T = tensor_clean.shape

    # Load Metrics
    with open(os.path.join(improved_dir, 'metrics_noise_30.json')) as f:
        m30 = json.load(f)
    with open(os.path.join(improved_dir, 'metrics_noise_60.json')) as f:
        m60 = json.load(f)

    # Load Reconstructions
    dt_30 = np.load(os.path.join(improved_dir, 'recon_noise_30_deeptensor.npy'))
    dt_60 = np.load(os.path.join(improved_dir, 'recon_noise_60_deeptensor.npy'))
    svd_30 = np.load(os.path.join(improved_dir, 'recon_noise_30_svd.npy'))
    svd_60 = np.load(os.path.join(improved_dir, 'recon_noise_60_svd.npy'))
    noisy_30 = np.load(os.path.join(improved_dir, 'noisy_noise_30.npy'))
    noisy_60 = np.load(os.path.join(improved_dir, 'noisy_noise_60.npy'))

    # Load Old Baseline for comparison
    with open(os.path.join(orig_dir, 'metrics.json')) as f:
        orig_metrics = json.load(f)

    # Combined Figure 10 Visual Comparison Panel
    fig, axes = plt.subplots(3, 4, figsize=(15, 11))
    sample_slice = 0 # Representative face slice

    # Row 0: Ground Truth & Corrupted Inputs
    axes[0, 0].imshow(tensor_clean[..., sample_slice], cmap='gray')
    axes[0, 0].set_title("Ground Truth\nExtended Yale-B ($192 \\times 168$)", fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(noisy_30[..., sample_slice], cmap='gray')
    axes[0, 1].set_title(f"Input: 30% Salt & Pepper\nPSNR: {m30['input_noisy_psnr']:.2f} dB (Fixed)", fontsize=11)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(noisy_60[..., sample_slice], cmap='gray')
    axes[0, 2].set_title(f"Input: 60% Salt & Pepper\nPSNR: {m60['input_noisy_psnr']:.2f} dB (Fixed)", fontsize=11)
    axes[0, 2].axis('off')

    axes[0, 3].text(0.1, 0.5, "Input Corrupted\nObservations\n(Exact Fixed %)", fontsize=12, va='center', fontweight='bold')
    axes[0, 3].axis('off')

    # Row 1: Classical SVD Baselines
    axes[1, 0].axis('off')
    axes[1, 1].imshow(svd_30[..., sample_slice], cmap='gray')
    axes[1, 1].set_title(f"Truncated SVD (30%)\nPSNR: {m30['svd_psnr']:.2f} dB | SSIM: {m30['svd_ssim']:.4f}", fontsize=11)
    axes[1, 1].axis('off')

    axes[1, 2].imshow(svd_60[..., sample_slice], cmap='gray')
    axes[1, 2].set_title(f"Truncated SVD (60%)\nPSNR: {m60['svd_psnr']:.2f} dB | SSIM: {m60['svd_ssim']:.4f}", fontsize=11)
    axes[1, 2].axis('off')

    axes[1, 3].text(0.1, 0.5, "Classical Baseline\n(Truncated SVD,\nMatricized)", fontsize=12, va='center', fontweight='bold')
    axes[1, 3].axis('off')

    # Row 2: DeepTensor Improved Reconstructions
    axes[2, 0].axis('off')
    axes[2, 1].imshow(dt_30[..., sample_slice], cmap='gray')
    axes[2, 1].set_title(f"DeepTensor Improved (30%)\nPSNR: {m30['deeptensor_psnr']:.2f} dB | SSIM: {m30['deeptensor_ssim']:.4f}\n(Paper: 28.9 dB | Old: {orig_metrics['noise_30']['deeptensor_psnr']:.2f} dB)",
                         fontsize=11, fontweight='bold', color='navy')
    axes[2, 1].axis('off')

    axes[2, 2].imshow(dt_60[..., sample_slice], cmap='gray')
    axes[2, 2].set_title(f"DeepTensor Improved (60%)\nPSNR: {m60['deeptensor_psnr']:.2f} dB | SSIM: {m60['deeptensor_ssim']:.4f}\n(Paper: 29.7 dB | Old: {orig_metrics['noise_60']['deeptensor_psnr']:.2f} dB)",
                         fontsize=11, fontweight='bold', color='navy')
    axes[2, 2].axis('off')

    axes[2, 3].text(0.1, 0.5, f"DeepTensor Rerun\n(Rank-2000 PARAFAC,\n128 Channels, 5k Ep)",
                    fontsize=12, va='center', fontweight='bold', color='navy')
    axes[2, 3].axis('off')

    plt.suptitle("Figure 10: 3-Way PARAFAC Tensor Denoising on Extended Yale Face Database B\nImproved DeepTensor Rerun under 30% and 60% Salt & Pepper Noise (Paper Protocol)", fontsize=13, y=0.98)
    plt.tight_layout()
    panel_path = os.path.join(images_dir, 'figure10_yale_tensor_denoising_improved_panel.png')
    plt.savefig(panel_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved combined Figure 10 panel to: {panel_path}", flush=True)

    # Combined Convergence Plot
    plt.figure(figsize=(13, 5))
    plt.subplot(1, 2, 1)
    with open(os.path.join(improved_dir, 'progression_noise_30.csv')) as f:
        lines_30 = f.readlines()[1:]
    with open(os.path.join(improved_dir, 'progression_noise_60.csv')) as f:
        lines_60 = f.readlines()[1:]

    ep_30 = [int(l.split(',')[0]) for l in lines_30]
    loss_30 = [float(l.split(',')[1]) for l in lines_30]
    psnr_30 = [float(l.split(',')[2]) for l in lines_30]

    ep_60 = [int(l.split(',')[0]) for l in lines_60]
    loss_60 = [float(l.split(',')[1]) for l in lines_60]
    psnr_60 = [float(l.split(',')[2]) for l in lines_60]

    plt.plot(ep_30, loss_30, color='navy', linewidth=1.8, label='30% Noise L1 Loss')
    plt.plot(ep_60, loss_60, color='crimson', linewidth=1.8, label='60% Noise L1 Loss')
    plt.yscale('log')
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('L1 Loss', fontsize=11)
    plt.title('DeepTensor PARAFAC: Loss Convergence (5000 Epochs)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(ep_30, psnr_30, color='navy', marker='o', markersize=3, label=f'30% Noise PSNR (Final: {m30["deeptensor_psnr"]:.2f} dB)')
    plt.plot(ep_60, psnr_60, color='crimson', marker='s', markersize=3, label=f'60% Noise PSNR (Final: {m60["deeptensor_psnr"]:.2f} dB)')
    plt.axhline(y=28.9, color='navy', linestyle='--', alpha=0.7, label='Paper 30% Target (28.9 dB)')
    plt.axhline(y=29.7, color='crimson', linestyle='--', alpha=0.7, label='Paper 60% Target (29.7 dB)')
    plt.axhline(y=orig_metrics['noise_30']['deeptensor_psnr'], color='navy', linestyle=':', alpha=0.7, label=f'Old 30% Baseline ({orig_metrics["noise_30"]["deeptensor_psnr"]:.2f} dB)')
    plt.axhline(y=orig_metrics['noise_60']['deeptensor_psnr'], color='crimson', linestyle=':', alpha=0.7, label=f'Old 60% Baseline ({orig_metrics["noise_60"]["deeptensor_psnr"]:.2f} dB)')
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('PSNR (dB)', fontsize=11)
    plt.title('DeepTensor PARAFAC: PSNR Progression vs Published Targets', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, loc='lower right')

    plt.tight_layout()
    comb_curves_path = os.path.join(plots_dir, 'figure10_combined_convergence_curves.png')
    plt.savefig(comb_curves_path, dpi=200)
    plt.close()
    print(f"Saved combined convergence curves to: {comb_curves_path}", flush=True)

    # Save Consolidated Comparison JSON
    summary_data = {
        'noise_30': {
            'input_noisy_psnr': m30['input_noisy_psnr'],
            'svd_psnr': m30['svd_psnr'],
            'old_baseline_psnr': orig_metrics['noise_30']['deeptensor_psnr'],
            'improved_psnr': m30['deeptensor_psnr'],
            'improved_ssim': m30['deeptensor_ssim'],
            'paper_reported_psnr': 28.90,
            'gain_over_old_db': m30['deeptensor_psnr'] - orig_metrics['noise_30']['deeptensor_psnr'],
            'gap_to_paper_db': m30['deeptensor_psnr'] - 28.90,
            'runtime_sec': m30['runtime_sec'],
            'peak_vram_mb': m30['peak_vram_mb'],
            'best_epoch': m30['best_epoch']
        },
        'noise_60': {
            'input_noisy_psnr': m60['input_noisy_psnr'],
            'svd_psnr': m60['svd_psnr'],
            'old_baseline_psnr': orig_metrics['noise_60']['deeptensor_psnr'],
            'improved_psnr': m60['deeptensor_psnr'],
            'improved_ssim': m60['deeptensor_ssim'],
            'paper_reported_psnr': 29.70,
            'gain_over_old_db': m60['deeptensor_psnr'] - orig_metrics['noise_60']['deeptensor_psnr'],
            'gap_to_paper_db': m60['deeptensor_psnr'] - 29.70,
            'runtime_sec': m60['runtime_sec'],
            'peak_vram_mb': m60['peak_vram_mb'],
            'best_epoch': m60['best_epoch']
        }
    }
    with open(os.path.join(improved_dir, 'figure10_consolidated_comparison.json'), 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved consolidated comparison JSON to: {os.path.join(improved_dir, 'figure10_consolidated_comparison.json')}", flush=True)

if __name__ == '__main__':
    main()

