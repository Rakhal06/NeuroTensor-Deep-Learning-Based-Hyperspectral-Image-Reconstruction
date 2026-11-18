"""
DeepTensor++ Side-by-Side Model Comparison Infrastructure
=========================================================
Standardized framework for rigorous comparative evaluation between
CNN Baselines, Hybrid CNN-Attention, and Transformer architectures.

Features:
- Full numerical metric computation (PSNR, SSIM, RMSE, MAE, Relative Frobenius Error).
- Parameter count and memory footprint auditing.
- Multi-panel publication-ready visual comparison figures with error maps.
- Export to JSON, CSV, and high-resolution figures (PNG/PDF).
"""

import os
import json
import csv
from typing import Dict, Any, Optional, List, Tuple, Union

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .metrics import calculate_psnr, calculate_ssim


def compute_metrics(
    gt: np.ndarray,
    recon: np.ndarray,
    data_range: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute comprehensive reconstruction fidelity metrics.

    Args:
        gt: Ground-truth array (numpy or torch tensor).
        recon: Reconstructed array.
        data_range: Dynamic range of signal. If None, computed as max(gt) - min(gt).

    Returns:
        Dictionary with 'psnr', 'ssim', 'rmse', 'mae', and 'rel_error'.
    """
    gt_np = gt.detach().cpu().numpy() if hasattr(gt, 'detach') else np.asarray(gt, dtype=np.float32)
    recon_np = recon.detach().cpu().numpy() if hasattr(recon, 'detach') else np.asarray(recon, dtype=np.float32)

    if data_range is None:
        data_range = float(max(np.ptp(gt_np), 1e-6))

    # Compute PSNR & SSIM
    psnr = calculate_psnr(gt_np, recon_np, data_range=data_range)
    
    # SSIM channel axis handling
    channel_axis = -1 if gt_np.ndim == 3 and gt_np.shape[-1] in (3, 4, 31, 64, 160, 260) else None
    if gt_np.ndim == 3 and channel_axis is None and gt_np.shape[0] < min(gt_np.shape[1], gt_np.shape[2]):
        channel_axis = 0
        
    ssim = calculate_ssim(gt_np, recon_np, data_range=data_range, channel_axis=channel_axis)

    # Error metrics
    diff = gt_np - recon_np
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    rel_error = float(np.linalg.norm(diff) / max(np.linalg.norm(gt_np), 1e-9))

    return {
        'psnr': float(psnr),
        'ssim': float(ssim),
        'rmse': float(rmse),
        'mae': float(mae),
        'rel_error': float(rel_error),
    }


def to_2d_slice(tensor: np.ndarray, slice_idx: Optional[int] = None) -> np.ndarray:
    """Extract a representative 2D slice from 2D, 3D, or 4D array for visualization."""
    arr = tensor.detach().cpu().numpy() if hasattr(tensor, 'detach') else np.asarray(tensor)
    arr = np.squeeze(arr)

    if arr.ndim == 2:
        return arr
    elif arr.ndim == 3:
        # Determine slice axis: pick middle slice of shortest or last dimension
        if slice_idx is None:
            # If shape is [H, W, C], slice along C; if [C, H, W], slice along C
            if arr.shape[0] < min(arr.shape[1], arr.shape[2]):
                idx = arr.shape[0] // 2
                return arr[idx, :, :]
            else:
                idx = arr.shape[2] // 2
                return arr[:, :, idx]
        else:
            if arr.shape[0] < min(arr.shape[1], arr.shape[2]):
                return arr[min(slice_idx, arr.shape[0] - 1), :, :]
            else:
                return arr[:, :, min(slice_idx, arr.shape[2] - 1)]
    elif arr.ndim == 4:
        return arr[0, arr.shape[1] // 2, :, :]
    else:
        return arr.reshape(arr.shape[0], -1)


class SideBySideComparator:
    """
    Manages side-by-side benchmarking and visual report generation
    for baseline, hybrid, transformer, and control model variants.
    """

    def __init__(self, experiment_name: str, output_dir: str):
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.records: Dict[str, Dict[str, Any]] = {}

    def add_model_result(
        self,
        name: str,
        recon: Union[np.ndarray, torch.Tensor],
        params: Optional[int] = None,
        runtime_sec: Optional[float] = None,
        peak_vram_mb: Optional[float] = None,
        loss_history: Optional[List[float]] = None,
        extra_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a model's reconstruction and performance profile."""
        recon_np = recon.detach().cpu().numpy() if hasattr(recon, 'detach') else np.asarray(recon, dtype=np.float32)
        self.records[name] = {
            'recon': recon_np,
            'params': params,
            'runtime_sec': runtime_sec,
            'peak_vram_mb': peak_vram_mb,
            'loss_history': loss_history,
            'extra_metrics': extra_metrics or {},
        }

    def generate_comparison(
        self,
        gt: Union[np.ndarray, torch.Tensor],
        observation: Optional[Union[np.ndarray, torch.Tensor]] = None,
        slice_idx: Optional[int] = None,
        save_figures: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate all registered models against ground truth and produce comparison artifacts.

        Returns:
            Dictionary containing compiled metrics, rankings, and file paths.
        """
        gt_np = gt.detach().cpu().numpy() if hasattr(gt, 'detach') else np.asarray(gt, dtype=np.float32)
        obs_np = None
        if observation is not None:
            obs_np = observation.detach().cpu().numpy() if hasattr(observation, 'detach') else np.asarray(observation, dtype=np.float32)

        data_range = float(max(np.ptp(gt_np), 1e-6))
        summary: Dict[str, Any] = {
            'experiment': self.experiment_name,
            'models': {},
            'comparisons': {},
        }

        # 1. Compute metrics for each model
        for name, data in self.records.items():
            metrics = compute_metrics(gt_np, data['recon'], data_range=data_range)
            model_entry = {
                'metrics': metrics,
                'params': data['params'],
                'runtime_sec': data['runtime_sec'],
                'peak_vram_mb': data['peak_vram_mb'],
                'extra_metrics': data['extra_metrics'],
            }
            summary['models'][name] = model_entry

        # 2. Relative comparisons against baseline (if 'baseline' or 'cnn' exists)
        ref_key = 'baseline' if 'baseline' in self.records else ('cnn' if 'cnn' in self.records else None)
        if ref_key:
            ref_metrics = summary['models'][ref_key]['metrics']
            ref_params = summary['models'][ref_key]['params']
            for name, entry in summary['models'].items():
                if name == ref_key:
                    continue
                d_psnr = entry['metrics']['psnr'] - ref_metrics['psnr']
                d_ssim = entry['metrics']['ssim'] - ref_metrics['ssim']
                d_rmse = entry['metrics']['rmse'] - ref_metrics['rmse']
                param_ratio = (entry['params'] / ref_params) if (entry['params'] and ref_params) else None
                summary['comparisons'][f'{name}_vs_{ref_key}'] = {
                    'delta_psnr_db': float(d_psnr),
                    'delta_ssim': float(d_ssim),
                    'delta_rmse': float(d_rmse),
                    'param_ratio': param_ratio,
                    'psnr_improved': bool(d_psnr > 0),
                }

        # 3. Export CSV summary
        csv_path = os.path.join(self.output_dir, f'{self.experiment_name}_comparison.csv')
        fieldnames = ['Model', 'PSNR (dB)', 'SSIM', 'RMSE', 'MAE', 'Params', 'Runtime (s)', 'VRAM (MB)']
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(fieldnames)
            for name, entry in summary['models'].items():
                m = entry['metrics']
                writer.writerow([
                    name,
                    f"{m['psnr']:.4f}",
                    f"{m['ssim']:.4f}",
                    f"{m['rmse']:.6f}",
                    f"{m['mae']:.6f}",
                    entry['params'] if entry['params'] is not None else 'N/A',
                    f"{entry['runtime_sec']:.2f}" if entry['runtime_sec'] is not None else 'N/A',
                    f"{entry['peak_vram_mb']:.2f}" if entry['peak_vram_mb'] is not None else 'N/A',
                ])
        summary['csv_path'] = csv_path

        # 4. Export JSON summary
        json_path = os.path.join(self.output_dir, f'{self.experiment_name}_comparison.json')
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        summary['json_path'] = json_path

        # 5. Visual Comparison Panels
        if save_figures and len(self.records) > 0:
            fig_path = self._generate_visual_panels(gt_np, obs_np, slice_idx=slice_idx)
            summary['figure_path'] = fig_path

        return summary

    def _generate_visual_panels(
        self,
        gt_np: np.ndarray,
        obs_np: Optional[np.ndarray],
        slice_idx: Optional[int] = None,
    ) -> str:
        """Construct multi-panel visual grid showing reconstructions, error maps, and differences."""
        gt_slice = to_2d_slice(gt_np, slice_idx)
        models = list(self.records.keys())
        n_models = len(models)

        # Layout:
        # Row 1: GT, Observation (or Model 1), Reconstructions...
        # Row 2: Error maps (|Recon - GT|) for each model + Difference Map between model 0 and model 1
        cols = max(3, n_models + (2 if obs_np is not None else 1))
        fig, axes = plt.subplots(2, cols, figsize=(4 * cols, 7))

        # Panel 1: Ground Truth
        axes[0, 0].imshow(gt_slice, cmap='gray')
        axes[0, 0].set_title("Ground Truth", fontsize=11, fontweight='bold')
        axes[0, 0].axis('off')

        # Error map placeholder for GT is black / zero
        axes[1, 0].imshow(np.zeros_like(gt_slice), cmap='inferno', vmin=0, vmax=0.1)
        axes[1, 0].set_title("Zero Error Ref", fontsize=10)
        axes[1, 0].axis('off')

        col_offset = 1
        # Panel 2 (Optional): Noisy/Undersampled Observation
        if obs_np is not None:
            obs_slice = to_2d_slice(obs_np, slice_idx)
            axes[0, col_offset].imshow(obs_slice, cmap='gray')
            axes[0, col_offset].set_title("Degraded Input", fontsize=11, fontweight='bold')
            axes[0, col_offset].axis('off')

            obs_err = np.abs(gt_slice - obs_slice)
            im_err = axes[1, col_offset].imshow(obs_err, cmap='inferno')
            axes[1, col_offset].set_title(f"Input Error (MAE: {np.mean(obs_err):.3f})", fontsize=10)
            axes[1, col_offset].axis('off')
            col_offset += 1

        # Model Reconstruction & Error Panels
        max_err = 1e-6
        recon_slices = {}
        for m_idx, m_name in enumerate(models):
            m_slice = to_2d_slice(self.records[m_name]['recon'], slice_idx)
            recon_slices[m_name] = m_slice
            cur_err = np.abs(gt_slice - m_slice)
            max_err = max(max_err, float(np.percentile(cur_err, 99)))

        for m_idx, m_name in enumerate(models):
            c = col_offset + m_idx
            if c >= cols:
                break
            m_slice = recon_slices[m_name]
            err_map = np.abs(gt_slice - m_slice)
            
            psnr_val = calculate_psnr(gt_slice, m_slice, data_range=float(max(np.ptp(gt_slice), 1e-6)))

            axes[0, c].imshow(m_slice, cmap='gray')
            axes[0, c].set_title(f"{m_name.upper()}\nPSNR: {psnr_val:.2f} dB", fontsize=11, fontweight='bold')
            axes[0, c].axis('off')

            im_m = axes[1, c].imshow(err_map, cmap='inferno', vmin=0, vmax=max_err)
            axes[1, c].set_title(f"{m_name} Error Map\nMAE: {np.mean(err_map):.4f}", fontsize=10)
            axes[1, c].axis('off')

        # Clear remaining unused axes
        for r in range(2):
            for c in range(col_offset + n_models, cols):
                axes[r, c].axis('off')

        # Add shared colorbar on the right
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.3])
        cbar = fig.colorbar(im_m, cax=cbar_ax)
        cbar.set_label('Absolute Error |GT - Recon|', fontsize=9)

        plt.subplots_adjust(top=0.90, bottom=0.08, left=0.03, right=0.90, hspace=0.25, wspace=0.15)
        plt.suptitle(f"DeepTensor++ Architecture Comparison: {self.experiment_name}", fontsize=14, fontweight='bold')

        fig_path = os.path.join(self.output_dir, f"{self.experiment_name}_comparison_panel.png")
        plt.savefig(fig_path, dpi=200, bbox_inches='tight')
        plt.close(fig)

        # Generate dedicated Difference Map (Hybrid vs Parameter-Matched CNN / Baseline)
        ref_model = 'cnn_control' if 'cnn_control' in recon_slices else ('baseline' if 'baseline' in recon_slices else None)
        if 'hybrid' in recon_slices and ref_model:
            fig_diff, ax_d = plt.subplots(1, 3, figsize=(14, 4))
            h_slice = recon_slices['hybrid']
            c_slice = recon_slices[ref_model]
            abs_diff = np.abs(h_slice - c_slice)
            signed_diff = h_slice - c_slice

            ax_d[0].imshow(h_slice, cmap='gray')
            ax_d[0].set_title("Hybrid Reconstruction", fontsize=11, fontweight='bold')
            ax_d[0].axis('off')

            ax_d[1].imshow(c_slice, cmap='gray')
            ax_d[1].set_title(f"{ref_model.upper()} Reconstruction", fontsize=11, fontweight='bold')
            ax_d[1].axis('off')

            vmax = max(1e-5, float(np.percentile(abs_diff, 99.5)))
            im_d = ax_d[2].imshow(signed_diff, cmap='coolwarm', vmin=-vmax, vmax=vmax)
            ax_d[2].set_title(f"Difference (Hybrid - {ref_model})\nMean Abs Diff: {np.mean(abs_diff):.4f}", fontsize=11, fontweight='bold')
            ax_d[2].axis('off')

            fig_diff.colorbar(im_d, ax=ax_d[2], fraction=0.046, pad=0.04)
            plt.suptitle(f"{self.experiment_name}: Hybrid vs {ref_model} Residual Analysis", fontsize=13, fontweight='bold')
            diff_path = os.path.join(self.output_dir, f"{self.experiment_name}_diff_map.png")
            plt.savefig(diff_path, dpi=200, bbox_inches='tight')
            plt.close(fig_diff)

        return fig_path
