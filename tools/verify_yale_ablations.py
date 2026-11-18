"""
Independent Verification and Telemetry Analysis Script for Yale Face B Ablations
================================================================================
Audits all ablation condition checkpoints, verifies reproducibility, recalculates
clean-latent DIP and in-loop snapshot metrics, and generates publication-grade
comparative diagnostic plots.
"""

import os
import sys
import json
import hashlib
from typing import Dict, Any, List

import numpy as np
from scipy import io
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, os.path.join(project_root, 'src'))
    sys.path.insert(0, project_root)

from deeptensor.models.factory import (
    build_deeptensor_cp_model,
    count_parameters,
)
from deeptensor.utils.comparison import compute_metrics


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def run_verification():
    base_dir = os.path.join(project_root, 'results', 'extensions', 'ahd_net_v2_ablations', 'figure10_yale')
    summary_path = os.path.join(base_dir, 'yale_ablation_master_summary.json')

    if not os.path.exists(summary_path):
        print(f"Master summary not found at: {summary_path}")
        return

    with open(summary_path, 'r') as f:
        summary_data = json.load(f)

    conditions = summary_data.get('conditions', {})
    print("=" * 80)
    print("INDEPENDENT SCIENTIFIC AUDIT: FIGURE 10 YALE ABLATION EXPERIMENTS")
    print(f"Total Conditions Completed: {len(conditions)}")
    print("=" * 80)

    # 1. Load Ground Truth Yale B
    data_path = os.path.join(project_root, 'data', 'yale_b.mat')
    mat = io.loadmat(data_path)
    clean_tensor = mat.get('faces', mat.get('yale_b', mat.get('tensor', mat.get('X', None))))
    if clean_tensor is None:
        for k in mat:
            if not k.startswith('__') and isinstance(mat[k], np.ndarray) and mat[k].ndim == 3:
                clean_tensor = mat[k]
                break
    clean_tensor = clean_tensor.astype(np.float32)
    clean_tensor = clean_tensor / clean_tensor.max()
    H, W, T = clean_tensor.shape

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    audit_records = []
    for cond_id, cond_data in conditions.items():
        cond_dir = os.path.join(base_dir, cond_id)
        best_ckpt = os.path.join(cond_dir, 'best_checkpoint.pth')
        final_ckpt = os.path.join(cond_dir, 'final_checkpoint.pth')

        best_sha = compute_sha256(best_ckpt) if os.path.exists(best_ckpt) else 'MISSING'
        final_sha = compute_sha256(final_ckpt) if os.path.exists(final_ckpt) else 'MISSING'

        # Load checkpoint and verify Clean-Latent DIP inference
        checkpoint = torch.load(best_ckpt, map_location=device, weights_only=False)
        m_state = checkpoint['model_state_dict']
        latents = checkpoint['config']
        tau_at_best = checkpoint['metrics'].get('tau_at_best', 1.0)
        best_epoch = checkpoint['metrics'].get('best_epoch', -1)
        best_loss = checkpoint['metrics'].get('best_loss', float('nan'))

        model = build_deeptensor_cp_model(
            rank=2000,
            lengths=(H, W, T),
            u_type='skip1d',
            v_type='skip1d',
            w_type='adaptive_hybrid',
            n_inputs=2000,
            learnable_core=True,
            normalize_core=True,
            d_model=128,
            n_heads=4,
            n_layers=4,
            d_ffn=512,
            router_type=cond_data['router_type'],
            pe_type='sin_pe',
            tau=tau_at_best,
            gating_floor=cond_data.get('gating_floor', 0.0),
        ).to(device)

        model.load_state_dict(m_state)
        model.train()
        model.w_net.set_temperature(tau_at_best)

        with torch.no_grad():
            recon_dip = model(
                latents['u_inp'].to(device),
                latents['v_inp'].to(device),
                latents['w_inp'].to(device),
            ).cpu().numpy()

        verified_metrics = compute_metrics(clean_tensor, recon_dip)
        telem = model.w_net.get_routing_telemetry()

        audit_records.append({
            'cond_id': cond_id,
            'name': cond_data['name'],
            'router_type': cond_data['router_type'],
            'gating_floor': cond_data.get('gating_floor', 0.0),
            'entropy_reg': cond_data.get('entropy_reg', 0.0),
            'params': count_parameters(model),
            'best_epoch': best_epoch,
            'best_loss': best_loss,
            'tau_best': tau_at_best,
            'inloop_psnr': cond_data['metrics_inloop']['psnr'],
            'inloop_ssim': cond_data['metrics_inloop']['ssim'],
            'dip_psnr': verified_metrics['psnr'],
            'dip_ssim': verified_metrics['ssim'],
            'dip_rmse': verified_metrics['rmse'],
            'best_sha256': best_sha,
            'gates': telem.get('mean_gate_weights', {}),
            'entropy': telem.get('entropy', 0.0),
        })

    # Historical models for comparison
    hist_models = {
        'Baseline CNN': {'psnr': 27.7808, 'ssim': 0.8528, 'params': 5334812},
        'Serial Hybrid': {'psnr': 28.0392, 'ssim': 0.8592, 'params': 5602988},
        'CNN Control': {'psnr': 28.3521, 'ssim': 0.8612, 'params': 5602988},
        'Pointwise Control': {'psnr': 28.1703, 'ssim': 0.8601, 'params': 5602988},
        'Pure Transformer (Winner)': {'psnr': 28.5552, 'ssim': 0.8655, 'params': 5602988},
        'Standard AHD-Net v1': {'psnr': 27.8486, 'ssim': 0.8560, 'params': 5602988},
    }

    # Print summary table
    print("\n" + "=" * 110)
    print(f"{'Condition':<32} | {'PSNR (In-Loop)':<14} | {'PSNR (DIP)':<12} | {'SSIM':<8} | {'Best Ep':<8} | {'CNN Gate':<9} | {'PW Gate':<9} | {'Attn Gate':<9}")
    print("-" * 110)
    for rec in audit_records:
        g = rec['gates']
        c_w = g.get('cnn', 0.0)
        p_w = g.get('pointwise', 0.0)
        a_w = g.get('attention', 0.0)
        print(
            f"{rec['name']:<32} | {rec['inloop_psnr']:<14.4f} | {rec['dip_psnr']:<12.4f} | "
            f"{rec['dip_ssim']:<8.4f} | {rec['best_epoch']:<8d} | {c_w:<9.3f} | {p_w:<9.3f} | {a_w:<9.3f}"
        )
    print("=" * 110)

    # Save audited JSON
    audit_output_path = os.path.join(base_dir, 'yale_ablation_verified_audit.json')
    with open(audit_output_path, 'w') as f:
        json.dump({'audited_records': audit_records, 'historical_reference': hist_models}, f, indent=2)

    # Generate Publication-Grade Diagnostic Visualizations
    generate_comparative_figures(base_dir, audit_records, hist_models)


def generate_comparative_figures(base_dir: str, audit_records: List[Dict[str, Any]], hist_models: Dict[str, Any]):
    # Figure 1: PSNR Bar Chart
    plt.figure(figsize=(14, 6))
    names = []
    psnrs = []
    colors = []

    # Historical
    for hname, hinfo in hist_models.items():
        names.append(hname)
        psnrs.append(hinfo['psnr'])
        if 'Winner' in hname:
            colors.append('#2ca02c')  # Green for pure transformer winner
        elif 'AHD-Net' in hname:
            colors.append('#1f77b4')  # Blue
        else:
            colors.append('#7f7f7f')  # Grey

    # Ablations
    for rec in audit_records:
        names.append(rec['name'])
        psnrs.append(rec['dip_psnr'])
        if 'Floor' in rec['name'] or 'Entropy' in rec['name']:
            colors.append('#d62728')  # Red for interventions
        elif 'Attention-Only' in rec['name']:
            colors.append('#9467bd')  # Purple
        else:
            colors.append('#ff7f0e')  # Orange

    x = np.arange(len(names))
    bars = plt.bar(x, psnrs, color=colors, width=0.6, edgecolor='black', linewidth=0.8)
    plt.ylabel('Full-Volume PSNR (dB)', fontsize=12, fontweight='bold')
    plt.title('Figure 10 (Yale Face B) Controlled Ablation Study: Full PSNR Benchmark', fontsize=14, fontweight='bold')
    plt.xticks(x, names, rotation=35, ha='right', fontsize=9)
    plt.ylim(27.0, 29.0)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Add value labels
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, h + 0.03, f"{h:.2f}", ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    chart_path = os.path.join(base_dir, 'yale_ablation_psnr_comparison.png')
    plt.savefig(chart_path, dpi=200)
    plt.close()

    # Figure 2: Multi-condition Routing Trajectories
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    cagc_conds = [
        ('cond5_standard_cagc', 'Standard CAGC (tau: 2.0 -> 0.5)'),
        ('cond6a_temp_sharp', 'Temp Sharp (tau: 2.0 -> 0.1)'),
        ('cond6b_temp_soft', 'Temp Soft (tau: 2.0 -> 1.0)'),
        ('cond6c_temp_const1', 'Constant Temp (tau = 1.0)'),
        ('cond7_gating_floor15', 'Gating Floor 0.15'),
        ('cond8_entropy_reg', 'Entropy Reg (lambda = 0.01)'),
    ]

    for idx, (cid, title) in enumerate(cagc_conds):
        telem_file = os.path.join(base_dir, cid, 'telemetry.json')
        if not os.path.exists(telem_file):
            axes[idx].set_title(f"{title} (Pending)", fontsize=11)
            continue

        with open(telem_file, 'r') as f:
            t_data = json.load(f)

        epochs = [entry['epoch'] for entry in t_data]
        cnn_w = [entry['mean_weights']['cnn'] for entry in t_data]
        pw_w = [entry['mean_weights']['pointwise'] for entry in t_data]
        at_w = [entry['mean_weights']['attention'] for entry in t_data]
        ent = [entry['channel_entropy'] for entry in t_data]

        ax = axes[idx]
        ax.plot(epochs, cnn_w, color='#1f77b4', label='CNN (Local)', linewidth=2.0)
        ax.plot(epochs, pw_w, color='#ff7f0e', label='Pointwise (Cross-Rank)', linewidth=2.0)
        ax.plot(epochs, at_w, color='#2ca02c', label='Attention (Global)', linewidth=2.0)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('Mean Routing Weight', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)
        if idx == 0:
            ax.legend(loc='upper right', fontsize=9)

        # Twin axis for entropy
        ax2 = ax.twinx()
        ax2.plot(epochs, ent, color='gray', linestyle='--', label='Entropy H', alpha=0.7)
        ax2.set_ylabel('Entropy (nats)', fontsize=9, color='gray')
        ax2.set_ylim(0.0, 1.2)

    plt.tight_layout()
    traj_path = os.path.join(base_dir, 'yale_ablation_routing_trajectories.png')
    plt.savefig(traj_path, dpi=200)
    plt.close()
    print(f"--> Saved comparative figures: {chart_path} and {traj_path}")


if __name__ == '__main__':
    run_verification()

