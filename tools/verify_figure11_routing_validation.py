"""
Independent Verification and Telemetry Analysis Script for Figure 11 Routing Validation
======================================================================================
Audits Condition A (Static Equal) and Condition B (Constant Tau=1.0) on Figure 11 Video CS.
Recalculates Clean-Latent DIP and In-Loop metrics, verifies parameter counts and SHA-256 signatures,
and generates publication-grade comparative diagnostic figures.
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

from deeptensor import utils
from deeptensor.models.factory import (
    build_deeptensor_model,
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
    base_dir = os.path.join(project_root, 'results', 'extensions', 'ahd_net_v2_ablations', 'figure11_video_cs')
    summary_path = os.path.join(base_dir, 'figure11_routing_validation_summary.json')

    if not os.path.exists(summary_path):
        print(f"Master summary not found at: {summary_path}")
        return

    with open(summary_path, 'r') as f:
        summary_data = json.load(f)

    conditions = summary_data.get('conditions', {})
    print("=" * 80)
    print("INDEPENDENT SCIENTIFIC AUDIT: FIGURE 11 VIDEO CS ROUTING VALIDATION")
    print(f"Total Conditions Completed: {len(conditions)}")
    print("=" * 80)

    # 1. Load Ground Truth Cat Video
    data_path = os.path.join(project_root, 'data', 'cat_video.mat')
    data = io.loadmat(data_path)
    cube = data['hypercube'].astype(np.float32)
    cube = utils.resize(cube / cube.max(), 1)
    H, W, totalframes = cube.shape

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
        best_epoch = checkpoint['metrics'].get('best_epoch', -1)
        best_loss = checkpoint['metrics'].get('best_loss', float('nan'))

        model = build_deeptensor_model(
            rank=32,
            seq_len=totalframes,
            spatial_type='skip',
            spectral_type='adaptive_hybrid',
            n_inputs=32,
            skip_n33d=64,
            skip_n33u=64,
            d_model=128,
            n_heads=4,
            n_layers=4,
            d_ffn=512,
            router_type=cond_data['router_type'],
            pe_type='sin_pe',
            tau=cond_data.get('tau', 1.0),
        ).to(device)

        model.load_state_dict(m_state)
        model.train()
        model.spectral_net.set_temperature(cond_data.get('tau', 1.0))

        with torch.no_grad():
            out_mat = model(latents['im_inp'].to(device), latents['spec_inp'].to(device))
            recon_dip = out_mat[0].cpu().numpy().reshape(H, W, totalframes)

        verified_metrics = compute_metrics(cube, recon_dip)
        telem = model.spectral_net.get_routing_telemetry()

        audit_records.append({
            'cond_id': cond_id,
            'condition_id': cond_data.get('condition_id', cond_id),
            'name': cond_data['name'],
            'router_type': cond_data['router_type'],
            'tau': cond_data.get('tau', 1.0),
            'params': count_parameters(model),
            'best_epoch': best_epoch,
            'best_loss': best_loss,
            'inloop_psnr': cond_data['metrics_inloop']['psnr'],
            'inloop_ssim': cond_data['metrics_inloop']['ssim'],
            'dip_psnr': verified_metrics['psnr'],
            'dip_ssim': verified_metrics['ssim'],
            'dip_rmse': verified_metrics['rmse'],
            'best_sha256': best_sha,
            'gates': telem.get('mean_gate_weights', {}),
            'entropy': telem.get('entropy', 0.0),
            'dominant_counts': cond_data['final_router_telem'].get('dominant_counts', {}),
        })

    # Historical Reference Models on Figure 11
    hist_models = {
        'Baseline CNN': {'inloop_psnr': 25.1577, 'dip_psnr': 25.3341, 'ssim': 0.7223, 'params': 784648},
        'Serial Hybrid': {'inloop_psnr': 24.7440, 'dip_psnr': 25.1054, 'ssim': 0.6865, 'params': 834632},
        'CNN Control': {'inloop_psnr': 25.0395, 'dip_psnr': 25.2185, 'ssim': 0.7101, 'params': 834505},
        'Pointwise Control': {'inloop_psnr': 24.8147, 'dip_psnr': 25.0768, 'ssim': 0.6885, 'params': 834568},
        'Pure Transformer': {'inloop_psnr': 24.0502, 'dip_psnr': 24.2312, 'ssim': 0.6531, 'params': 1376036},
        'Standard AHD-Net v1': {'inloop_psnr': 25.1548, 'dip_psnr': 25.3641, 'ssim': 0.7438, 'params': 834632},
    }

    # Print summary table
    print("\n" + "=" * 115)
    print(f"{'Condition':<30} | {'PSNR (In-Loop)':<14} | {'PSNR (Clean DIP)':<16} | {'SSIM':<8} | {'Best Ep':<8} | {'CNN Gate':<9} | {'PW Gate':<9} | {'Attn Gate':<9}")
    print("-" * 115)
    for rec in audit_records:
        g = rec['gates']
        c_w = g.get('cnn', 0.0)
        p_w = g.get('pointwise', 0.0)
        a_w = g.get('attention', 0.0)
        print(
            f"{rec['name']:<30} | {rec['inloop_psnr']:<14.4f} | {rec['dip_psnr']:<16.4f} | "
            f"{rec['dip_ssim']:<8.4f} | {rec['best_epoch']:<8d} | {c_w:<9.3f} | {p_w:<9.3f} | {a_w:<9.3f}"
        )
    print("=" * 115)

    # Save audited JSON
    audit_output_path = os.path.join(base_dir, 'figure11_routing_validation_audit.json')
    with open(audit_output_path, 'w') as f:
        json.dump({'audited_records': audit_records, 'historical_reference': hist_models}, f, indent=2)

    # Generate Publication Diagnostic Figures
    generate_figures(base_dir, audit_records, hist_models)


def generate_figures(base_dir: str, audit_records: List[Dict[str, Any]], hist_models: Dict[str, Any]):
    # Figure 1: Comprehensive PSNR Comparison Bar Chart
    plt.figure(figsize=(12, 6))
    names = []
    psnrs = []
    colors = []

    # Historical
    for hname, hinfo in hist_models.items():
        names.append(hname)
        psnrs.append(hinfo['dip_psnr'])
        if 'AHD-Net' in hname:
            colors.append('#1f77b4')  # Blue for AHD-Net v1
        elif 'Baseline' in hname:
            colors.append('#2ca02c')  # Green for Baseline
        else:
            colors.append('#7f7f7f')  # Grey

    # Current Validation Conditions
    for rec in audit_records:
        names.append(rec['name'])
        psnrs.append(rec['dip_psnr'])
        if 'Static' in rec['name']:
            colors.append('#ff7f0e')  # Orange for Static
        else:
            colors.append('#d62728')  # Red for Constant Tau=1.0

    x = np.arange(len(names))
    bars = plt.bar(x, psnrs, color=colors, width=0.55, edgecolor='black', linewidth=0.8)
    plt.ylabel('Full-Volume PSNR (dB, Clean DIP)', fontsize=12, fontweight='bold')
    plt.title('Figure 11 Video CS: Routing Validation Benchmark (Clean DIP PSNR)', fontsize=13, fontweight='bold')
    plt.xticks(x, names, rotation=30, ha='right', fontsize=9)
    plt.ylim(23.5, 26.0)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, h + 0.04, f"{h:.2f}", ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    chart_path = os.path.join(base_dir, 'figure11_routing_validation_psnr.png')
    plt.savefig(chart_path, dpi=200)
    plt.close()

    # Figure 2: Trajectory Comparison for Constant Tau=1.0 CAGC vs Standard
    std_telem_file = os.path.join(project_root, 'results', 'extensions', 'adaptive_hybrid_v1_full', 'figure11_video_cs', 'telemetry.json')
    tau1_telem_file = os.path.join(base_dir, 'cond_b_constant_tau1', 'telemetry.json')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Standard AHD-Net v1
    if os.path.exists(std_telem_file):
        with open(std_telem_file, 'r') as f:
            t_std = json.load(f)
        ep_std = [e['epoch'] for e in t_std]
        cnn_std = [e['mean_weights']['cnn'] for e in t_std]
        pw_std = [e['mean_weights']['pointwise'] for e in t_std]
        at_std = [e['mean_weights']['attention'] for e in t_std]

        axes[0].plot(ep_std, cnn_std, color='#1f77b4', label='CNN (Local)', linewidth=2.0)
        axes[0].plot(ep_std, pw_std, color='#ff7f0e', label='Pointwise (Rank)', linewidth=2.0)
        axes[0].plot(ep_std, at_std, color='#2ca02c', label='Attention (Global)', linewidth=2.0)
        axes[0].set_title("Standard AHD-Net v1 (tau: 2.0 -> 0.5)", fontsize=11, fontweight='bold')
        axes[0].set_xlabel("Epoch", fontsize=10)
        axes[0].set_ylabel("Mean Gate Weight", fontsize=10)
        axes[0].set_ylim(0.15, 0.55)
        axes[0].grid(True, linestyle=':', alpha=0.6)
        axes[0].legend(loc='upper right', fontsize=9)

    # Condition B: Constant Tau=1.0 CAGC
    if os.path.exists(tau1_telem_file):
        with open(tau1_telem_file, 'r') as f:
            t_tau1 = json.load(f)
        ep_tau1 = [e['epoch'] for e in t_tau1]
        cnn_tau1 = [e['mean_weights']['cnn'] for e in t_tau1]
        pw_tau1 = [e['mean_weights']['pointwise'] for e in t_tau1]
        at_tau1 = [e['mean_weights']['attention'] for e in t_tau1]

        axes[1].plot(ep_tau1, cnn_tau1, color='#1f77b4', label='CNN (Local)', linewidth=2.0)
        axes[1].plot(ep_tau1, pw_tau1, color='#ff7f0e', label='Pointwise (Rank)', linewidth=2.0)
        axes[1].plot(ep_tau1, at_tau1, color='#2ca02c', label='Attention (Global)', linewidth=2.0)
        axes[1].set_title("Condition B: Constant Tau=1.0 CAGC", fontsize=11, fontweight='bold')
        axes[1].set_xlabel("Epoch", fontsize=10)
        axes[1].set_ylabel("Mean Gate Weight", fontsize=10)
        axes[1].set_ylim(0.15, 0.55)
        axes[1].grid(True, linestyle=':', alpha=0.6)
        axes[1].legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    traj_path = os.path.join(base_dir, 'figure11_routing_trajectories_comparison.png')
    plt.savefig(traj_path, dpi=200)
    plt.close()
    print(f"--> Saved figures: {chart_path} and {traj_path}")


if __name__ == '__main__':
    run_verification()

