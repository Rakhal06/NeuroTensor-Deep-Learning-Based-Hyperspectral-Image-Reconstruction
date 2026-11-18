"""
Independent Scientific Verification Script for Figure 10 AHD-Net Benchmark
==========================================================================
Audits the saved checkpoint, recalculates full-volume fidelity metrics,
verifies evaluation modes, inspects router telemetry and channel distributions.
"""

import os
import sys
import json
import math
import numpy as np
from scipy import io
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from deeptensor import utils, losses
from deeptensor.models.factory import build_deeptensor_cp_model, count_parameters
from deeptensor.utils.comparison import compute_metrics
from deeptensor.utils.metrics import calculate_psnr, calculate_ssim


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 80)
    print("INDEPENDENT SCIENTIFIC AUDIT: FIGURE 10 YALE FACE B AHD-NET BENCHMARK")
    print(f"Device: {device}")
    print("=" * 80)

    # 1. Load Ground Truth Data
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
    ptp_gt = float(np.ptp(clean_tensor))

    print(f"Dataset: Extended Yale Face B Tensor Shape: ({H}, {W}, {T})")
    print(f"Tensor Min: {clean_tensor.min():.6f}, Max: {clean_tensor.max():.6f}, Peak-to-Peak: {ptp_gt:.6f}")

    # Generate 30% salt-and-pepper noise under Seed 42
    np.random.seed(42)
    mask = np.random.rand(*clean_tensor.shape)
    noisy_tensor = clean_tensor.copy()
    noisy_tensor[mask < 0.15] = 1.0
    noisy_tensor[(mask >= 0.15) & (mask < 0.30)] = 0.0
    noisy_ten = torch.tensor(noisy_tensor, dtype=torch.float32).to(device)
    criterion_l1 = losses.L1Norm()

    # 2. Audit Checkpoint
    ckpt_dir = os.path.join(project_root, 'results', 'extensions', 'adaptive_hybrid_v1_full', 'figure10_yale_denoising')
    best_ckpt_path = os.path.join(ckpt_dir, 'best_checkpoint.pth')
    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)

    best_epoch = ckpt['epoch']
    best_loss = ckpt['metrics']['best_loss']
    tau_best = ckpt['metrics'].get('tau_at_best', 0.5000)

    print(f"\nAHD-Net Checkpoint Audit:")
    print(f"  Checkpoint File: {best_ckpt_path}")
    print(f"  Best Epoch: {best_epoch} / 1200")
    print(f"  Best Unsupervised L1 Loss: {best_loss:.8f}")
    print(f"  Prescribed Schedule Temperature: {tau_best:.6f}")

    # 3. Build Model
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
        router_type='cagc',
        pe_type='sin_pe',
        tau=tau_best,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    total_params = count_parameters(model)
    print(f"  AHD-Net Parameter Count: {total_params:,} (Target: 5,602,988, Delta: {total_params - 5602988})")

    u_inp = ckpt['config']['u_inp'].to(device)
    v_inp = ckpt['config']['v_inp'].to(device)
    w_inp = ckpt['config']['w_inp'].to(device)

    # Evaluation Protocol 1: DIP Train Mode (Clean Latents, Prescribed tau)
    model.train()
    model.w_net.adaptive_block.set_temperature(tau_best)
    with torch.no_grad():
        recon_dip = model(u_inp, v_inp, w_inp).cpu().numpy()
    m_dip = compute_metrics(clean_tensor, recon_dip)

    # Evaluation Protocol 2: Diagnostic Unit Temperature tau=1.0
    model.w_net.adaptive_block.set_temperature(1.0)
    with torch.no_grad():
        recon_tau1 = model(u_inp, v_inp, w_inp).cpu().numpy()
    m_tau1 = compute_metrics(clean_tensor, recon_tau1)
    model.w_net.adaptive_block.set_temperature(tau_best)

    # Evaluation Protocol 3: Eval Mode (Running Statistics)
    model.eval()
    with torch.no_grad():
        recon_eval = model(u_inp, v_inp, w_inp).cpu().numpy()
    m_eval = compute_metrics(clean_tensor, recon_eval)

    print("\n" + "=" * 80)
    print("AHD-NET RECONSTRUCTION METRICS UNDER ALL EVALUATION PROTOCOLS:")
    print("=" * 80)
    print(f"Protocol 1 (DIP Train Mode, Prescribed tau={tau_best:.4f}):")
    print(f"  PSNR: {m_dip['psnr']:.4f} dB | SSIM: {m_dip['ssim']:.4f} | RMSE: {m_dip['rmse']:.6f} | MAE: {m_dip['mae']:.6f} | RelErr: {m_dip['rel_error']:.6f}")
    print(f"Protocol 2 (DIP Train Mode, Diagnostic tau=1.0000):")
    print(f"  PSNR: {m_tau1['psnr']:.4f} dB | SSIM: {m_tau1['ssim']:.4f} | RMSE: {m_tau1['rmse']:.6f}")
    print(f"Protocol 3 (Eval Mode, Running Statistics, tau={tau_best:.4f}):")
    print(f"  PSNR: {m_eval['psnr']:.4f} dB | SSIM: {m_eval['ssim']:.4f} | RMSE: {m_eval['rmse']:.6f}")

    # Central Slice Verification
    central_t = T // 2
    gt_slice = clean_tensor[:, :, central_t]
    recon_slice = recon_dip[:, :, central_t]
    psnr_slice_local = calculate_psnr(gt_slice, recon_slice, data_range=float(np.ptp(gt_slice)))
    psnr_slice_global = calculate_psnr(gt_slice, recon_slice, data_range=1.0)
    print("\nCentral Slice 80 Disambiguation:")
    print(f"  Full Volume PSNR:         {m_dip['psnr']:.4f} dB (Evaluated on 192x168x160 tensor)")
    print(f"  Central Slice Local PSNR: {psnr_slice_local:.4f} dB (ptp={np.ptp(gt_slice):.4f})")
    print(f"  Central Slice Global PSNR:{psnr_slice_global:.4f} dB (ptp=1.0000)")

    # Router Gate Breakdown
    ab = model.w_net.adaptive_block
    gates = ab.last_gates[0].cpu().numpy()  # [128, 3]
    mean_w = gates.mean(axis=0)
    dominant = np.argmax(gates, axis=-1)
    num_cnn = np.sum(dominant == 0)
    num_pw = np.sum(dominant == 1)
    num_attn = np.sum(dominant == 2)
    entropy = -np.sum(mean_w * np.log(mean_w + 1e-9))

    print("\n" + "=" * 80)
    print("ROUTER TELEMETRY & CHANNEL SPECIALIZATION AUDIT:")
    print("=" * 80)
    print(f"Mean Branch Weights: CNN={mean_w[0]:.4f}, Pointwise={mean_w[1]:.4f}, Attention={mean_w[2]:.4f}")
    print(f"Router Entropy:      {entropy:.4f} nats (Max ln(3) = {math.log(3.0):.4f})")
    print(f"Channel Dominance:   CNN={num_cnn}/128 ({num_cnn/128*100:.1f}%), PW={num_pw}/128 ({num_pw/128*100:.1f}%), Attn={num_attn}/128 ({num_attn/128*100:.1f}%)")
    for b_idx, b_name in enumerate(['CNN', 'Pointwise', 'Attention']):
        w_b = gates[:, b_idx]
        print(f"  {b_name:10s}: Mean={np.mean(w_b):.4f}, Std={np.std(w_b):.4f}, Min={np.min(w_b):.4f}, Max={np.max(w_b):.4f}")

    # Load historical models
    hist_json = os.path.join(project_root, 'results', 'hybrid', 'figure10_yale_denoising', 'figure10_yale_b_comparison.json')
    with open(hist_json, 'r') as f:
        hist_data = json.load(f)

    print("\n" + "=" * 80)
    print("OFFICIAL BENCHMARK COMPARISON TABLE (FIGURE 10 YALE FACE B):")
    print("=" * 80)
    print(f"{'Model Architecture':<22} | {'PSNR (dB)':<10} | {'SSIM':<8} | {'RMSE':<9} | {'Params':<10} | {'Delta vs Base':<12}")
    print("-" * 80)
    base_psnr = hist_data['models']['baseline']['metrics']['psnr']
    for name in ['baseline', 'hybrid', 'cnn_control', 'pointwise_control', 'transformer']:
        m = hist_data['models'][name]['metrics']
        p = hist_data['models'][name]['params']
        d = m['psnr'] - base_psnr
        print(f"{name:<22} | {m['psnr']:<10.4f} | {m['ssim']:<8.4f} | {m['rmse']:<9.5f} | {p:<10,d} | {d:+.4f} dB")
    d_ahd = m_dip['psnr'] - base_psnr
    print(f"{'AHD-Net v1 (Ours)':<22} | {m_dip['psnr']:<10.4f} | {m_dip['ssim']:<8.4f} | {m_dip['rmse']:<9.5f} | {total_params:<10,d} | {d_ahd:+.4f} dB")
    print("=" * 80)


if __name__ == '__main__':
    main()

