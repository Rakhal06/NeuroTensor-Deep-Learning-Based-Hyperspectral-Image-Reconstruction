"""
Independent Scientific Verification Script for Figure 11 AHD-Net Benchmark
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
from deeptensor.physics import operators as lin_inverse
from deeptensor.models.factory import build_deeptensor_model, count_parameters
from deeptensor.utils.comparison import compute_metrics
from deeptensor.utils.metrics import calculate_psnr, calculate_ssim


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 80)
    print("INDEPENDENT SCIENTIFIC AUDIT: FIGURE 11 VIDEO CS AHD-NET BENCHMARK")
    print(f"Device: {device}")
    print("=" * 80)

    # 1. Load Ground Truth Data
    data_path = os.path.join(project_root, 'data', 'cat_video.mat')
    data = io.loadmat(data_path)
    cube = data['hypercube'].astype(np.float32)
    cube = utils.resize(cube / cube.max(), 1)
    H, W, totalframes = cube.shape
    ptp_gt = float(np.ptp(cube))

    print(f"Dataset: Cat Video Volume Shape: ({H}, {W}, {totalframes})")
    print(f"Volume Min: {cube.min():.6f}, Max: {cube.max():.6f}, Peak-to-Peak: {ptp_gt:.6f}")

    # Generate CACTI observation under Seed 42
    np.random.seed(42)
    torch.manual_seed(42)
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
    criterion_l2 = losses.L2Norm()

    # 2. Audit Checkpoint
    ckpt_dir = os.path.join(project_root, 'results', 'extensions', 'adaptive_hybrid_v1_full', 'figure11_video_cs')
    best_ckpt_path = os.path.join(ckpt_dir, 'best_checkpoint.pth')
    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)

    best_epoch = ckpt['epoch']
    best_loss = ckpt['metrics']['best_loss']
    tau_best = ckpt['metrics'].get('tau_at_best', 0.5000)

    print(f"\nAHD-Net Checkpoint Audit:")
    print(f"  Checkpoint File: {best_ckpt_path}")
    print(f"  Best Epoch: {best_epoch} / 3000")
    print(f"  Best Unsupervised L2 Loss: {best_loss:.8e}")
    print(f"  Prescribed Schedule Temperature: {tau_best:.6f}")

    # 3. Build Model
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
        router_type='cagc',
        pe_type='sin_pe',
        tau=tau_best,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    total_params = count_parameters(model)
    print(f"  AHD-Net Parameter Count: {total_params:,} (Target: 834,632, Delta: {total_params - 834632})")

    im_inp = ckpt['config']['im_inp'].to(device)
    spec_inp = ckpt['config']['spec_inp'].to(device)

    # Mode 1: Clean Latents DIP Inference (Prescribed tau)
    model.train()
    model.spectral_net.adaptive_block.set_temperature(tau_best)
    with torch.no_grad():
        out_mat = model(im_inp, spec_inp)
        recon_dip = out_mat[0].cpu().numpy().reshape(H, W, totalframes)
    m_dip = compute_metrics(cube, recon_dip)

    # Mode 2: In-Loop Perturbed Simulation
    reg_noise_std = 1.0 / 30.0
    torch.manual_seed(2996)
    im_pert = im_inp + torch.randn_like(im_inp) * reg_noise_std
    spec_pert = spec_inp + torch.randn_like(spec_inp) * reg_noise_std
    with torch.no_grad():
        out_mat_pert = model(im_pert, spec_pert)
        recon_pert = out_mat_pert[0].cpu().numpy().reshape(H, W, totalframes)
    m_pert = compute_metrics(cube, recon_pert)

    # Mode 3: Diagnostic Unit Temperature tau=1.0
    model.spectral_net.adaptive_block.set_temperature(1.0)
    with torch.no_grad():
        out_mat_tau1 = model(im_inp, spec_inp)
        recon_tau1 = out_mat_tau1[0].cpu().numpy().reshape(H, W, totalframes)
    m_tau1 = compute_metrics(cube, recon_tau1)
    model.spectral_net.adaptive_block.set_temperature(tau_best)

    # Mode 4: Eval Mode (Running Statistics)
    model.eval()
    with torch.no_grad():
        out_mat_eval = model(im_inp, spec_inp)
        recon_eval = out_mat_eval[0].cpu().numpy().reshape(H, W, totalframes)
    m_eval = compute_metrics(cube, recon_eval)

    print("\n" + "=" * 80)
    print("AHD-NET RECONSTRUCTION METRICS UNDER ALL EVALUATION PROTOCOLS:")
    print("=" * 80)
    print(f"Protocol 1 (Clean Latent DIP Inference, tau={tau_best:.4f}):")
    print(f"  PSNR: {m_dip['psnr']:.4f} dB | SSIM: {m_dip['ssim']:.4f} | RMSE: {m_dip['rmse']:.6f} | MAE: {m_dip['mae']:.6f} | RelErr: {m_dip['rel_error']:.6f}")
    print(f"Protocol 2 (In-Loop Perturbed Simulation):")
    print(f"  PSNR: {m_pert['psnr']:.4f} dB | SSIM: {m_pert['ssim']:.4f} | RMSE: {m_pert['rmse']:.6f}")
    print(f"Protocol 3 (Diagnostic tau=1.0000):")
    print(f"  PSNR: {m_tau1['psnr']:.4f} dB | SSIM: {m_tau1['ssim']:.4f} | RMSE: {m_tau1['rmse']:.6f}")
    print(f"Protocol 4 (Eval Mode, Running Statistics):")
    print(f"  PSNR: {m_eval['psnr']:.4f} dB | SSIM: {m_eval['ssim']:.4f} | RMSE: {m_eval['rmse']:.6f}")

    # Central Frame Disambiguation
    central_t = totalframes // 2  # Frame 32
    gt_frame = cube[:, :, central_t]
    recon_frame = recon_dip[:, :, central_t]
    psnr_frame_local = calculate_psnr(gt_frame, recon_frame, data_range=float(np.ptp(gt_frame)))
    psnr_frame_global = calculate_psnr(gt_frame, recon_frame, data_range=1.0)
    print("\nCentral Frame 32 Disambiguation:")
    print(f"  Full Volume PSNR:        {m_dip['psnr']:.4f} dB (Evaluated on 128x128x64 volume)")
    print(f"  Central Frame Local PSNR:{psnr_frame_local:.4f} dB (ptp={np.ptp(gt_frame):.4f})")
    print(f"  Central Frame Global PSNR:{psnr_frame_global:.4f} dB (ptp=1.0000)")

    # Router Gate Breakdown
    ab = model.spectral_net.adaptive_block
    gates = ab.last_gates[0].cpu().numpy()  # [C, 3]
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
    print(f"Channel Dominance:   CNN={num_cnn}/{len(gates)} ({num_cnn/len(gates)*100:.1f}%), PW={num_pw}/{len(gates)} ({num_pw/len(gates)*100:.1f}%), Attn={num_attn}/{len(gates)} ({num_attn/len(gates)*100:.1f}%)")
    for b_idx, b_name in enumerate(['CNN', 'Pointwise', 'Attention']):
        w_b = gates[:, b_idx]
        print(f"  {b_name:10s}: Mean={np.mean(w_b):.4f}, Std={np.std(w_b):.4f}, Min={np.min(w_b):.4f}, Max={np.max(w_b):.4f}")

    # Load historical models
    hist_json = os.path.join(project_root, 'results', 'hybrid', 'figure11_video_cs', 'figure11_video_cs_comparison.json')
    with open(hist_json, 'r') as f:
        hist_data = json.load(f)

    print("\n" + "=" * 80)
    print("OFFICIAL BENCHMARK COMPARISON TABLE (FIGURE 11 VIDEO CS):")
    print("=" * 80)
    print(f"{'Model Architecture':<22} | {'PSNR (dB)':<10} | {'SSIM':<8} | {'RMSE':<9} | {'Params':<10} | {'Delta vs Base':<12}")
    print("-" * 80)
    base_psnr = hist_data['models']['baseline']['metrics']['psnr']
    for name in ['baseline', 'cnn_control', 'pointwise_control', 'hybrid', 'transformer']:
        m = hist_data['models'][name]['metrics']
        p = hist_data['models'][name]['params']
        d = m['psnr'] - base_psnr
        print(f"{name:<22} | {m['psnr']:<10.4f} | {m['ssim']:<8.4f} | {m['rmse']:<9.5f} | {p:<10,d} | {d:+.4f} dB")
    d_ahd_dip = m_dip['psnr'] - base_psnr
    d_ahd_pert = m_pert['psnr'] - base_psnr
    print(f"{'AHD-Net v1 (In-Loop)':<22} | {m_pert['psnr']:<10.4f} | {m_pert['ssim']:<8.4f} | {m_pert['rmse']:<9.5f} | {total_params:<10,d} | {d_ahd_pert:+.4f} dB")
    print(f"{'AHD-Net v1 (Clean DIP)':<22} | {m_dip['psnr']:<10.4f} | {m_dip['ssim']:<8.4f} | {m_dip['rmse']:<9.5f} | {total_params:<10,d} | {d_ahd_dip:+.4f} dB")
    print("=" * 80)


if __name__ == '__main__':
    main()

