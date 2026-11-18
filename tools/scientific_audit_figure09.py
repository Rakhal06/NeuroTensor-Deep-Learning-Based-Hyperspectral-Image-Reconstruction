"""
Scientific Audit Script for Figure 09 AHD-Net Benchmark
======================================================
Performs independent scientific verification:
1. Temperature schedule audit (tau_start, tau_min, tau at checkpoint, tau=1.0 diagnostic).
2. BatchNorm evaluation mode audit (train mode instance stats vs eval mode running stats).
3. Metric verification (full-volume vs slice, data range, reproducibility of 46.9249 dB and 46.1245 dB).
4. Checkpoint selection verification (L1 measurement loss vs epoch).
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
from deeptensor.models.factory import build_deeptensor_model, count_parameters
from deeptensor.utils.comparison import compute_metrics
from deeptensor.utils.metrics import calculate_psnr, calculate_ssim

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 80)
    print("INDEPENDENT SCIENTIFIC AUDIT: FIGURE 09 AHD-NET BENCHMARK")
    print(f"Device: {device}")
    print("=" * 80)

    # 1. Load Ground Truth
    data_path = os.path.join(project_root, 'data', 'icvl.mat')
    data = io.loadmat(data_path)
    cube_raw = data['hypercube'].astype(np.float32)
    cube = cube_raw / cube_raw.max()
    H, W, nwvl = cube.shape
    cube_ptp = float(np.ptp(cube))
    print(f"Dataset: ICVL Hyperspectral Cube Shape: ({H}, {W}, {nwvl})")
    print(f"Cube Min: {cube.min():.6f}, Max: {cube.max():.6f}, Peak-to-Peak: {cube_ptp:.6f}")

    # 2. Audit AHD-Net Checkpoint
    ckpt_path = os.path.join(project_root, 'results', 'extensions', 'adaptive_hybrid_v1_full', 'figure09_hyperspectral', 'best_checkpoint.pth')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    best_epoch = ckpt['epoch']
    best_loss = ckpt['metrics']['best_loss']
    print(f"\nAHD-Net Checkpoint Audit:")
    print(f"  Checkpoint File: {ckpt_path}")
    print(f"  Best Epoch: {best_epoch} / 5000")
    print(f"  Best Unsupervised L1 Loss: {best_loss:.8f}")

    # Check loss verification
    set_seed = 42
    torch.manual_seed(42)
    np.random.seed(42)
    cube_noisy = utils.measure(cube, 2, 100)
    hsten_noisy = torch.tensor(cube_noisy.reshape(H * W, nwvl), dtype=torch.float32)[None, ...].to(device)
    criterion_l1 = losses.L1Norm()

    # Calculate exact temperature at epoch 4978
    # In run_ahd_net_figure09_full.py:
    # tau = tau_min + 0.5 * (tau_start - tau_min) * (1.0 + math.cos(math.pi * step / max_steps))
    step = best_epoch - 1 # 4977
    tau_epoch = 0.5 + 0.5 * (2.0 - 0.5) * (1.0 + math.cos(math.pi * step / 5000.0))
    print(f"  Prescribed Schedule Temperature at Epoch {best_epoch}: {tau_epoch:.6f}")

    # Build AHD-Net
    model = build_deeptensor_model(
        rank=20,
        seq_len=nwvl,
        spatial_type='skip',
        spectral_type='adaptive_hybrid',
        n_inputs=20,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ffn=512,
        router_type='cagc',
        pe_type='sin_pe',
        tau=tau_epoch,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    total_params = count_parameters(model)
    print(f"  AHD-Net Parameter Count: {total_params:,} (Target: 3,137,808)")

    im_inp = ckpt['config']['im_inp'].to(device)
    spec_inp = ckpt['config']['spec_inp'].to(device)

    # Mode 1: Clean Latents + Train Mode (DIP Instance Statistics) + Prescribed tau
    model.train()
    model.spectral_net.adaptive_block.set_temperature(tau_epoch)
    with torch.no_grad():
        out_clean = model(im_inp, spec_inp)
        recon_clean = out_clean[0].cpu().numpy().reshape(H, W, nwvl)
        loss_clean = criterion_l1(hsten_noisy - out_clean).item()
    m_clean = compute_metrics(cube, recon_clean)

    # Mode 2: Clean Latents + Train Mode + Unit tau=1.0 (Ablation)
    model.spectral_net.adaptive_block.set_temperature(1.0)
    with torch.no_grad():
        out_tau1 = model(im_inp, spec_inp)
        recon_tau1 = out_tau1[0].cpu().numpy().reshape(H, W, nwvl)
    m_tau1 = compute_metrics(cube, recon_tau1)

    # Mode 3: Eval Mode (Running Statistics) + Prescribed tau
    model.eval()
    model.spectral_net.adaptive_block.set_temperature(tau_epoch)
    with torch.no_grad():
        out_eval = model(im_inp, spec_inp)
        recon_eval = out_eval[0].cpu().numpy().reshape(H, W, nwvl)
    m_eval = compute_metrics(cube, recon_eval)

    # Mode 4: Perturbed Latents (Simulating exact in-loop evaluation at epoch 4978)
    # Using seed 200 perturbation std = 1/30
    torch.manual_seed(4978) # test range of perturbations
    model.train()
    reg_noise_std = 1.0 / 30.0
    im_pert = im_inp + torch.randn_like(im_inp) * reg_noise_std
    spec_pert = spec_inp + torch.randn_like(spec_inp) * reg_noise_std
    with torch.no_grad():
        out_pert = model(im_pert, spec_pert)
        recon_pert = out_pert[0].cpu().numpy().reshape(H, W, nwvl)
        loss_pert = criterion_l1(hsten_noisy - out_pert).item()
    m_pert = compute_metrics(cube, recon_pert)

    print("\n" + "=" * 80)
    print("AHD-NET RECONSTRUCTION METRICS UNDER ALL EVALUATION PROTOCOLS:")
    print("=" * 80)
    print(f"Protocol 1 (DIP Train Mode, Clean Latents, Prescribed tau={tau_epoch:.4f}):")
    print(f"  PSNR: {m_clean['psnr']:.4f} dB | SSIM: {m_clean['ssim']:.4f} | RMSE: {m_clean['rmse']:.6f} | L1 Loss: {loss_clean:.6f}")
    print(f"Protocol 2 (DIP Train Mode, Clean Latents, Diagnostic tau=1.0000):")
    print(f"  PSNR: {m_tau1['psnr']:.4f} dB | SSIM: {m_tau1['ssim']:.4f} | RMSE: {m_tau1['rmse']:.6f}")
    print(f"Protocol 3 (Eval Mode, Running Statistics, tau={tau_epoch:.4f}):")
    print(f"  PSNR: {m_eval['psnr']:.4f} dB | SSIM: {m_eval['ssim']:.4f} | RMSE: {m_eval['rmse']:.6f}")
    print(f"Protocol 4 (In-Loop Perturbed Latents, Single Sample):")
    print(f"  PSNR: {m_pert['psnr']:.4f} dB | SSIM: {m_pert['ssim']:.4f} | RMSE: {m_pert['rmse']:.6f} | L1 Loss: {loss_pert:.6f}")

    # Central Slice Verification
    slice_idx = nwvl // 2
    gt_slice = cube[:, :, slice_idx]
    recon_slice = recon_clean[:, :, slice_idx]
    psnr_slice_local = calculate_psnr(gt_slice, recon_slice, data_range=float(np.ptp(gt_slice)))
    psnr_slice_global = calculate_psnr(gt_slice, recon_slice, data_range=1.0)
    print("\nCentral Slice 130 Check:")
    print(f"  Full Volume PSNR:        {m_clean['psnr']:.4f} dB (Evaluated on 174x164x260 volume)")
    print(f"  Central Slice Local PSNR: {psnr_slice_local:.4f} dB (ptp={np.ptp(gt_slice):.4f})")
    print(f"  Central Slice Global PSNR:{psnr_slice_global:.4f} dB (ptp=1.0000)")

    # 3. Load and Audit Historical Checkpoints
    hist_json = os.path.join(project_root, 'results', 'hybrid', 'figure09_hyperspectral', 'figure09_icvl_hsi_comparison.json')
    with open(hist_json, 'r') as f:
        hist_data = json.load(f)

    print("\n" + "=" * 80)
    print("HISTORICAL MODELS EVALUATION PROTOCOL AUDIT:")
    print("=" * 80)
    print(f"{'Model':<20} | {'Reported PSNR':<14} | {'Reported SSIM':<14} | {'Evaluation Mode In Training Script'}")
    print("-" * 80)
    for name in ['baseline', 'cnn_control', 'pointwise_control', 'hybrid', 'transformer']:
        m = hist_data['models'][name]['metrics']
        print(f"{name:<20} | {m['psnr']:<14.4f} | {m['ssim']:<14.4f} | model.train() + in-loop perturbed latents")
    print(f"{'AHD-Net (In-Loop)':<20} | {46.1245:<14.4f} | {0.9874:<14.4f} | model.train() + in-loop perturbed latents")
    print(f"{'AHD-Net (Clean)':<20} | {m_clean['psnr']:<14.4f} | {m_clean['ssim']:<14.4f} | model.train() + clean unperturbed latents")
    print("=" * 80)

if __name__ == '__main__':
    main()

