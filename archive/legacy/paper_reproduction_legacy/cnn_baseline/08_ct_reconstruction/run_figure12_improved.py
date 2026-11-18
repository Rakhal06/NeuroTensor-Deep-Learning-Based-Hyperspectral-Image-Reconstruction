#!/usr/bin/env python
'''
Figure 12 Improved Rerun: 3D PET CT Volume Reconstruction from Limited, Noisy Tomography
Paper: "DeepTensor: Low-Rank Tensor Decomposition with Deep Network Priors" (Section 4.1, Page 8-9)

Target Paper Element: Figure 12 & Section 4.1
- Volume: data/pet.mat (128 x 128 x 56)
- Measurements: Radon transform along 40 angles (0 to 180 degrees)
- Decomposition: Rank-1000 PARAFAC tensor (3x 1D DIP networks with positional encoding)
- Measurement Noise: Poisson (tau=100, paper-aligned) + Gaussian readout (snr=2)
- Training Budget: 4,000 epochs (extended from 2,000 for asymptotic convergence)
- Checkpoint Selection: Strictly by minimum measurement loss (Rule 9 / GT isolation)
- Baselines:
  1. Algorithmic Inversion (Linear gradient descent, no prior)
  2. 2D Total Variation (TV) Regularization
  3. DeepTensor (Rank-1000 PARAFAC with deep network priors)
'''

import os
import sys
import copy
import json
import time
import argparse
import numpy as np
from scipy import io
from skimage.metrics import structural_similarity as ssim_func
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import tensorly
tensorly.set_backend('pytorch')

# Add modules and root paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
modules_path = os.path.join(project_root, 'modules')
if modules_path not in sys.path:
    sys.path.insert(0, modules_path)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import models, utils, losses, deep_prior, lin_inverse

def compute_metrics(gt, rec):
    data_range = float(gt.max() - gt.min())
    psnr_val = float(utils.psnr(gt, rec))
    ssim_val = float(ssim_func(gt, rec, channel_axis=-1, data_range=data_range))
    rmse_val = float(np.sqrt(np.mean((gt - rec) ** 2)))
    return psnr_val, ssim_val, rmse_val

def main():
    parser = argparse.ArgumentParser(description="Figure 12: 3D PET CT Reconstruction (Improved Rerun)")
    parser.add_argument('--dataset', type=str, default='data/pet.mat')
    parser.add_argument('--epochs', type=int, default=4000, help="Training epochs for DeepTensor (paper budget)")
    parser.add_argument('--nangles', type=int, default=40, help="Number of tomographic angles")
    parser.add_argument('--rank', type=int, default=1000, help="PARAFAC rank")
    parser.add_argument('--tau', type=float, default=100.0, help="Poisson shot noise parameter (paper-aligned tau=100)")
    parser.add_argument('--noise_snr', type=float, default=2.0, help="Gaussian readout noise SNR")
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str,
                        default='paper_reproduction/cnn_baseline/08_ct_reconstruction/improved_run')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.output_dir, exist_ok=True)
    images_dir = os.path.join(args.output_dir, 'images')
    plots_dir = os.path.join(args.output_dir, 'plots')
    ckpt_dir = os.path.join(args.output_dir, 'checkpoint')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    print("=" * 80)
    print("FIGURE 12 IMPROVED RERUN: 3D PET CT RECONSTRUCTION FROM NOISY TOMOGRAPHY")
    print("=" * 80, flush=True)
    print(f"Device: {device} | Output Directory: {args.output_dir}")

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 1. Load Data
    print(f"\n[1/4] Loading 3D PET Volume from {args.dataset}...", flush=True)
    data = io.loadmat(args.dataset)
    cube_raw = data['hypercube'].astype(np.float32)
    cube = cube_raw / cube_raw.max()
    H, W, T = cube.shape
    print(f"  3D PET Volume Dimensions: {H} x {W} x {T} slices")

    # 2. Simulate Noisy Radon Tomography Measurements
    print(f"\n[2/4] Simulating Tomographic Measurements across {args.nangles} angles...", flush=True)
    cube_ten = torch.tensor(cube, dtype=torch.float32, device=device).permute(2, 0, 1)[None, ...]
    angles = torch.tensor(np.linspace(0, 180, args.nangles).astype(np.float32), device=device)

    clean_measurements = lin_inverse.radon(cube_ten, angles).detach().cpu().numpy()
    noisy_measurements = utils.measure(clean_measurements, noise_snr=args.noise_snr, tau=args.tau)
    meas_ten = torch.tensor(noisy_measurements, dtype=torch.float32, device=device)
    meas_psnr = float(utils.psnr(clean_measurements, noisy_measurements))
    print(f"  Sinogram shape: {clean_measurements.shape} | Sinogram Noise PSNR: {meas_psnr:.2f} dB")

    # 3. Baselines: Algorithmic Inversion, 2D TV, and 2D TV + PARAFAC
    print("\n[3/4] Running Classical Baselines (Algorithmic, 2D TV, 2D TV + PARAFAC)...", flush=True)
    crit_l2 = losses.L2Norm()
    crit_tv = losses.TVNorm(mode='l1')

    # A. Algorithmic Inversion (Linear gradient descent, no prior)
    t0_alg = time.time()
    rec_alg = utils.get_inp([H, W, T], device=device)
    opt_alg = torch.optim.Adam([rec_alg], lr=0.1)
    for _ in range(200):
        m_est = lin_inverse.radon(rec_alg[None, ...].permute(0, 3, 1, 2), angles)
        l = crit_l2(meas_ten - m_est)
        opt_alg.zero_grad()
        l.backward()
        opt_alg.step()
    alg_time = time.time() - t0_alg
    cube_alg = np.clip(rec_alg.detach().cpu().numpy(), 0.0, 1.0)
    alg_psnr, alg_ssim, alg_rmse = compute_metrics(cube, cube_alg)
    print(f"  Algorithmic Rec:   PSNR = {alg_psnr:.2f} dB | SSIM = {alg_ssim:.4f} | RMSE = {alg_rmse:.4f} ({alg_time:.1f}s)")

    # B. 2D Total Variation (TV) Baseline (run_figure12_TV.py formulation, lambda_tv=1.0)
    t0_tv = time.time()
    rec_tv = utils.get_inp([H, W, T], device=device)
    opt_tv = torch.optim.Adam([rec_tv], lr=0.1)
    for _ in range(250):
        m_est = lin_inverse.radon(rec_tv[None, ...].permute(0, 3, 1, 2), angles)
        l = crit_l2(meas_ten - m_est) + 1.0 * crit_tv(rec_tv.permute(2, 0, 1)[None, ...])
        opt_tv.zero_grad()
        l.backward()
        opt_tv.step()
    tv_time = time.time() - t0_tv
    cube_tv = np.clip(rec_tv.detach().cpu().numpy(), 0.0, 1.0)
    tv_psnr, tv_ssim, tv_rmse = compute_metrics(cube, cube_tv)
    print(f"  2D TV Rec:         PSNR = {tv_psnr:.2f} dB | SSIM = {tv_ssim:.4f} | RMSE = {tv_rmse:.4f} ({tv_time:.1f}s)")

    # C. 2D TV + PARAFAC Baseline (Rank-1000 CP with TV penalty, no deep prior)
    t0_tv_cp = time.time()
    U_cp = utils.get_inp([H, args.rank], device=device)
    V_cp = utils.get_inp([W, args.rank], device=device)
    W_cp = utils.get_inp([T, args.rank], device=device)
    core_cp = utils.get_inp(args.rank, device=device)
    with torch.no_grad():
        core_cp[...] = 1.0 / args.rank
    opt_tv_cp = torch.optim.Adam([U_cp, V_cp, W_cp, core_cp], lr=0.05)
    for _ in range(250):
        t_est = tensorly.cp_to_tensor((core_cp, [U_cp, V_cp, W_cp]))
        m_est = lin_inverse.radon(t_est[None, ...].permute(0, 3, 1, 2), angles)
        l = crit_l2(meas_ten - m_est) + 1.0 * crit_tv(t_est.permute(2, 0, 1)[None, ...])
        opt_tv_cp.zero_grad()
        l.backward()
        opt_tv_cp.step()
    tv_cp_time = time.time() - t0_tv_cp
    cube_tv_cp = np.clip(tensorly.cp_to_tensor((core_cp, [U_cp, V_cp, W_cp])).detach().cpu().numpy(), 0.0, 1.0)
    tv_cp_psnr, tv_cp_ssim, tv_cp_rmse = compute_metrics(cube, cube_tv_cp)
    print(f"  2D TV + PARAFAC:   PSNR = {tv_cp_psnr:.2f} dB | SSIM = {tv_cp_ssim:.4f} | RMSE = {tv_cp_rmse:.4f} ({tv_cp_time:.1f}s)")

    # 4. DeepTensor Rank-1000 PARAFAC Reconstruction
    print(f"\n[4/4] Training Rank-{args.rank} DeepTensor PARAFAC ({args.epochs} Epochs on CUDA)...", flush=True)
    u_inp = utils.get_1d_posencode_inp(H, args.rank // 2, device=device)
    v_inp = utils.get_1d_posencode_inp(W, args.rank // 2, device=device)
    w_inp = utils.get_1d_posencode_inp(T, args.rank // 2, device=device)

    u_net = deep_prior.get_net(args.rank, 'skip1d', 'reflection', 'linear',
                               n_channels=args.rank, skip_n33d=128, skip_n33u=128,
                               num_scales=5).to(device)
    v_net = deep_prior.get_net(args.rank, 'skip1d', 'reflection', 'linear',
                               n_channels=args.rank, skip_n33d=128, skip_n33u=128,
                               num_scales=5).to(device)
    w_net = deep_prior.get_net(args.rank, 'skip1d', 'reflection', 'linear',
                               n_channels=args.rank, skip_n33d=128, skip_n33u=128,
                               num_scales=5).to(device)

    u_net.train(); v_net.train(); w_net.train()

    core = utils.get_inp(args.rank, device=device)
    with torch.no_grad():
        core[...] = 1.0 / args.rank

    params = list(u_net.parameters()) + list(v_net.parameters()) + list(w_net.parameters()) + [u_inp, v_inp, w_inp, core]
    optimizer = torch.optim.Adam(lr=args.learning_rate, params=params)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-4)
    criterion_loss = losses.L2Norm()

    reg_noise_std = 1.0 / 30.0
    u_inp_per = u_inp.detach().clone()
    v_inp_per = v_inp.detach().clone()
    w_inp_per = w_inp.detach().clone()

    best_loss = float('inf')
    best_cube_estim = None
    best_epoch = 0
    loss_history = []
    psnr_progression = []

    progression_csv = os.path.join(args.output_dir, 'progression_deeptensor.csv')
    with open(progression_csv, 'w') as f:
        f.write("epoch,measurement_loss,psnr_db,ssim,best_measurement_loss,lr\n")

    t0_dt = time.time()
    for epoch in range(1, args.epochs + 1):
        u_inp_pert = u_inp + u_inp_per.normal_() * reg_noise_std
        v_inp_pert = v_inp + v_inp_per.normal_() * reg_noise_std
        w_inp_pert = w_inp + w_inp_per.normal_() * reg_noise_std

        Uk = u_net(u_inp_pert)
        Vk = v_net(v_inp_pert)
        Wk = w_net(w_inp_pert)

        factors = [Uk[0].permute(1, 0), Vk[0].permute(1, 0), Wk[0].permute(1, 0)]
        tensor_estim = tensorly.cp_to_tensor((core, factors))

        measurements_estim = lin_inverse.radon(tensor_estim[None, ...].permute(0, 3, 1, 2), angles)
        loss = criterion_loss(meas_ten - measurements_estim)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        cur_loss = loss.item()
        loss_history.append(cur_loss)

        # Strictly select by measurement loss (Rule 9 / GT isolation)
        if cur_loss < best_loss:
            best_loss = cur_loss
            best_cube_estim = tensor_estim.detach().clone()
            best_epoch = epoch

        if epoch % 100 == 0 or epoch == 1 or epoch == args.epochs:
            with torch.no_grad():
                cur_rec = np.clip(tensor_estim.detach().cpu().numpy(), 0.0, 1.0)
                cur_p, cur_s, _ = compute_metrics(cube, cur_rec)
                cur_lr = optimizer.param_groups[0]['lr']
                psnr_progression.append((epoch, cur_loss, cur_p, cur_s))
                with open(progression_csv, 'a') as f:
                    f.write(f"{epoch},{cur_loss:.6e},{cur_p:.4f},{cur_s:.4f},{best_loss:.6e},{cur_lr:.6e}\n")
                if epoch % 500 == 0 or epoch == args.epochs:
                    print(f"  Epoch [{epoch:4d}/{args.epochs:4d}] | Loss: {cur_loss:.5e} | Best Loss: {best_loss:.5e} | PSNR: {cur_p:.2f} dB | SSIM: {cur_s:.4f}", flush=True)

    dt_time = time.time() - t0_dt
    cube_dt = np.clip(best_cube_estim.squeeze().cpu().numpy(), 0.0, 1.0)
    dt_psnr, dt_ssim, dt_rmse = compute_metrics(cube, cube_dt)
    peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if torch.cuda.is_available() else 0.0

    print("\n" + "=" * 80)
    print("FIGURE 12 FINAL IMPROVEMENT RESULTS:")
    print("=" * 80)
    print(f"Algorithmic Rec:        PSNR = {alg_psnr:6.2f} dB | SSIM = {alg_ssim:.4f} | RMSE = {alg_rmse:.4f} | Time = {alg_time:.1f}s")
    print(f"2D TV Regularization:   PSNR = {tv_psnr:6.2f} dB | SSIM = {tv_ssim:.4f} | RMSE = {tv_rmse:.4f} | Time = {tv_time:.1f}s")
    print(f"2D TV + PARAFAC:        PSNR = {tv_cp_psnr:6.2f} dB | SSIM = {tv_cp_ssim:.4f} | RMSE = {tv_cp_rmse:.4f} | Time = {tv_cp_time:.1f}s")
    print(f"DeepTensor (Rank-1000): PSNR = {dt_psnr:6.2f} dB | SSIM = {dt_ssim:.4f} | RMSE = {dt_rmse:.4f} | Time = {dt_time:.1f}s")
    print("=" * 80)
    print(f"DeepTensor PSNR Gain over Algorithmic: {dt_psnr - alg_psnr:+.2f} dB")
    print(f"DeepTensor PSNR Gain over 2D TV:       {dt_psnr - tv_psnr:+.2f} dB")
    print(f"DeepTensor PSNR Gain over 2D TV+CP:    {dt_psnr - tv_cp_psnr:+.2f} dB")
    print(f"DeepTensor SSIM Gain over 2D TV:       {dt_ssim - tv_ssim:+.4f}")
    print(f"DeepTensor SSIM Gain over 2D TV+CP:    {dt_ssim - tv_cp_ssim:+.4f}")
    print(f"Peak VRAM: {peak_vram:.2f} MB | Best Checkpoint Epoch: {best_epoch}")

    # 5. Save Checkpoint
    checkpoint_path = os.path.join(ckpt_dir, 'figure12_deeptensor_improved_checkpoint.pt')
    torch.save({
        'u_net': u_net.state_dict(),
        'v_net': v_net.state_dict(),
        'w_net': w_net.state_dict(),
        'core': core.detach().cpu(),
        'rank': args.rank,
        'best_epoch': best_epoch,
        'best_loss': best_loss,
        'metrics': {
            'PSNR': dt_psnr,
            'SSIM': dt_ssim,
            'RMSE': dt_rmse
        },
        'epochs': args.epochs
    }, checkpoint_path)
    print(f"Checkpoint saved to: {checkpoint_path}")

    # 6. Save Configuration & Metrics JSON
    metrics_data = {
        'experiment': 'Figure 12: 3D PET CT Reconstruction (Improved Rerun)',
        'dataset': args.dataset,
        'volume_dimensions': [H, W, T],
        'decomposition_rank': args.rank,
        'tomographic_angles': args.nangles,
        'noise_configuration': {
            'noise_snr': args.noise_snr,
            'tau': args.tau,
            'model': 'Poisson (tau=100) + Gaussian readout (snr=2)'
        },
        'optimizer': 'Adam + CosineAnnealingLR',
        'learning_rate': args.learning_rate,
        'epochs': args.epochs,
        'device': str(device),
        'best_epoch': best_epoch,
        'runtime_seconds': dt_time,
        'peak_vram_mb': peak_vram,
        'published_paper_reference': {
            'deeptensor_psnr': 31.1,
            'deeptensor_ssim': 0.87,
            'bacca_psnr': 30.5,
            'bacca_ssim': 0.84,
            'fbpconvnet_psnr': 29.1,
            'fbpconvnet_ssim': 0.65,
            'gong_psnr': 28.6,
            'gong_ssim': 0.69,
            'tv_2d_psnr': 28.6,
            'tv_2d_ssim': 0.71,
            'tv_2d_parafac_psnr': 27.6,
            'tv_2d_parafac_ssim': 0.59,
            'algorithmic_psnr': 26.3,
            'algorithmic_ssim': 0.48
        },
        'metrics': {
            'algorithmic': {'PSNR': alg_psnr, 'SSIM': alg_ssim, 'RMSE': alg_rmse, 'time': alg_time},
            'tv_2d': {'PSNR': tv_psnr, 'SSIM': tv_ssim, 'RMSE': tv_rmse, 'time': tv_time},
            'tv_2d_parafac': {'PSNR': tv_cp_psnr, 'SSIM': tv_cp_ssim, 'RMSE': tv_cp_rmse, 'time': tv_cp_time},
            'deeptensor_improved': {'PSNR': dt_psnr, 'SSIM': dt_ssim, 'RMSE': dt_rmse, 'time': dt_time}
        }
    }
    json_path = os.path.join(args.output_dir, 'figure12_improved_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(metrics_data, f, indent=4)

    # 7. Save Metrics Summary Text
    summary_path = os.path.join(args.output_dir, 'metrics.txt')
    with open(summary_path, 'w') as f:
        f.write('============================================================\n')
        f.write('DEEPTENSOR BASELINE: FIGURE 12 IMPROVED METRICS SUMMARY\n')
        f.write('============================================================\n')
        f.write('Experiment: Figure 12 (3D PET CT Reconstruction)\n')
        f.write(f'Dataset: PET CT Scan ({args.dataset}) | Dimensions: {H} x {W} x {T}\n')
        f.write(f'PARAFAC Rank: {args.rank} | Tomographic Angles: {args.nangles}\n')
        f.write(f'Noise: Poisson (tau={args.tau}) + Gaussian readout (snr={args.noise_snr})\n')
        f.write(f'Epochs: {args.epochs} | Optimizer: Adam (lr={args.learning_rate}) + CosineAnnealing\n')
        f.write(f'Device: {device} | Peak VRAM: {peak_vram:.2f} MB\n\n')
        f.write('METHOD                     PSNR (dB)     SSIM         RMSE         RUNTIME (s)\n')
        f.write('--------------------------------------------------------------------------------\n')
        f.write(f'Algorithmic Rec            {alg_psnr:<13.2f} {alg_ssim:<12.4f} {alg_rmse:<12.4f} {alg_time:.2f}\n')
        f.write(f'2D TV Regularization       {tv_psnr:<13.2f} {tv_ssim:<12.4f} {tv_rmse:<12.4f} {tv_time:.2f}\n')
        f.write(f'2D TV + PARAFAC            {tv_cp_psnr:<13.2f} {tv_cp_ssim:<12.4f} {tv_cp_rmse:<12.4f} {tv_cp_time:.2f}\n')
        f.write(f'DeepTensor (Rank-1000)     {dt_psnr:<13.2f} {dt_ssim:<12.4f} {dt_rmse:<12.4f} {dt_time:.2f}\n')
        f.write('--------------------------------------------------------------------------------\n')
        f.write('Paper Reference (DeepTensor): 31.10 dB     0.8700\n')
        f.write('Paper Reference (2D TV):      28.60 dB     0.7100\n')
        f.write('Paper Reference (2D TV+CP):   27.60 dB     0.5900\n')
        f.write('Paper Reference (Algorithmic):26.30 dB     0.4800\n\n')
        f.write(f'DeepTensor PSNR Gain over Algorithmic: {dt_psnr - alg_psnr:+.2f} dB\n')
        f.write(f'DeepTensor PSNR Gain over 2D TV:       {dt_psnr - tv_psnr:+.2f} dB\n')
        f.write(f'DeepTensor PSNR Gain over 2D TV+CP:    {dt_psnr - tv_cp_psnr:+.2f} dB\n')
        f.write(f'DeepTensor SSIM Gain over 2D TV:       {dt_ssim - tv_ssim:+.4f}\n')
        f.write(f'DeepTensor SSIM Gain over 2D TV+CP:    {dt_ssim - tv_cp_ssim:+.4f}\n')
        f.write(f'Final Measurement Loss:                {best_loss:.6e}\n')

    # 8. Save Visual Reconstruction Panel (Center Slice 28 as in Paper Fig. 12)
    slice_idx = 28
    gt_slice = cube[..., slice_idx]
    alg_slice = cube_alg[..., slice_idx]
    tv_slice = cube_tv[..., slice_idx]
    tv_cp_slice = cube_tv_cp[..., slice_idx]
    dt_slice = cube_dt[..., slice_idx]
    sino_slice = noisy_measurements[slice_idx]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    # Row 1
    axes[0, 0].imshow(sino_slice, cmap='gray', aspect='auto')
    axes[0, 0].set_title(f'CT Meas. Slice {slice_idx}\n({args.nangles} Angles, tau={args.tau})', fontsize=10)
    axes[0, 0].set_xlabel('Sensor Detectors'); axes[0, 0].set_ylabel('Angles')

    axes[0, 1].imshow(gt_slice, cmap='gray')
    axes[0, 1].set_title(f'Ground Truth (Slice {slice_idx})', fontsize=10, fontweight='bold')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(alg_slice, cmap='gray')
    axes[0, 2].set_title(f'Algorithmic Rec (No Prior)\nPSNR: {alg_psnr:.1f} dB | SSIM: {alg_ssim:.2f}', fontsize=10)
    axes[0, 2].axis('off')

    axes[0, 3].imshow(tv_slice, cmap='gray')
    axes[0, 3].set_title(f'2D TV Regularization\nPSNR: {tv_psnr:.1f} dB | SSIM: {tv_ssim:.2f}', fontsize=10)
    axes[0, 3].axis('off')

    # Row 2
    axes[1, 0].imshow(tv_cp_slice, cmap='gray')
    axes[1, 0].set_title(f'2D TV + PARAFAC\nPSNR: {tv_cp_psnr:.1f} dB | SSIM: {tv_cp_ssim:.2f}', fontsize=10)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(dt_slice, cmap='gray')
    axes[1, 1].set_title(f'DeepTensor (Rank-{args.rank})\nPSNR: {dt_psnr:.1f} dB | SSIM: {dt_ssim:.2f}', fontsize=10, fontweight='bold', color='navy')
    axes[1, 1].axis('off')

    diff_slice = np.abs(gt_slice - dt_slice)
    im_diff = axes[1, 2].imshow(diff_slice, cmap='hot')
    plt.colorbar(im_diff, ax=axes[1, 2], fraction=0.046, pad=0.04)
    axes[1, 2].set_title(f'DeepTensor Abs Error\n(Slice {slice_idx})', fontsize=10)
    axes[1, 2].axis('off')

    # PSNR Progression curve in subpanel [1, 3]
    ep_prog = [p[0] for p in psnr_progression]
    psnr_prog = [p[2] for p in psnr_progression]
    axes[1, 3].plot(ep_prog, psnr_prog, color='navy', linewidth=2.0, label='DeepTensor')
    axes[1, 3].axhline(alg_psnr, color='gray', linestyle=':', label=f'Alg ({alg_psnr:.1f} dB)')
    axes[1, 3].axhline(tv_psnr, color='crimson', linestyle='--', label=f'2D TV ({tv_psnr:.1f} dB)')
    axes[1, 3].axhline(tv_cp_psnr, color='orange', linestyle='-.', label=f'TV+CP ({tv_cp_psnr:.1f} dB)')
    axes[1, 3].axhline(31.1, color='darkgreen', linestyle='-', alpha=0.7, label='Paper DT (31.1 dB)')
    axes[1, 3].set_title('Convergence PSNR vs Epoch', fontsize=10)
    axes[1, 3].set_xlabel('Epoch', fontsize=9); axes[1, 3].set_ylabel('PSNR (dB)', fontsize=9)
    axes[1, 3].legend(fontsize=7, loc='lower right')
    axes[1, 3].grid(True, alpha=0.3)

    plt.suptitle("Figure 12: 3D PET CT Reconstruction from Limited Noisy Tomographic Measurements\n(40 Angles, Poisson tau=100 + Gaussian Shot Noise)", fontsize=13, y=0.98)
    plt.tight_layout()
    panel_path = os.path.join(images_dir, 'figure12_reconstruction_panel.png')
    fig.savefig(panel_path, dpi=250)
    plt.close(fig)
    print(f"Visual panel saved to: {panel_path}")

    # 9. Save Convergence Curves
    plt.figure(figsize=(10, 4.5))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history, color='blue', label='Radon Measurement Loss')
    plt.yscale('log')
    plt.title('Measurement Loss vs Epoch', fontsize=11)
    plt.xlabel('Epoch'); plt.ylabel('L2 Loss'); plt.grid(True, alpha=0.3); plt.legend()

    plt.subplot(1, 2, 2)
    ep_prog = [p[0] for p in psnr_progression]
    psnr_prog = [p[2] for p in psnr_progression]
    plt.plot(ep_prog, psnr_prog, color='navy', linewidth=2.0, label='DeepTensor PSNR')
    plt.axhline(alg_psnr, color='gray', linestyle=':', label=f'Algorithmic ({alg_psnr:.1f} dB)')
    plt.axhline(tv_psnr, color='crimson', linestyle='--', label=f'2D TV ({tv_psnr:.1f} dB)')
    plt.axhline(30.8, color='darkgreen', linestyle='-.', alpha=0.8, label='Paper DeepTensor (30.8 dB)')
    plt.title('Reconstruction PSNR vs Epoch', fontsize=11)
    plt.xlabel('Epoch'); plt.ylabel('PSNR (dB)'); plt.legend(loc='lower right'); plt.grid(True, alpha=0.3)

    plt.tight_layout()
    curve_path = os.path.join(plots_dir, 'figure12_convergence_curves.png')
    plt.savefig(curve_path, dpi=250)
    plt.close()
    print(f"Convergence curves saved to: {curve_path}")

    # 10. Save Results Array Checkpoint
    npz_path = os.path.join(args.output_dir, 'figure12_results.npz')
    np.savez_compressed(
        npz_path,
        cube_dt=cube_dt,
        cube_tv=cube_tv,
        cube_tv_cp=cube_tv_cp,
        cube_alg=cube_alg,
        noisy_measurements=noisy_measurements,
        loss_history=loss_history
    )
    print(f"Figure 12 improved run complete! All artifacts saved to: {args.output_dir}")

if __name__ == '__main__':
    main()

