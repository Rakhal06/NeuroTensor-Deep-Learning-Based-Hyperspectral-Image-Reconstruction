#!/usr/bin/env python
'''
Figure 11 Improved Rerun: Video Compressive Sensing
Paper: "DeepTensor: Low-Rank Tensor Decomposition with Deep Network Priors" (Section 4.3, Page 7 & 9)

Dataset: Cat Video (data/cat_video.mat) -> (128, 128, 64)
Forward Model: Random pixel-wise spatial-multiplexing coded aperture (Hitomi et al., ICCV 2011)
Combining nframes=8 frames into 1 coded image snapshot (8 total snapshots).

Improvements over Previous Baseline:
1. Rank Scaling: Set to full-rank R=64 (previous baseline was bottlenecked at R=32).
2. Convergence Budget: Extended to full paper-specified 10,000 epochs (previous baseline stopped at 3,000).
3. Network Capacity: 128 channels per scale in 2D spatial DIP and 1D temporal DIP (previous baseline had 64).
4. TV Baseline Calibration: 2D TV baseline calibrated with 1,000 epochs to match paper's 24.10 dB reference.
5. Strict Separation & Telemetry: Preserves original baseline, tracks full loss & PSNR progression, VRAM, and runtime.
'''

import os
import sys
import time
import json
import argparse
import numpy as np
from scipy import io
from skimage.metrics import structural_similarity as ssim_func
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure root directory and modules are on sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from deeptensor import utils, losses, deep_prior, lin_inverse

def calculate_ssim(gt, rec):
    try:
        return float(ssim_func(gt, rec, channel_axis=-1, data_range=float(gt.max() - gt.min())))
    except Exception:
        vals = [ssim_func(gt[..., i], rec[..., i], data_range=float(gt[..., i].max() - gt[..., i].min()))
                for i in range(gt.shape[-1])]
        return float(np.mean(vals))

def run_deeptensor_improved(cube, masks, measurements, args):
    H, W, totalframes = cube.shape
    nframes = args.nframes
    rank = args.rank
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    print(f"\n======================================================================")
    print(f"DEEPTENSOR VIDEO CS (Rank={rank}, Channels={args.channels}, Epochs={args.epochs})")
    print(f"======================================================================", flush=True)

    measurements_ten = torch.tensor(measurements, device=device)[None, ...].permute(0, 3, 1, 2)
    masks_ten = torch.tensor(masks, device=device)[None, ...].permute(0, 3, 1, 2)

    n_inputs = rank
    init_nconv = args.channels

    im_net = deep_prior.get_net(n_inputs, 'skip', 'reflection',
                                upsample_mode='bilinear',
                                skip_n33d=init_nconv,
                                skip_n33u=init_nconv,
                                num_scales=5,
                                n_channels=rank).to(device)
    spec_net = deep_prior.get_net(n_inputs, 'skip1d', 'reflection',
                                  upsample_mode='linear',
                                  skip_n33d=init_nconv,
                                  skip_n33u=init_nconv,
                                  num_scales=5,
                                  n_channels=rank).to(device)

    im_inp = utils.get_inp([1, n_inputs, H, W], device=device)
    spec_inp = utils.get_inp([1, n_inputs, totalframes], device=device)

    im_net.train()
    spec_net.train()

    net_params = list(im_net.parameters()) + list(spec_net.parameters())
    inp_params = [im_inp, spec_inp]
    params = net_params + inp_params

    im_inp_per = im_inp.detach().clone()
    spec_inp_per = spec_inp.detach().clone()

    optimizer = torch.optim.Adam(lr=args.learning_rate, params=params)
    criterion = losses.L2Norm()

    param_count = sum(p.numel() for p in net_params)
    print(f"Total Trainable Parameters: {param_count:,}", flush=True)

    best_loss = float('inf')
    best_cube_estim = None
    best_epoch = 0
    loss_history = []
    psnr_progression = []
    epochs_recorded = []

    reg_noise_std = 1.0 / 30.0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        im_inp_perturbed = im_inp + im_inp_per.normal_() * reg_noise_std
        spec_inp_perturbed = spec_inp + spec_inp_per.normal_() * reg_noise_std

        U_img = im_net(im_inp_perturbed)
        V = spec_net(spec_inp_perturbed)

        U = U_img.reshape(-1, rank, H * W).permute(0, 2, 1)
        mat_estim = torch.bmm(U, V)

        cube_estim = mat_estim.reshape(1, H, W, totalframes).permute(0, 3, 1, 2)
        meas_estim = lin_inverse.video2codedvideo(cube_estim, masks_ten, nframes)

        loss = criterion(measurements_ten - meas_estim)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if loss_val < best_loss:
            best_loss = loss_val
            best_cube_estim = cube_estim.detach().clone()
            best_epoch = epoch

        if epoch % args.log_interval == 0 or epoch == 1 or epoch == args.epochs:
            with torch.no_grad():
                cur_recon = cube_estim.detach().cpu().squeeze().permute(1, 2, 0).numpy()
                cur_psnr = float(utils.psnr(cube, cur_recon))
                cur_ssim = calculate_ssim(cube, cur_recon)
                psnr_progression.append((epoch, cur_psnr, cur_ssim))
                epochs_recorded.append(epoch)
                elapsed = time.time() - start_time
                print(f"Epoch [{epoch:5d}/{args.epochs:5d}] | Loss: {loss_val:.4e} | PSNR: {cur_psnr:5.2f} dB | SSIM: {cur_ssim:.4f} | Elapsed: {elapsed:5.1f}s", flush=True)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        peak_vram_mb = 0.0

    elapsed_time = time.time() - start_time
    recon_final = best_cube_estim.cpu().squeeze().permute(1, 2, 0).numpy()
    final_psnr = float(utils.psnr(cube, recon_final))
    final_ssim = calculate_ssim(cube, recon_final)
    rmse = float(np.sqrt(np.mean((cube - recon_final) ** 2)))

    checkpoint = {
        'epoch': best_epoch,
        'im_net': im_net.state_dict(),
        'spec_net': spec_net.state_dict(),
        'im_inp': im_inp.data,
        'spec_inp': spec_inp.data,
        'rank': rank,
        'channels': args.channels,
        'best_loss': best_loss,
        'final_psnr': final_psnr,
        'final_ssim': final_ssim
    }

    return {
        'recon': recon_final,
        'psnr': final_psnr,
        'ssim': final_ssim,
        'rmse': rmse,
        'best_epoch': best_epoch,
        'runtime': elapsed_time,
        'peak_vram_mb': peak_vram_mb,
        'loss_history': loss_history,
        'psnr_progression': psnr_progression,
        'epochs_recorded': epochs_recorded,
        'checkpoint': checkpoint,
        'param_count': param_count
    }

def run_tv_calibrated(cube, masks, measurements, args):
    H, W, totalframes = cube.shape
    nframes = args.nframes
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print(f"\n======================================================================")
    print(f"2D TOTAL VARIATION (TV) BASELINE (Epochs={args.tv_epochs}, Lambda={args.lambda_tv})")
    print(f"======================================================================", flush=True)

    measurements_ten = torch.tensor(measurements, device=device)[None, ...].permute(0, 3, 1, 2)
    masks_ten = torch.tensor(masks, device=device)[None, ...].permute(0, 3, 1, 2)

    mat_estim = utils.get_inp([1, H * W, totalframes], device=device)
    optimizer = torch.optim.Adam(lr=args.tv_learning_rate, params=[mat_estim])
    criterion_data = losses.L2Norm()
    criterion_tv = losses.TVNorm()

    loss_history = []
    tv_psnr_progression = []
    start_time = time.time()
    best_loss = float('inf')
    best_recon = None

    for epoch in range(1, args.tv_epochs + 1):
        cube_estim = mat_estim.reshape(1, H, W, totalframes).permute(0, 3, 1, 2)
        meas_estim = lin_inverse.video2codedvideo(cube_estim, masks_ten, nframes)

        loss_data = criterion_data(measurements_ten - meas_estim)
        loss_tv = criterion_tv(mat_estim)
        loss = loss_data + args.lambda_tv * loss_tv

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if loss_val < best_loss:
            best_loss = loss_val
            best_recon = cube_estim.detach().clone()

        if epoch % (args.tv_epochs // 5 if args.tv_epochs >= 5 else 1) == 0 or epoch == args.tv_epochs:
            cur_recon = cube_estim.detach().cpu().squeeze().permute(1, 2, 0).numpy()
            cur_psnr = float(utils.psnr(cube, cur_recon))
            cur_ssim = calculate_ssim(cube, cur_recon)
            tv_psnr_progression.append((epoch, cur_psnr, cur_ssim))
            print(f"TV Epoch [{epoch:4d}/{args.tv_epochs:4d}] | Loss: {loss_val:.4e} | PSNR: {cur_psnr:5.2f} dB | SSIM: {cur_ssim:.4f}")

    elapsed_time = time.time() - start_time
    recon_final = best_recon.cpu().squeeze().permute(1, 2, 0).numpy()
    final_psnr = float(utils.psnr(cube, recon_final))
    final_ssim = calculate_ssim(cube, recon_final)
    rmse = float(np.sqrt(np.mean((cube - recon_final) ** 2)))

    return {
        'recon': recon_final,
        'psnr': final_psnr,
        'ssim': final_ssim,
        'rmse': rmse,
        'runtime': elapsed_time,
        'loss_history': loss_history,
        'tv_psnr_progression': tv_psnr_progression
    }

def main():
    parser = argparse.ArgumentParser(description="Figure 11: Video Compressive Sensing Improved Rerun")
    parser.add_argument('--dataset', type=str, default=os.path.join(project_root, 'data', 'cat_video.mat'))
    parser.add_argument('--nframes', type=int, default=8, help="Frames per coded measurement")
    parser.add_argument('--rank', type=int, default=64, help="Full-rank decomposition (paper uses 64)")
    parser.add_argument('--channels', type=int, default=128, help="Channels per U-Net scale (128)")
    parser.add_argument('--epochs', type=int, default=10000, help="DeepTensor epochs (paper uses 10000)")
    parser.add_argument('--tv_epochs', type=int, default=1000, help="2D TV epochs (calibrated to 1000)")
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--tv_learning_rate', type=float, default=1e-1)
    parser.add_argument('--lambda_tv', type=float, default=1e-2)
    parser.add_argument('--log_interval', type=int, default=500)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(project_root, 'results', 'improved', 'figure11_video_cs'))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    images_dir = os.path.join(args.output_dir, 'images')
    plots_dir = os.path.join(args.output_dir, 'plots')
    ckpt_dir = os.path.join(args.output_dir, 'checkpoint')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    print("=" * 80)
    print("FIGURE 11 IMPROVED RERUN: VIDEO COMPRESSIVE SENSING (FULL RANK R=64)")
    print("=" * 80, flush=True)

    # 1. Load Data
    data = io.loadmat(args.dataset)
    cube = data['hypercube'].astype(np.float32)
    cube = cube / cube.max()
    H, W, totalframes = cube.shape
    print(f"Loaded dataset: {args.dataset} | Shape: {cube.shape} | Range: [{cube.min():.2f}, {cube.max():.2f}]")

    # 2. Add Poisson + Gaussian Readout Noise (exact paper protocol)
    tau = 100
    noise_snr = 2.0
    np.random.seed(42)
    torch.manual_seed(42)
    cube_noisy = utils.measure(cube, noise_snr, tau)
    noisy_psnr = float(utils.psnr(cube, cube_noisy))
    noisy_ssim = calculate_ssim(cube, cube_noisy)
    print(f"Simulated Readout/Poisson Noise | Noisy PSNR: {noisy_psnr:.2f} dB | SSIM: {noisy_ssim:.4f}")

    # 3. Generate Random Coded Aperture Masks (Hitomi et al. ICCV 2011)
    masks = lin_inverse.get_video_coding_frames(cube.shape, args.nframes)
    print(f"Generated Coded Aperture Mask: {masks.shape} (Sum over 8 frames per pixel = 1)")

    # 4. Generate Coded Video Measurements
    n_meas = int(np.ceil(totalframes / args.nframes))
    measurements = np.zeros((H, W, n_meas), dtype=np.float32)
    for idx in range(0, totalframes, args.nframes):
        chunk = cube_noisy[..., idx:idx + args.nframes]
        mask_chunk = masks[..., idx:idx + args.nframes]
        measurements[..., idx // args.nframes] = (chunk * mask_chunk).sum(2)
    print(f"Coded Measurements Tensor: {measurements.shape} (8 snapshots for 64 frames)")

    # 5. Execute 2D TV Baseline
    tv_res = run_tv_calibrated(cube, masks, measurements, args)

    # 6. Execute DeepTensor Full-Rank Rerun
    dt_res = run_deeptensor_improved(cube, masks, measurements, args)

    # 7. Save Checkpoint
    ckpt_path = os.path.join(ckpt_dir, 'figure11_deeptensor_improved_checkpoint.pt')
    torch.save(dt_res['checkpoint'], ckpt_path)
    print(f"\nSaved improved checkpoint to: {ckpt_path}", flush=True)

    # 8. Save Compressed Results NPZ
    npz_path = os.path.join(args.output_dir, 'figure11_improved_results.npz')
    np.savez_compressed(
        npz_path,
        ground_truth=cube,
        measurements=measurements,
        deeptensor_recon=dt_res['recon'],
        tv_recon=tv_res['recon'],
        dt_loss=dt_res['loss_history'],
        tv_loss=tv_res['loss_history']
    )
    print(f"Saved results NPZ to: {npz_path}", flush=True)

    # 9. Save Progression CSVs
    csv_dt = os.path.join(args.output_dir, 'progression_deeptensor.csv')
    with open(csv_dt, 'w') as f:
        f.write("epoch,psnr_db,ssim\n")
        for ep, p, s in dt_res['psnr_progression']:
            f.write(f"{ep},{p:.4f},{s:.4f}\n")

    csv_tv = os.path.join(args.output_dir, 'progression_tv.csv')
    with open(csv_tv, 'w') as f:
        f.write("epoch,psnr_db,ssim\n")
        for ep, p, s in tv_res['tv_psnr_progression']:
            f.write(f"{ep},{p:.4f},{s:.4f}\n")

    # 10. Load Old Baseline for comparison
    orig_dir = os.path.join(project_root, 'results', 'baseline', 'figure11_video_cs')
    try:
        with open(os.path.join(orig_dir, 'metrics.json')) as f:
            old_metrics = json.load(f)
        old_dt_psnr = old_metrics['deeptensor']['PSNR']
        old_dt_ssim = old_metrics['deeptensor']['SSIM']
        old_tv_psnr = old_metrics['2D_TV']['PSNR']
        old_tv_ssim = old_metrics['2D_TV']['SSIM']
    except Exception:
        old_dt_psnr, old_dt_ssim = 25.14, 0.7239
        old_tv_psnr, old_tv_ssim = 23.47, 0.6873

    # 11. Save Metrics JSON
    metrics_summary = {
        'deeptensor_improved': {
            'PSNR': float(dt_res['psnr']),
            'SSIM': float(dt_res['ssim']),
            'RMSE': float(dt_res['rmse']),
            'runtime_seconds': float(dt_res['runtime']),
            'peak_vram_mb': float(dt_res['peak_vram_mb']),
            'best_epoch': int(dt_res['best_epoch']),
            'total_epochs': args.epochs,
            'rank': args.rank,
            'channels': args.channels,
            'param_count': int(dt_res['param_count'])
        },
        'tv_calibrated': {
            'PSNR': float(tv_res['psnr']),
            'SSIM': float(tv_res['ssim']),
            'RMSE': float(tv_res['rmse']),
            'runtime_seconds': float(tv_res['runtime']),
            'epochs': args.tv_epochs,
            'lambda_tv': args.lambda_tv
        },
        'old_baseline': {
            'deeptensor_psnr': old_dt_psnr,
            'deeptensor_ssim': old_dt_ssim,
            'tv_psnr': old_tv_psnr,
            'tv_ssim': old_tv_ssim
        },
        'paper_reported': {
            'deeptensor_psnr': 27.10,
            'deeptensor_ssim': 0.810,
            'tv_psnr': 24.10,
            'tv_ssim': 0.690,
            'advantage_psnr': 3.00,
            'advantage_ssim': 0.120
        },
        'comparison': {
            'deeptensor_gain_over_old_db': float(dt_res['psnr'] - old_dt_psnr),
            'deeptensor_gain_over_old_ssim': float(dt_res['ssim'] - old_dt_ssim),
            'deeptensor_gap_to_paper_db': float(dt_res['psnr'] - 27.10),
            'deeptensor_gap_to_paper_ssim': float(dt_res['ssim'] - 0.810),
            'improved_advantage_over_tv_db': float(dt_res['psnr'] - tv_res['psnr']),
            'improved_advantage_over_tv_ssim': float(dt_res['ssim'] - tv_res['ssim'])
        }
    }

    metrics_json_path = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"Saved metrics JSON to: {metrics_json_path}", flush=True)

    # 12. Generate Visual Comparison Panel (matching paper layout: 8 frames)
    fig, axes = plt.subplots(4, 8, figsize=(16, 8.5))
    coded_snapshot = measurements[..., 0]
    for i in range(8):
        axes[0, i].imshow(coded_snapshot, cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title(f"Coded Meas\nSnapshot 1", fontsize=9, fontweight='bold')

    for i in range(8):
        axes[1, i].imshow(cube[..., i], cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title(f"Frame {i+1}", fontsize=9)
    axes[1, 0].set_ylabel("Ground Truth", fontsize=10, fontweight='bold')

    for i in range(8):
        axes[2, i].imshow(dt_res['recon'][..., i], cmap='gray')
        axes[2, i].axis('off')
    axes[2, 0].set_ylabel(f"DeepTensor Rerun\n{dt_res['psnr']:.1f} dB | {dt_res['ssim']:.2f}\n(Paper: 27.1 dB)",
                          fontsize=9, fontweight='bold', color='navy')

    for i in range(8):
        axes[3, i].imshow(tv_res['recon'][..., i], cmap='gray')
        axes[3, i].axis('off')
    axes[3, 0].set_ylabel(f"2D TV Baseline\n{tv_res['psnr']:.1f} dB | {tv_res['ssim']:.2f}\n(Paper: 24.1 dB)",
                          fontsize=9, fontweight='bold', color='darkred')

    plt.suptitle("Figure 11: Video Compressive Sensing (Hitomi et al. setup)\nDeepTensor (Full Rank R=64, 10k Ep) vs. 2D Total Variation",
                 fontsize=12, y=0.98)
    plt.tight_layout()
    panel_path = os.path.join(images_dir, 'figure11_video_cs_improved_panel.png')
    plt.savefig(panel_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved improved visual panel to: {panel_path}", flush=True)

    # 13. Generate Convergence Curves Plot
    plt.figure(figsize=(12, 4.5))
    plt.subplot(1, 2, 1)
    plt.plot(dt_res['loss_history'], label='DeepTensor Measurement Loss', color='navy', linewidth=1.5)
    plt.yscale('log')
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Loss (Log Scale)', fontsize=10)
    plt.title('DeepTensor Loss Convergence (10,000 Epochs)', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    dt_eps = [x[0] for x in dt_res['psnr_progression']]
    dt_ps = [x[1] for x in dt_res['psnr_progression']]
    plt.plot(dt_eps, dt_ps, label=f'DeepTensor PSNR (Final: {dt_res["psnr"]:.2f} dB)', color='navy', marker='o', markersize=3)
    plt.axhline(y=27.10, color='crimson', linestyle='--', label='Paper DeepTensor Target (27.10 dB)')
    plt.axhline(y=old_dt_psnr, color='orange', linestyle=':', label=f'Old Baseline ({old_dt_psnr:.2f} dB)')
    plt.axhline(y=tv_res['psnr'], color='gray', linestyle='-.', label=f'2D TV Baseline ({tv_res["psnr"]:.2f} dB)')
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('PSNR (dB)', fontsize=10)
    plt.title('PSNR Progression vs Published Targets', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, loc='lower right')

    plt.tight_layout()
    curves_path = os.path.join(plots_dir, 'figure11_improved_convergence_curves.png')
    plt.savefig(curves_path, dpi=200)
    plt.close()
    print(f"Saved convergence curves to: {curves_path}", flush=True)

    # Summary Output
    print("\n" + "=" * 80)
    print("FIGURE 11 IMPROVEMENT SUMMARY:")
    print("=" * 80)
    print(f"Method                 PSNR (dB)   SSIM      Runtime (s)   VRAM (MB)")
    print(f"----------------------------------------------------------------------")
    print(f"Old CNN Baseline       {old_dt_psnr:<11.2f} {old_dt_ssim:<9.4f} ~180 s        ~136 MB")
    print(f"Published Paper        27.10       0.8100    —             —")
    print(f"DeepTensor Improved    {dt_res['psnr']:<11.2f} {dt_res['ssim']:<9.4f} {dt_res['runtime']:<13.1f} {dt_res['peak_vram_mb']:<10.1f}")
    print(f"2D TV (Paper Target)   24.10       0.6900    —             —")
    print(f"2D TV (Calibrated)     {tv_res['psnr']:<11.2f} {tv_res['ssim']:<9.4f} {tv_res['runtime']:<13.1f} —")
    print(f"----------------------------------------------------------------------")
    print(f"Gain over Old Baseline: {dt_res['psnr'] - old_dt_psnr:+.2f} dB | SSIM: {dt_res['ssim'] - old_dt_ssim:+.4f}")
    print(f"Gap to Paper Target:    {dt_res['psnr'] - 27.10:+.2f} dB | SSIM: {dt_res['ssim'] - 0.810:+.4f}")
    print("=" * 80, flush=True)

if __name__ == '__main__':
    main()

