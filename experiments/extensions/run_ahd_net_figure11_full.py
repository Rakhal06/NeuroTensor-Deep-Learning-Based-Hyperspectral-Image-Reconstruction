"""
Full-Budget Benchmark Execution Script for AHD-Net v1 on Figure 11
===================================================================
Application: Figure 11 — Video Compressive Sensing (CACTI)
Dataset: Cat Video (data/cat_video.mat), shape: (128, 128, 64)
Forward Model: 8-Frame Temporal Multiplexing with Random Binary Coded Aperture
Tensor Factorization Rank: R = 32
Noise: Poisson tau = 100 and Gaussian SNR = 2 (Observation Seed 42)
Latent Initialization: Seed 200 (Clean Noise Latents)
Training Budget: 3,000 Epochs
Temperature Schedule: Cosine Decay (2.0 -> 0.5)
Target Parameters: 834,632 (Delta: 0)
Supervision: 100% Unsupervised Measurement L2 Loss
Output: results/extensions/adaptive_hybrid_v1_full/figure11_video_cs/
"""

import os
import sys
import copy
import time
import math
import json
import random
from typing import Dict, Any, List

import numpy as np
from scipy import io
import torch
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, os.path.join(project_root, 'src'))
    sys.path.insert(0, project_root)

from deeptensor import utils, losses
from deeptensor.physics import operators as lin_inverse
from deeptensor.models.factory import (
    build_deeptensor_model,
    count_parameters,
    save_checkpoint,
)
from deeptensor.models.attention import AdaptiveTriBranchBlock
from deeptensor.utils.comparison import compute_metrics, to_2d_slice
from deeptensor.utils.metrics import calculate_psnr, calculate_ssim


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_cosine_temperature(step: int, max_steps: int, tau_start: float = 2.0, tau_min: float = 0.5) -> float:
    return tau_min + 0.5 * (tau_start - tau_min) * (1.0 + math.cos(math.pi * step / max_steps))


def extract_router_telemetry(adaptive_block: AdaptiveTriBranchBlock) -> Dict[str, Any]:
    with torch.no_grad():
        gates = adaptive_block.last_gates
        if gates is None:
            return {}
        mean_weights = gates.mean(dim=(0, 1)).cpu().tolist()
        entropy = float(adaptive_block.compute_router_entropy().item())
        channel_weights = gates[0].cpu().numpy()  # [C, 3]

        c_w = channel_weights[:, 0]
        p_w = channel_weights[:, 1]
        a_w = channel_weights[:, 2]

        dominant = np.argmax(channel_weights, axis=-1)
        num_cnn = int(np.sum(dominant == 0))
        num_pw = int(np.sum(dominant == 1))
        num_attn = int(np.sum(dominant == 2))

        return {
            'mean_weights': {
                'cnn': mean_weights[0],
                'pointwise': mean_weights[1],
                'attention': mean_weights[2],
            },
            'router_entropy': entropy,
            'current_tau': float(adaptive_block.tau),
            'dominant_counts': {
                'cnn': num_cnn,
                'pointwise': num_pw,
                'attention': num_attn,
            },
            'channel_distribution': {
                'cnn': {'mean': float(np.mean(c_w)), 'std': float(np.std(c_w)), 'min': float(np.min(c_w)), 'max': float(np.max(c_w))},
                'pointwise': {'mean': float(np.mean(p_w)), 'std': float(np.std(p_w)), 'min': float(np.min(p_w)), 'max': float(np.max(p_w))},
                'attention': {'mean': float(np.mean(a_w)), 'std': float(np.std(a_w)), 'min': float(np.min(a_w)), 'max': float(np.max(a_w))},
            }
        }


def compute_branch_gradient_norms(adaptive_block: AdaptiveTriBranchBlock) -> Dict[str, float]:
    g_norms = {}
    for name, branch in [
        ('cnn', adaptive_block.cnn_branch),
        ('pointwise', adaptive_block.pw_branch),
        ('attention', adaptive_block.attn_branch),
        ('router', adaptive_block.router),
    ]:
        norms = [torch.norm(p.grad) for p in branch.parameters() if p.grad is not None]
        if norms:
            g_norms[name] = float(torch.norm(torch.stack(norms)).item())
        else:
            g_norms[name] = 0.0
    return g_norms


def run_benchmark(epochs: int = 3000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rank = 32
    n_inputs = rank
    lr = 1e-3
    reg_noise_std = 1.0 / 30.0
    tau_start = 2.0
    tau_min = 0.5
    target_params = 834632
    nframes = 8

    out_dir = os.path.join(project_root, 'results', 'extensions', 'adaptive_hybrid_v1_full', 'figure11_video_cs')
    os.makedirs(out_dir, exist_ok=True)
    log_file = os.path.join(out_dir, "training.log")

    def log_print(msg: str):
        print(msg)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    # Clear log file
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"=== AHD-Net Figure 11 Full-Budget Benchmark ({epochs} Epochs) ===\n")

    log_print("=" * 80)
    log_print("AHD-NET V1 FULL-BUDGET BENCHMARK: FIGURE 11 VIDEO COMPRESSIVE SENSING (CACTI)")
    log_print(f"Device: {device} ({torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'CPU'})")
    log_print(f"Budget: {epochs} epochs | Rank: R={rank} | LR: {lr} | Noise: Poisson tau=100, Gaussian SNR=2")
    log_print(f"Temperature Schedule: Cosine Decay {tau_start} -> {tau_min}")
    log_print(f"Output Directory: {out_dir}")
    log_print("=" * 80)

    # 1. Load Video Dataset
    data_path = os.path.join(project_root, 'data', 'cat_video.mat')
    data = io.loadmat(data_path)
    cube = data['hypercube'].astype(np.float32)
    cube = utils.resize(cube / cube.max(), 1)
    H, W, totalframes = cube.shape
    log_print(f"--> Loaded Cat Video dataset: ({H}, {W}, {totalframes}) (H={H}, W={W}, T={totalframes})")

    # Generate 8-frame CACTI measurements under Seed 42
    set_seed(42)
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

    obs_metrics = compute_metrics(cube, cube_noisy)
    log_print(f"--> Generated observation under Seed 42 (Poisson tau=100, Gaussian SNR=2): PSNR={obs_metrics['psnr']:.2f} dB, SSIM={obs_metrics['ssim']:.4f}")
    log_print(f"--> CACTI Measurements: {measurements_ten.shape} from 64 frames (8x compression)")

    # 2. Build AHD-Net Model under Seed 100
    set_seed(100)
    model = build_deeptensor_model(
        rank=rank,
        seq_len=totalframes,
        spatial_type='skip',
        spectral_type='adaptive_hybrid',
        n_inputs=n_inputs,
        skip_n33d=64,
        skip_n33u=64,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ffn=512,
        router_type='cagc',
        pe_type='sin_pe',
        tau=tau_start,
    ).to(device)

    total_params = count_parameters(model)
    delta_params = total_params - target_params
    log_print(f"--> Model constructed: {total_params:,} trainable parameters (Target: {target_params:,}, Delta: {delta_params})")
    assert delta_params == 0, f"Parameter mismatch: {total_params} != {target_params}"

    # 3. Latent Initialization under Seed 200
    set_seed(200)
    im_inp = utils.get_inp([1, n_inputs, H, W], device=device)
    spec_inp = utils.get_inp([1, n_inputs, totalframes], device=device)
    im_inp_per = im_inp.detach().clone()
    spec_inp_per = spec_inp.detach().clone()

    optimizer = torch.optim.Adam(
        list(model.parameters()) + [im_inp, spec_inp],
        lr=lr
    )

    adaptive_block = model.spectral_net.adaptive_block

    best_loss = float('inf')
    best_epoch = -1
    best_recon = None
    best_model_state = None
    best_latents = None

    loss_history = []
    telemetry_history = []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    train_start_time = time.time()
    log_print(f"\n--> Starting {epochs}-Epoch Optimization Loop...")

    model.train()
    for idx in range(epochs):
        # Cosine temperature annealing
        current_tau = get_cosine_temperature(idx, epochs, tau_start=tau_start, tau_min=tau_min)
        adaptive_block.set_temperature(current_tau)

        # DIP input perturbation
        im_pert = im_inp + im_inp_per.normal_() * reg_noise_std
        spec_pert = spec_inp + spec_inp_per.normal_() * reg_noise_std

        # Forward pass
        mat_estim = model(im_pert, spec_pert)  # (1, H*W, totalframes)
        cube_estim = mat_estim.reshape(1, H, W, totalframes).permute(0, 3, 1, 2)  # (1, totalframes, H, W)
        meas_estim = lin_inverse.video2codedvideo(cube_estim, masks_ten, nframes)  # (1, 8, H, W)

        loss = criterion_l2(measurements_ten - meas_estim)

        optimizer.zero_grad()
        loss.backward()

        cur_loss = float(loss.item())
        loss_history.append(cur_loss)

        # Telemetry capture every 100 epochs or at epoch 0 / final epoch
        if idx == 0 or (idx + 1) % 100 == 0 or idx == epochs - 1:
            grad_norms = compute_branch_gradient_norms(adaptive_block)
            router_telem = extract_router_telemetry(adaptive_block)
            telem_entry = {
                'epoch': idx + 1,
                'loss': cur_loss,
                'tau': current_tau,
                'router_entropy': router_telem.get('router_entropy', 0.0),
                'mean_weights': router_telem.get('mean_weights', {}),
                'dominant_counts': router_telem.get('dominant_counts', {}),
                'channel_std': {
                    'cnn': router_telem['channel_distribution']['cnn']['std'],
                    'pointwise': router_telem['channel_distribution']['pointwise']['std'],
                    'attention': router_telem['channel_distribution']['attention']['std'],
                },
                'gradient_norms': grad_norms,
            }
            telemetry_history.append(telem_entry)

            if (idx + 1) % 200 == 0 or idx == 0 or idx == epochs - 1:
                mw = router_telem['mean_weights']
                dc = router_telem['dominant_counts']
                log_print(
                    f"  [Epoch {idx + 1:4d}/{epochs}] Loss: {cur_loss:.5e} | Best Loss: {best_loss:.5e} (Ep {best_epoch}) | "
                    f"tau: {current_tau:.2f} | H: {router_telem['router_entropy']:.4f} | "
                    f"Gates: CNN={mw['cnn']:.3f}, PW={mw['pointwise']:.3f}, Attn={mw['attention']:.3f} | "
                    f"Dominance: CNN={dc['cnn']}, PW={dc['pointwise']}, Attn={dc['attention']}"
                )

        # Strictly unsupervised checkpoint selection: lowest measurement L2 loss
        if cur_loss < best_loss:
            best_loss = cur_loss
            best_epoch = idx + 1
            best_recon = cube_estim.permute(0, 2, 3, 1)[0].detach().cpu().numpy()
            best_model_state = copy.deepcopy(model.state_dict())
            best_latents = {
                'im_inp': im_inp.detach().cpu(),
                'spec_inp': spec_inp.detach().cpu(),
            }

        optimizer.step()

    total_train_time = time.time() - train_start_time
    peak_vram = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if torch.cuda.is_available() else 0.0

    log_print("\n" + "=" * 80)
    log_print("TRAINING COMPLETE")
    log_print(f"Total Runtime: {total_train_time:.2f}s ({total_train_time / 60.0:.2f} min) | Peak VRAM: {peak_vram:.1f} MB")
    log_print(f"Best Measurement L2 Loss: {best_loss:.6e} at Epoch {best_epoch}")
    log_print("=" * 80)

    # 4. Save Checkpoints
    best_ckpt_path = os.path.join(out_dir, "best_checkpoint.pth")
    final_ckpt_path = os.path.join(out_dir, "final_checkpoint.pth")

    tau_best = get_cosine_temperature(best_epoch - 1, epochs, tau_start=tau_start, tau_min=tau_min)

    save_checkpoint(
        model,
        best_ckpt_path,
        optimizer=optimizer,
        epoch=best_epoch,
        metrics={'best_loss': best_loss, 'best_epoch': best_epoch, 'peak_vram_mb': peak_vram, 'tau_at_best': tau_best},
        config=best_latents
    )
    save_checkpoint(
        model,
        final_ckpt_path,
        optimizer=optimizer,
        epoch=epochs,
        metrics={'final_loss': cur_loss, 'best_loss': best_loss, 'peak_vram_mb': peak_vram},
        config={
            'im_inp': im_inp.detach().cpu(),
            'spec_inp': spec_inp.detach().cpu(),
        }
    )
    log_print(f"--> Saved best checkpoint to: {best_ckpt_path} ({os.path.getsize(best_ckpt_path) / (1024**2):.2f} MB)")
    log_print(f"--> Saved final checkpoint to: {final_ckpt_path} ({os.path.getsize(final_ckpt_path) / (1024**2):.2f} MB)")

    # 5. Evaluate Reconstruction Fidelity Metrics (Full Volume)
    # Protocol A: In-loop snapshot captured at best epoch (identical to historical benchmark script)
    metrics_inloop = compute_metrics(cube, best_recon)

    # Protocol B: Reloaded checkpoint evaluated with clean saved latents in DIP train mode (prescribed tau)
    model.load_state_dict(best_model_state)
    adaptive_block.set_temperature(tau_best)
    model.train()
    with torch.no_grad():
        out_mat = model(best_latents['im_inp'].to(device), best_latents['spec_inp'].to(device))
        recon_dip = out_mat[0].cpu().numpy().reshape(H, W, totalframes)
    metrics_dip = compute_metrics(cube, recon_dip)

    # Protocol C: Unit temperature tau=1.0 diagnostic ablation
    adaptive_block.set_temperature(1.0)
    with torch.no_grad():
        out_mat_tau1 = model(best_latents['im_inp'].to(device), best_latents['spec_inp'].to(device))
        recon_tau1 = out_mat_tau1[0].cpu().numpy().reshape(H, W, totalframes)
    metrics_tau1 = compute_metrics(cube, recon_tau1)
    adaptive_block.set_temperature(tau_best)

    # Protocol D: Eval mode (accumulated running statistics)
    model.eval()
    with torch.no_grad():
        out_mat_eval = model(best_latents['im_inp'].to(device), best_latents['spec_inp'].to(device))
        recon_eval = out_mat_eval[0].cpu().numpy().reshape(H, W, totalframes)
    metrics_eval = compute_metrics(cube, recon_eval)

    log_print("\n" + "=" * 80)
    log_print("RECONSTRUCTION FIDELITY EVALUATION (Full 128x128x64 Video Volume)")
    log_print("=" * 80)
    log_print("Protocol A (In-Loop Snapshot / Historical Protocol Alignment):")
    log_print(f"  PSNR (Full-Volume)       : {metrics_inloop['psnr']:.4f} dB")
    log_print(f"  SSIM                     : {metrics_inloop['ssim']:.4f}")
    log_print(f"  RMSE                     : {metrics_inloop['rmse']:.6f}")
    log_print(f"  MAE                      : {metrics_inloop['mae']:.6f}")
    log_print(f"  Relative Frobenius Error : {metrics_inloop['rel_error']:.6f}")
    log_print("-" * 80)
    log_print(f"Protocol B (Clean-Latent DIP Inference, tau={tau_best:.4f}):")
    log_print(f"  PSNR (Full-Volume)       : {metrics_dip['psnr']:.4f} dB")
    log_print(f"  SSIM                     : {metrics_dip['ssim']:.4f}")
    log_print(f"  RMSE                     : {metrics_dip['rmse']:.6f}")
    log_print("-" * 80)
    log_print(f"Protocol C (Diagnostic Unit Temperature tau=1.0):")
    log_print(f"  PSNR (Full-Volume)       : {metrics_tau1['psnr']:.4f} dB")
    log_print(f"  SSIM                     : {metrics_tau1['ssim']:.4f}")
    log_print("-" * 80)
    log_print(f"Protocol D (Eval Mode Running Statistics):")
    log_print(f"  PSNR (Full-Volume)       : {metrics_eval['psnr']:.4f} dB (BN discrepancy: -{metrics_dip['psnr'] - metrics_eval['psnr']:.2f} dB)")
    log_print(f"  SSIM                     : {metrics_eval['ssim']:.4f}")
    log_print("=" * 80)

    # 6. Load Historical Figure 11 Results for Direct Benchmark Comparison
    hist_path = os.path.join(project_root, 'results', 'hybrid', 'figure11_video_cs', 'figure11_video_cs_comparison.json')
    with open(hist_path, 'r') as f:
        hist_data = json.load(f)
    hist_models = hist_data['models']

    base_psnr = hist_models['baseline']['metrics']['psnr']

    log_print("\n" + "=" * 80)
    log_print("COMPARATIVE BENCHMARK TABLE: FIGURE 11 VIDEO CS (All 6 Architectures)")
    log_print("=" * 80)
    log_print(f"{'Model Architecture':<22} | {'PSNR (dB)':<10} | {'SSIM':<8} | {'RMSE':<9} | {'Params':<10} | {'Delta vs Base':<12}")
    log_print("-" * 80)

    for m_name in ['baseline', 'cnn_control', 'pointwise_control', 'hybrid', 'transformer']:
        m_info = hist_models[m_name]
        m_psnr = m_info['metrics']['psnr']
        m_ssim = m_info['metrics']['ssim']
        m_rmse = m_info['metrics']['rmse']
        m_p = m_info['params']
        d_base = m_psnr - base_psnr
        log_print(f"{m_name:<22} | {m_psnr:<10.4f} | {m_ssim:<8.4f} | {m_rmse:<9.5f} | {m_p:<10,d} | {d_base:+.4f} dB")

    # AHD-Net Results under both protocols
    d_inloop = metrics_inloop['psnr'] - base_psnr
    d_dip = metrics_dip['psnr'] - base_psnr
    log_print(f"{'AHD-Net v1 (In-Loop)':<22} | {metrics_inloop['psnr']:<10.4f} | {metrics_inloop['ssim']:<8.4f} | {metrics_inloop['rmse']:<9.5f} | {total_params:<10,d} | {d_inloop:+.4f} dB")
    log_print(f"{'AHD-Net v1 (Clean DIP)':<22} | {metrics_dip['psnr']:<10.4f} | {metrics_dip['ssim']:<8.4f} | {metrics_dip['rmse']:<9.5f} | {total_params:<10,d} | {d_dip:+.4f} dB")
    log_print("=" * 80)

    # Final router telemetry at checkpoint
    final_router_telem = extract_router_telemetry(adaptive_block)

    # 7. Save Comparison Summary JSON
    comparison_summary = {
        'experiment': 'figure11_video_cs_full_budget',
        'ahd_net_v1': {
            'metrics_inloop': metrics_inloop,
            'metrics_dip': metrics_dip,
            'metrics_tau1': metrics_tau1,
            'metrics_eval_mode': metrics_eval,
            'params': total_params,
            'best_epoch': best_epoch,
            'best_loss': best_loss,
            'tau_at_best': tau_best,
            'runtime_sec': total_train_time,
            'peak_vram_mb': peak_vram,
            'router_telemetry_at_best': final_router_telem,
        },
        'historical_models': hist_models,
        'deltas_vs_historical': {
            'vs_baseline_inloop': metrics_inloop['psnr'] - hist_models['baseline']['metrics']['psnr'],
            'vs_hybrid_inloop': metrics_inloop['psnr'] - hist_models['hybrid']['metrics']['psnr'],
            'vs_cnn_control_inloop': metrics_inloop['psnr'] - hist_models['cnn_control']['metrics']['psnr'],
            'vs_pointwise_control_inloop': metrics_inloop['psnr'] - hist_models['pointwise_control']['metrics']['psnr'],
            'vs_transformer_inloop': metrics_inloop['psnr'] - hist_models['transformer']['metrics']['psnr'],
            'vs_baseline_dip': metrics_dip['psnr'] - hist_models['baseline']['metrics']['psnr'],
            'vs_hybrid_dip': metrics_dip['psnr'] - hist_models['hybrid']['metrics']['psnr'],
            'vs_cnn_control_dip': metrics_dip['psnr'] - hist_models['cnn_control']['metrics']['psnr'],
            'vs_pointwise_control_dip': metrics_dip['psnr'] - hist_models['pointwise_control']['metrics']['psnr'],
            'vs_transformer_dip': metrics_dip['psnr'] - hist_models['transformer']['metrics']['psnr'],
        },
        'telemetry_history': telemetry_history,
    }

    comp_file = os.path.join(out_dir, 'figure11_ahd_net_comparison.json')
    with open(comp_file, 'w') as f:
        json.dump(comparison_summary, f, indent=2)
    log_print(f"\n--> Saved comparison summary to: {comp_file}")

    # 8. Generate Visual Diagnostic Panels
    central_t = totalframes // 2  # Frame 32
    gt_slice = cube[:, :, central_t]
    ahd_slice = recon_dip[:, :, central_t]
    err_map = np.abs(gt_slice - ahd_slice)

    # Panel 1: Reconstruction & Error Map
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(gt_slice, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f"Ground Truth (Frame {central_t})")
    axes[0].axis('off')

    axes[1].imshow(ahd_slice, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f"AHD-Net v1 Recon\nFull PSNR: {metrics_dip['psnr']:.2f} dB | SSIM: {metrics_dip['ssim']:.4f}")
    axes[1].axis('off')

    im2 = axes[2].imshow(err_map, cmap='inferno', vmin=0, vmax=0.15)
    axes[2].set_title(f"Absolute Error Map\nMean Error: {np.mean(err_map):.5f}")
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "figure11_ahd_net_reconstruction.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_print(f"--> Saved reconstruction panel to: {fig_path}")

    # Panel 2: Temporal Frame Strip (Frames 8, 16, 24, 32, 40, 48)
    sample_frames = [8, 16, 24, 32, 40, 48]
    fig, axes = plt.subplots(2, len(sample_frames), figsize=(18, 6))
    for col_idx, f_idx in enumerate(sample_frames):
        axes[0, col_idx].imshow(cube[:, :, f_idx], cmap='gray', vmin=0, vmax=1)
        axes[0, col_idx].set_title(f"GT Frame {f_idx}", fontsize=10, fontweight='bold')
        axes[0, col_idx].axis('off')

        axes[1, col_idx].imshow(recon_dip[:, :, f_idx], cmap='gray', vmin=0, vmax=1)
        axes[1, col_idx].set_title(f"AHD-Net Frame {f_idx}", fontsize=10)
        axes[1, col_idx].axis('off')

    plt.tight_layout()
    strip_path = os.path.join(out_dir, "figure11_temporal_strip.png")
    plt.savefig(strip_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_print(f"--> Saved temporal strip to: {strip_path}")

    # Panel 3: Router Trajectory Plot
    epochs_log = [e['epoch'] for e in telemetry_history]
    cnn_log = [e['mean_weights']['cnn'] for e in telemetry_history]
    pw_log = [e['mean_weights']['pointwise'] for e in telemetry_history]
    attn_log = [e['mean_weights']['attention'] for e in telemetry_history]
    ent_log = [e['router_entropy'] for e in telemetry_history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.plot(epochs_log, cnn_log, label='CNN Branch (Local)', color='tab:blue', linewidth=2)
    ax1.plot(epochs_log, pw_log, label='Pointwise Branch (Inter-Channel)', color='tab:green', linewidth=2)
    ax1.plot(epochs_log, attn_log, label='Attention Branch (Global)', color='tab:red', linewidth=2)
    ax1.axhline(1/3, color='gray', linestyle='--', alpha=0.7, label='Uniform (1/3)')
    ax1.set_ylabel('Mean Routing Weight')
    ax1.set_title(f'AHD-Net v1 Router Trajectory: Figure 11 Video CS ({epochs} Epochs)')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_log, ent_log, label='Router Shannon Entropy', color='tab:purple', linewidth=2)
    ax2.axhline(math.log(3.0), color='gray', linestyle='--', alpha=0.7, label='Max Entropy ln(3) ≈ 1.0986')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Entropy (nats)')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    traj_path = os.path.join(out_dir, "figure11_router_trajectory.png")
    plt.savefig(traj_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_print(f"--> Saved router trajectory plot to: {traj_path}")

    # Panel 4: Comprehensive Multi-Model Benchmark Panel
    fig, axes = plt.subplots(1, 6, figsize=(24, 4))
    axes[0].imshow(gt_slice, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f"Ground Truth\nFrame {central_t}")
    axes[0].axis('off')

    axes[1].imshow(ahd_slice, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f"AHD-Net v1 (Ours)\n{metrics_dip['psnr']:.2f} dB | {metrics_dip['ssim']:.4f}")
    axes[1].axis('off')

    axes[2].imshow(err_map, cmap='inferno', vmin=0, vmax=0.15)
    axes[2].set_title(f"AHD-Net Error Map\nMAE: {np.mean(err_map):.5f}")
    axes[2].axis('off')

    # Bar chart of PSNRs across all 6 models
    model_names = ['Baseline', 'CNN Ctrl', 'PW Ctrl', 'Hybrid', 'Transf.', 'AHD-Net']
    psnrs = [
        hist_models['baseline']['metrics']['psnr'],
        hist_models['cnn_control']['metrics']['psnr'],
        hist_models['pointwise_control']['metrics']['psnr'],
        hist_models['hybrid']['metrics']['psnr'],
        hist_models['transformer']['metrics']['psnr'],
        metrics_dip['psnr'],
    ]
    colors = ['#1f77b4', '#aec7e8', '#ffbb78', '#ff7f0e', '#d62728', '#2ca02c']

    ax_bar = axes[3]
    bars = ax_bar.bar(model_names, psnrs, color=colors)
    ax_bar.set_ylim(23.0, 26.5)
    ax_bar.set_ylabel('PSNR (dB)')
    ax_bar.set_title('Figure 11 Model Comparison')
    ax_bar.tick_params(axis='x', rotation=45)
    for bar in bars:
        yval = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f"{yval:.2f}", ha='center', va='bottom', fontsize=8)

    # Loss convergence curve
    axes[4].plot(loss_history, color='blue', linewidth=1.5)
    axes[4].set_yscale('log')
    axes[4].set_xlabel('Epoch')
    axes[4].set_ylabel('L2 Meas Loss (log)')
    axes[4].set_title('Loss Convergence')
    axes[4].grid(True, alpha=0.3)

    # Final gating breakdown
    mw_final = final_router_telem.get('mean_weights', {'cnn': 1/3, 'pointwise': 1/3, 'attention': 1/3})
    axes[5].pie(
        [mw_final['cnn'], mw_final['pointwise'], mw_final['attention']],
        labels=['CNN (Local)', 'PW (Inter-Ch)', 'Attn (Global)'],
        autopct='%1.1f%%',
        colors=['#1f77b4', '#2ca02c', '#d62728'],
        startangle=90
    )
    axes[5].set_title('Final Gating Balance')

    plt.tight_layout()
    comp_panel_path = os.path.join(out_dir, "figure11_full_benchmark_panel.png")
    plt.savefig(comp_panel_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_print(f"--> Saved full benchmark panel to: {comp_panel_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="AHD-Net Figure 11 Full-Budget Benchmark")
    parser.add_argument('--epochs', type=int, default=3000, help="Training epochs (default: 3000)")
    args = parser.parse_args()
    run_benchmark(epochs=args.epochs)
