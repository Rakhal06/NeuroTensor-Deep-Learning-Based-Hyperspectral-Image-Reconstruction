"""
Adaptive Hybrid DeepTensor v1 (AHD-Net): Figure 09 Full-Budget Benchmark
========================================================================
Full-scale 5,000-epoch evaluation on ICVL Hyperspectral Denoising (174x164x260, Rank R=20).
Evaluates AHD-Net with input-conditioned Channel-Aware Global Context Gate (CAGC)
under cosine temperature annealing against historical baseline, controls, and transformer.
"""

import os
import sys
import time
import math
import copy
import json
import random
from typing import Dict, Any, List

import numpy as np
from scipy import io
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, os.path.join(project_root, 'src'))
    sys.path.insert(0, project_root)

from deeptensor import utils, losses
from deeptensor.models.factory import (
    build_deeptensor_model,
    count_parameters,
    save_checkpoint,
    load_checkpoint,
)
from deeptensor.models.attention import (
    AdaptiveTriBranchBlock,
    AdaptiveHybridSpectralNet,
    get_cosine_temperature,
)
from deeptensor.utils.comparison import compute_metrics, to_2d_slice


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_branch_gradient_norms(adaptive_block: AdaptiveTriBranchBlock) -> Dict[str, float]:
    """Compute L2 gradient norms for router and each of the three branches."""
    norms = {}
    modules = {
        'router': adaptive_block.router,
        'cnn_branch': adaptive_block.cnn_branch,
        'pw_branch': adaptive_block.pw_branch,
        'attn_branch': adaptive_block.attn_branch,
    }
    for mod_name, mod in modules.items():
        if mod is None:
            norms[mod_name] = 0.0
            continue
        sq_sum = 0.0
        for p in mod.parameters():
            if p.grad is not None:
                sq_sum += float(p.grad.norm(2).item() ** 2)
        norms[mod_name] = math.sqrt(sq_sum)
    return norms


def extract_router_telemetry(adaptive_block: AdaptiveTriBranchBlock) -> Dict[str, Any]:
    """Extract routing weights, channel distribution, and entropy."""
    gates = adaptive_block.last_gates
    if gates is None:
        return {}
    with torch.no_grad():
        if gates.ndim == 3:
            g_mean = gates.mean(dim=(0, 1)).cpu().numpy()
            c_weights = gates[0].cpu().numpy()  # [C, 3]
        elif gates.ndim == 4:
            g_mean = gates.mean(dim=(0, 1, 2)).cpu().numpy()
            c_weights = gates[0].mean(dim=1).cpu().numpy()
        else:
            g_mean = np.array([1/3, 1/3, 1/3])
            c_weights = np.full((128, 3), 1/3)

        entropy = float(adaptive_block.compute_router_entropy().item())
        sums = gates.sum(dim=-1)
        sums_valid = bool(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))

        telemetry = {
            'mean_weights': {
                'cnn': float(g_mean[0]),
                'pointwise': float(g_mean[1]),
                'attention': float(g_mean[2]),
            },
            'channel_distribution': {
                'cnn': {
                    'min': float(np.min(c_weights[:, 0])),
                    'max': float(np.max(c_weights[:, 0])),
                    'mean': float(np.mean(c_weights[:, 0])),
                    'std': float(np.std(c_weights[:, 0])),
                },
                'pointwise': {
                    'min': float(np.min(c_weights[:, 1])),
                    'max': float(np.max(c_weights[:, 1])),
                    'mean': float(np.mean(c_weights[:, 1])),
                    'std': float(np.std(c_weights[:, 1])),
                },
                'attention': {
                    'min': float(np.min(c_weights[:, 2])),
                    'max': float(np.max(c_weights[:, 2])),
                    'mean': float(np.mean(c_weights[:, 2])),
                    'std': float(np.std(c_weights[:, 2])),
                },
            },
            'router_entropy': entropy,
            'current_tau': float(adaptive_block.tau),
            'sums_to_one': sums_valid,
        }
    return telemetry


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 5000
    rank = 20
    n_inputs = rank
    lr = 1e-3
    reg_noise_std = 1.0 / 30.0
    tau_start = 2.0
    tau_min = 0.5

    out_dir = os.path.join(project_root, 'results', 'extensions', 'adaptive_hybrid_v1_full', 'figure09_hyperspectral')
    os.makedirs(out_dir, exist_ok=True)
    log_file = os.path.join(out_dir, 'training.log')

    def log_print(msg: str):
        print(msg)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    # Initialize log
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("=== AHD-Net Figure 09 Full-Budget Benchmark (5000 Epochs) ===\n")

    log_print("=" * 80)
    log_print("AHD-NET V1 FULL-BUDGET BENCHMARK: FIGURE 09 ICVL HYPERSPECTRAL")
    log_print(f"Device: {device} ({torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'CPU'})")
    log_print(f"Budget: {epochs} epochs | Rank: R={rank} | LR: {lr} | Noise Std: {reg_noise_std:.4f}")
    log_print(f"Temperature Schedule: Cosine Decay {tau_start} -> {tau_min}")
    log_print(f"Output Directory: {out_dir}")
    log_print("=" * 80)

    # 1. Load ICVL Data
    data_path = os.path.join(project_root, 'data', 'icvl.mat')
    data = io.loadmat(data_path)
    cube = data['hypercube'].astype(np.float32)
    cube = utils.resize(cube / cube.max(), 1)
    H, W, nwvl = cube.shape
    log_print(f"--> Loaded ICVL dataset: {cube.shape} (H={H}, W={W}, L={nwvl})")

    # Degrade observation exactly once under Seed 42
    set_seed(42)
    cube_noisy = utils.measure(cube, 2, 100)
    hsten_noisy = torch.tensor(cube_noisy.reshape(H * W, nwvl), dtype=torch.float32)[None, ...].to(device)
    criterion_l1 = losses.L1Norm()
    obs_metrics = compute_metrics(cube, cube_noisy)
    log_print(f"--> Generated observation under Seed 42 (Poisson tau=100, Gaussian SNR=2): PSNR={obs_metrics['psnr']:.2f} dB, SSIM={obs_metrics['ssim']:.4f}")

    # 2. Build Model under Seed 100
    set_seed(100)
    model = build_deeptensor_model(
        rank=rank,
        seq_len=nwvl,
        spatial_type='skip',
        spectral_type='adaptive_hybrid',
        n_inputs=n_inputs,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ffn=512,
        router_type='cagc',
        pe_type='sin_pe',
        tau=tau_start,
    ).to(device)

    total_params = count_parameters(model)
    target_params = 3137808
    delta_params = total_params - target_params
    log_print(f"--> Model constructed: {total_params:,} trainable parameters (Target: {target_params:,}, Delta: {delta_params})")
    assert delta_params == 0, f"Parameter mismatch: {total_params} != {target_params}"

    # 3. Latent Initialization under Seed 200
    set_seed(200)
    im_inp = utils.get_inp([1, n_inputs, H, W]).to(device)
    spec_inp = utils.get_inp([1, n_inputs, nwvl]).to(device)
    im_inp_per = im_inp.detach().clone()
    spec_inp_per = spec_inp.detach().clone()

    optimizer = torch.optim.Adam(list(model.parameters()) + [im_inp, spec_inp], lr=lr)

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
    log_print("\n--> Starting 5,000-Epoch Optimization Loop...")

    model.train()
    for idx in range(epochs):
        # Cosine temperature annealing
        current_tau = get_cosine_temperature(idx, epochs, tau_start=tau_start, tau_min=tau_min)
        adaptive_block.set_temperature(current_tau)

        # DIP input perturbation
        im_pert = im_inp + im_inp_per.normal_() * reg_noise_std
        spec_pert = spec_inp + spec_inp_per.normal_() * reg_noise_std

        hs_estim = model(im_pert, spec_pert)
        loss = criterion_l1(hsten_noisy - hs_estim)

        optimizer.zero_grad()
        loss.backward()

        cur_loss = float(loss.item())
        loss_history.append(cur_loss)

        # Record telemetry every 100 epochs or at epoch 0 / final epoch
        if idx == 0 or (idx + 1) % 100 == 0 or idx == epochs - 1:
            grad_norms = compute_branch_gradient_norms(adaptive_block)
            router_telem = extract_router_telemetry(adaptive_block)
            telem_entry = {
                'epoch': idx + 1,
                'loss': cur_loss,
                'tau': current_tau,
                'router_entropy': router_telem.get('router_entropy', 0.0),
                'mean_weights': router_telem.get('mean_weights', {}),
                'channel_std': {
                    'cnn': router_telem['channel_distribution']['cnn']['std'],
                    'pw': router_telem['channel_distribution']['pointwise']['std'],
                    'attn': router_telem['channel_distribution']['attention']['std'],
                },
                'grad_norms': grad_norms,
            }
            telemetry_history.append(telem_entry)

            if (idx + 1) % 500 == 0 or idx == 0 or idx == epochs - 1:
                mw = router_telem['mean_weights']
                log_print(
                    f"  [Epoch {idx + 1:4d}/{epochs}] Loss: {cur_loss:.6f} | Best Loss: {best_loss:.6f} (Ep {best_epoch}) | "
                    f"tau: {current_tau:.2f} | H: {router_telem['router_entropy']:.4f} | "
                    f"Gates: CNN={mw['cnn']:.3f}, PW={mw['pointwise']:.3f}, Attn={mw['attention']:.3f}"
                )

        # Strictly unsupervised checkpoint selection: lowest measurement L1 loss
        if cur_loss < best_loss:
            best_loss = cur_loss
            best_epoch = idx + 1
            # Reshape reconstruction to [H, W, L]
            recon_mat = hs_estim.detach()[0].cpu().numpy()  # [H*W, L]
            best_recon = recon_mat.reshape(H, W, nwvl)
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
    log_print(f"Best Measurement Loss: {best_loss:.6f} at Epoch {best_epoch}")
    log_print("=" * 80)

    # 4. Save Checkpoints
    best_ckpt_path = os.path.join(out_dir, "best_checkpoint.pth")
    final_ckpt_path = os.path.join(out_dir, "final_checkpoint.pth")

    save_checkpoint(
        model,
        final_ckpt_path,
        optimizer=optimizer,
        epoch=epochs,
        metrics={'final_loss': cur_loss, 'best_loss': best_loss, 'best_epoch': best_epoch, 'peak_vram_mb': peak_vram},
        config={'im_inp': im_inp.detach().cpu(), 'spec_inp': spec_inp.detach().cpu()}
    )

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    save_checkpoint(
        model,
        best_ckpt_path,
        optimizer=optimizer,
        epoch=best_epoch,
        metrics={'best_loss': best_loss, 'best_epoch': best_epoch, 'peak_vram_mb': peak_vram},
        config=best_latents
    )
    log_print(f"--> Saved best checkpoint to: {best_ckpt_path} ({os.path.getsize(best_ckpt_path) / (1024**2):.2f} MB)")
    log_print(f"--> Saved final checkpoint to: {final_ckpt_path} ({os.path.getsize(final_ckpt_path) / (1024**2):.2f} MB)")

    # 5. Evaluate Reconstruction Fidelity Metrics
    metrics = compute_metrics(cube, best_recon)
    log_print("\n" + "=" * 80)
    log_print("RECONSTRUCTION FIDELITY EVALUATION (Full 174x164x260 Volume)")
    log_print(f"  PSNR (Full-Volume)          : {metrics['psnr']:.4f} dB")
    log_print(f"  SSIM                        : {metrics['ssim']:.4f}")
    log_print(f"  RMSE                        : {metrics['rmse']:.6f}")
    log_print(f"  MAE                         : {metrics['mae']:.6f}")
    log_print(f"  Relative Frobenius Error    : {metrics['rel_error']:.6f}")
    log_print("=" * 80)

    # 6. Independent Checkpoint Verification
    log_print("\n--> Performing Independent Checkpoint Verification...")
    verify_model = build_deeptensor_model(
        rank=rank,
        seq_len=nwvl,
        spatial_type='skip',
        spectral_type='adaptive_hybrid',
        n_inputs=n_inputs,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ffn=512,
        router_type='cagc',
        pe_type='sin_pe',
    ).to(device)
    verify_model.eval()

    ckpt_data = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    verify_model.load_state_dict(ckpt_data['model_state_dict'])
    saved_im = ckpt_data['config']['im_inp'].to(device)
    saved_spec = ckpt_data['config']['spec_inp'].to(device)

    with torch.no_grad():
        verify_estim = verify_model(saved_im, saved_spec)
        verify_recon = verify_estim[0].cpu().numpy().reshape(H, W, nwvl)

    verify_metrics = compute_metrics(cube, verify_recon)
    psnr_diff = abs(metrics['psnr'] - verify_metrics['psnr'])
    ssim_diff = abs(metrics['ssim'] - verify_metrics['ssim'])
    log_print(f"  Reloaded Checkpoint PSNR    : {verify_metrics['psnr']:.4f} dB (Discrepancy: {psnr_diff:.6f} dB)")
    log_print(f"  Reloaded Checkpoint SSIM    : {verify_metrics['ssim']:.4f} (Discrepancy: {ssim_diff:.6f})")
    assert psnr_diff < 1e-4, f"Checkpoint verification failed: PSNR discrepancy {psnr_diff}"
    assert ssim_diff < 1e-4, f"Checkpoint verification failed: SSIM discrepancy {ssim_diff}"
    log_print("  INDEPENDENT VERIFICATION STATUS: 100% REPRODUCIBLE (PASS)")

    # 7. Comparison Against Historical Models
    historical_json_path = os.path.join(project_root, 'results', 'hybrid', 'figure09_hyperspectral', 'figure09_icvl_hsi_comparison.json')
    comparison_summary = {}
    if os.path.exists(historical_json_path):
        with open(historical_json_path, 'r') as f:
            hist_data = json.load(f)
        hist_models = hist_data.get('models', {})

        log_print("\n" + "=" * 80)
        log_print("COMPARATIVE BENCHMARK: AHD-NET V1 VS HISTORICAL ARCHITECTURES")
        log_print("=" * 80)
        log_print(f"{'Model Architecture':<22} | {'PSNR (dB)':<10} | {'SSIM':<8} | {'RMSE':<9} | {'Params':<10} | {'Delta vs Base':<12}")
        log_print("-" * 80)

        base_psnr = hist_models.get('baseline', {}).get('metrics', {}).get('psnr', 46.0653)

        # Print all historical models
        for m_name in ['baseline', 'cnn_control', 'pointwise_control', 'hybrid', 'transformer']:
            if m_name in hist_models:
                m_info = hist_models[m_name]
                m_psnr = m_info['metrics']['psnr']
                m_ssim = m_info['metrics']['ssim']
                m_rmse = m_info['metrics']['rmse']
                m_p = m_info['params']
                d_base = m_psnr - base_psnr
                log_print(f"{m_name:<22} | {m_psnr:<10.4f} | {m_ssim:<8.4f} | {m_rmse:<9.5f} | {m_p:<10,d} | {d_base:+.4f} dB")

        # Print AHD-Net
        d_ahd_base = metrics['psnr'] - base_psnr
        log_print(f"{'adaptive_hybrid_v1':<22} | {metrics['psnr']:<10.4f} | {metrics['ssim']:<8.4f} | {metrics['rmse']:<9.5f} | {total_params:<10,d} | {d_ahd_base:+.4f} dB")
        log_print("-" * 80)

        # Comparisons dict
        comparison_summary = {
            'ahd_net_v1': {
                'metrics': metrics,
                'params': total_params,
                'best_epoch': best_epoch,
                'best_loss': best_loss,
                'runtime_sec': total_train_time,
                'peak_vram_mb': peak_vram,
            },
            'historical_models': hist_models,
            'deltas_vs_historical': {
                'vs_baseline': metrics['psnr'] - hist_models.get('baseline', {}).get('metrics', {}).get('psnr', 0.0),
                'vs_hybrid': metrics['psnr'] - hist_models.get('hybrid', {}).get('metrics', {}).get('psnr', 0.0),
                'vs_cnn_control': metrics['psnr'] - hist_models.get('cnn_control', {}).get('metrics', {}).get('psnr', 0.0),
                'vs_pointwise_control': metrics['psnr'] - hist_models.get('pointwise_control', {}).get('metrics', {}).get('psnr', 0.0),
                'vs_transformer': metrics['psnr'] - hist_models.get('transformer', {}).get('metrics', {}).get('psnr', 0.0),
            }
        }

    # Save telemetry history and comparison JSON
    telem_file = os.path.join(out_dir, "telemetry_history.json")
    with open(telem_file, 'w') as f:
        json.dump(telemetry_history, f, indent=2)

    comp_file = os.path.join(out_dir, "figure09_ahd_net_comparison.json")
    with open(comp_file, 'w') as f:
        json.dump(comparison_summary, f, indent=2)

    # 8. Generate Visual Diagnostic Figures
    log_print("\n--> Generating diagnostic visual figures...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    central_wvl = nwvl // 2

    im0 = axes[0].imshow(cube[:, :, central_wvl], cmap='viridis', vmin=0, vmax=1)
    axes[0].set_title(f"Ground Truth (Band {central_wvl})")
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(best_recon[:, :, central_wvl], cmap='viridis', vmin=0, vmax=1)
    axes[1].set_title(f"AHD-Net v1 (PSNR: {metrics['psnr']:.2f} dB)")
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    err = np.abs(cube[:, :, central_wvl] - best_recon[:, :, central_wvl])
    im2 = axes[2].imshow(err, cmap='magma', vmin=0, vmax=0.05)
    axes[2].set_title(f"Absolute Error (MAE: {np.mean(err):.4f})")
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "figure09_ahd_net_reconstruction.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_print(f"--> Saved visual diagnostic figure to: {fig_path}")

    # Plot routing trajectory
    if telemetry_history:
        epochs_logged = [e['epoch'] for e in telemetry_history]
        g_cnn = [e['mean_weights']['cnn'] for e in telemetry_history]
        g_pw = [e['mean_weights']['pointwise'] for e in telemetry_history]
        g_attn = [e['mean_weights']['attention'] for e in telemetry_history]
        entropy_traj = [e['router_entropy'] for e in telemetry_history]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax1.plot(epochs_logged, g_cnn, label='CNN Branch (Local)', color='tab:blue', linewidth=2)
        ax1.plot(epochs_logged, g_pw, label='Pointwise Branch (Inter-Channel)', color='tab:green', linewidth=2)
        ax1.plot(epochs_logged, g_attn, label='Attention Branch (Global)', color='tab:red', linewidth=2)
        ax1.axhline(1/3, color='gray', linestyle='--', alpha=0.7, label='Uniform (1/3)')
        ax1.set_ylabel('Mean Routing Weight')
        ax1.set_title('AHD-Net v1 Router Trajectory: Figure 09 ICVL HSI (5000 Epochs)')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs_logged, entropy_traj, label='Router Shannon Entropy', color='tab:purple', linewidth=2)
        ax2.axhline(math.log(3.0), color='gray', linestyle='--', alpha=0.7, label='Max Entropy ln(3) ≈ 1.0986')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Entropy (nats)')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        traj_path = os.path.join(out_dir, "figure09_router_trajectory.png")
        plt.savefig(traj_path, dpi=300, bbox_inches='tight')
        plt.close()
        log_print(f"--> Saved router trajectory figure to: {traj_path}")

    log_print("\n" + "=" * 80)
    log_print("FIGURE 09 FULL-BUDGET BENCHMARK COMPLETE AND AUDITED")
    log_print("=" * 80)


if __name__ == '__main__':
    main()

