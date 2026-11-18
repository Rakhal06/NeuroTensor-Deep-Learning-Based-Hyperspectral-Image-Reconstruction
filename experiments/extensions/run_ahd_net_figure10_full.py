"""
Full-Budget Benchmark Execution Script for AHD-Net v1 on Figure 10
===================================================================
Application: Figure 10 — Extended Yale Face B Tensor Denoising
Decomposition: 3-Way CP/PARAFAC (Rank R=2000, 192x168x160)
Noise: 30% Salt & Pepper Noise (Observation Seed 42)
Latent Initialization: Seed 200 (Positional Encodings)
Training Budget: 1,200 Epochs
Temperature Schedule: Cosine Decay (2.0 -> 0.5)
Target Parameters: 5,602,988 (Delta: 0)
Supervision: 100% Unsupervised Measurement L1 Loss
Output: results/extensions/adaptive_hybrid_v1_full/figure10_yale_denoising/
"""

import os
import sys
import copy
import time
import math
import json
import random
from typing import Dict, Any

import numpy as np
from scipy import io
import torch
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
    build_deeptensor_cp_model,
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


def add_salt_and_pepper_noise(tensor: np.ndarray, noise_ratio: float = 0.30) -> np.ndarray:
    noisy = tensor.copy()
    mask = np.random.rand(*tensor.shape)
    salt = mask < (noise_ratio / 2.0)
    pepper = (mask >= (noise_ratio / 2.0)) & (mask < noise_ratio)
    noisy[salt] = 1.0
    noisy[pepper] = 0.0
    return noisy


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


def run_benchmark(epochs: int = 1200):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rank = 2000
    n_inputs = rank
    lr = 1e-3
    tau_start = 2.0
    tau_min = 0.5
    target_params = 5602988

    out_dir = os.path.join(project_root, 'results', 'extensions', 'adaptive_hybrid_v1_full', 'figure10_yale_denoising')
    os.makedirs(out_dir, exist_ok=True)
    log_file = os.path.join(out_dir, "training.log")

    def log_print(msg: str):
        print(msg)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    # Clear log file
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("=== AHD-Net Figure 10 Full-Budget Benchmark (1200 Epochs) ===\n")

    log_print("=" * 80)
    log_print("AHD-NET V1 FULL-BUDGET BENCHMARK: FIGURE 10 YALE FACE B TENSOR DENOISING")
    log_print(f"Device: {device} ({torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'CPU'})")
    log_print(f"Budget: {epochs} epochs | Rank: R={rank} | LR: {lr} | Noise: 30% Salt & Pepper")
    log_print(f"Temperature Schedule: Cosine Decay {tau_start} -> {tau_min}")
    log_print(f"Output Directory: {out_dir}")
    log_print("=" * 80)

    # 1. Load Yale Face B Dataset
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
    log_print(f"--> Loaded Yale Face B dataset: ({H}, {W}, {T}) (H={H}, W={W}, T={T})")

    # Generate 30% salt-and-pepper noise under Seed 42
    set_seed(42)
    noisy_tensor = add_salt_and_pepper_noise(clean_tensor, 0.30)
    noisy_ten = torch.tensor(noisy_tensor, dtype=torch.float32).to(device)
    criterion_l1 = losses.L1Norm()
    obs_metrics = compute_metrics(clean_tensor, noisy_tensor)
    log_print(f"--> Generated observation under Seed 42 (30% Salt & Pepper): PSNR={obs_metrics['psnr']:.2f} dB, SSIM={obs_metrics['ssim']:.4f}")

    # 2. Build AHD-Net Model under Seed 100
    set_seed(100)
    model = build_deeptensor_cp_model(
        rank=rank,
        lengths=(H, W, T),
        u_type='skip1d',
        v_type='skip1d',
        w_type='adaptive_hybrid',
        n_inputs=n_inputs,
        learnable_core=True,
        normalize_core=True,
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

    # 3. Latent Positional Encoding Inputs under Seed 200
    set_seed(200)
    u_inp = utils.get_1d_posencode_inp(H, rank // 2, device=device)
    v_inp = utils.get_1d_posencode_inp(W, rank // 2, device=device)
    w_inp = utils.get_1d_posencode_inp(T, rank // 2, device=device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + [u_inp, v_inp, w_inp],
        lr=lr
    )

    adaptive_block = model.w_net.adaptive_block

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
    log_print("\n--> Starting 1,200-Epoch Optimization Loop...")

    model.train()
    for idx in range(epochs):
        # Cosine temperature annealing
        current_tau = get_cosine_temperature(idx, epochs, tau_start=tau_start, tau_min=tau_min)
        adaptive_block.set_temperature(current_tau)

        # Forward reconstruction
        recon = model(u_inp, v_inp, w_inp)
        loss = criterion_l1(noisy_ten - recon)

        optimizer.zero_grad()
        loss.backward()

        cur_loss = float(loss.item())
        loss_history.append(cur_loss)

        # Telemetry capture every 50 epochs or at epoch 0 / final epoch
        if idx == 0 or (idx + 1) % 50 == 0 or idx == epochs - 1:
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

            if (idx + 1) % 100 == 0 or idx == 0 or idx == epochs - 1:
                mw = router_telem['mean_weights']
                dc = router_telem['dominant_counts']
                log_print(
                    f"  [Epoch {idx + 1:4d}/{epochs}] Loss: {cur_loss:.6f} | Best Loss: {best_loss:.6f} (Ep {best_epoch}) | "
                    f"tau: {current_tau:.2f} | H: {router_telem['router_entropy']:.4f} | "
                    f"Gates: CNN={mw['cnn']:.3f}, PW={mw['pointwise']:.3f}, Attn={mw['attention']:.3f} | "
                    f"Dominance: CNN={dc['cnn']}, PW={dc['pointwise']}, Attn={dc['attention']}"
                )

        # Strictly unsupervised checkpoint selection: lowest measurement L1 loss
        if cur_loss < best_loss:
            best_loss = cur_loss
            best_epoch = idx + 1
            best_recon = recon.detach().cpu().numpy()
            best_model_state = copy.deepcopy(model.state_dict())
            best_latents = {
                'u_inp': u_inp.detach().cpu(),
                'v_inp': v_inp.detach().cpu(),
                'w_inp': w_inp.detach().cpu(),
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
            'u_inp': u_inp.detach().cpu(),
            'v_inp': v_inp.detach().cpu(),
            'w_inp': w_inp.detach().cpu(),
        }
    )
    log_print(f"--> Saved best checkpoint to: {best_ckpt_path} ({os.path.getsize(best_ckpt_path) / (1024**2):.2f} MB)")
    log_print(f"--> Saved final checkpoint to: {final_ckpt_path} ({os.path.getsize(final_ckpt_path) / (1024**2):.2f} MB)")

    # 5. Evaluate Reconstruction Fidelity Metrics (Full Volume)
    # Protocol A: In-loop snapshot captured at best epoch (identical to historical benchmark script)
    metrics_inloop = compute_metrics(clean_tensor, best_recon)

    # Protocol B: Reloaded checkpoint evaluated with saved latents in DIP train mode (prescribed tau)
    model.load_state_dict(best_model_state)
    adaptive_block.set_temperature(tau_best)
    model.train()
    with torch.no_grad():
        recon_dip = model(
            best_latents['u_inp'].to(device),
            best_latents['v_inp'].to(device),
            best_latents['w_inp'].to(device),
        ).cpu().numpy()
    metrics_dip = compute_metrics(clean_tensor, recon_dip)

    # Protocol C: Unit temperature tau=1.0 diagnostic ablation
    adaptive_block.set_temperature(1.0)
    with torch.no_grad():
        recon_tau1 = model(
            best_latents['u_inp'].to(device),
            best_latents['v_inp'].to(device),
            best_latents['w_inp'].to(device),
        ).cpu().numpy()
    metrics_tau1 = compute_metrics(clean_tensor, recon_tau1)
    adaptive_block.set_temperature(tau_best)

    # Protocol D: Eval mode (accumulated running statistics)
    model.eval()
    with torch.no_grad():
        recon_eval = model(
            best_latents['u_inp'].to(device),
            best_latents['v_inp'].to(device),
            best_latents['w_inp'].to(device),
        ).cpu().numpy()
    metrics_eval = compute_metrics(clean_tensor, recon_eval)

    log_print("\n" + "=" * 80)
    log_print("RECONSTRUCTION FIDELITY EVALUATION (Full 192x168x160 Volume)")
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

    # 6. Load Historical Figure 10 Results for Direct Benchmark Comparison
    hist_path = os.path.join(project_root, 'results', 'hybrid', 'figure10_yale_denoising', 'figure10_yale_b_comparison.json')
    with open(hist_path, 'r') as f:
        hist_data = json.load(f)
    hist_models = hist_data['models']

    base_psnr = hist_models['baseline']['metrics']['psnr']

    log_print("\n" + "=" * 80)
    log_print("COMPARATIVE BENCHMARK TABLE: FIGURE 10 YALE FACE B (All 6 Architectures)")
    log_print("=" * 80)
    log_print(f"{'Model Architecture':<22} | {'PSNR (dB)':<10} | {'SSIM':<8} | {'RMSE':<9} | {'Params':<10} | {'Delta vs Base':<12}")
    log_print("-" * 80)

    for m_name in ['baseline', 'hybrid', 'cnn_control', 'pointwise_control', 'transformer']:
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
        'experiment': 'figure10_yale_b_full_budget',
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

    comp_file = os.path.join(out_dir, 'figure10_ahd_net_comparison.json')
    with open(comp_file, 'w') as f:
        json.dump(comparison_summary, f, indent=2)
    log_print(f"\n--> Saved comparison summary to: {comp_file}")

    # 8. Generate Visual Diagnostic Panels
    central_t = T // 2
    gt_slice = clean_tensor[:, :, central_t]
    ahd_slice = recon_dip[:, :, central_t]
    noisy_slice = noisy_tensor[:, :, central_t]
    err_map = np.abs(gt_slice - ahd_slice)

    # Panel 1: Reconstruction & Error Map
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(gt_slice, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f"Ground Truth (Slice {central_t})")
    axes[0].axis('off')

    axes[1].imshow(noisy_slice, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f"30% Salt & Pepper Input\nPSNR: {obs_metrics['psnr']:.2f} dB")
    axes[1].axis('off')

    axes[2].imshow(ahd_slice, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title(f"AHD-Net v1 Recon\nFull PSNR: {metrics_dip['psnr']:.2f} dB | SSIM: {metrics_dip['ssim']:.4f}")
    axes[2].axis('off')

    im3 = axes[3].imshow(err_map, cmap='magma', vmin=0, vmax=0.15)
    axes[3].set_title(f"Absolute Error Map\nMean Error: {np.mean(err_map):.5f}")
    axes[3].axis('off')
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "figure10_ahd_net_reconstruction.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_print(f"--> Saved reconstruction panel to: {fig_path}")

    # Panel 2: Router Trajectory Plot
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
    ax1.set_title('AHD-Net v1 Router Trajectory: Figure 10 Yale Face B (1200 Epochs)')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_log, ent_log, label='Router Shannon Entropy', color='tab:purple', linewidth=2)
    ax2.axhline(math.log(3.0), color='gray', linestyle='--', alpha=0.7, label='Max Entropy ln(3) ≈ 1.0986')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Entropy (nats)')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    traj_path = os.path.join(out_dir, "figure10_router_trajectory.png")
    plt.savefig(traj_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_print(f"--> Saved router trajectory plot to: {traj_path}")

    # Panel 3: Comprehensive Multi-Model Benchmark Panel
    fig, axes = plt.subplots(1, 6, figsize=(24, 4))
    axes[0].imshow(gt_slice, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f"Ground Truth\nSlice {central_t}")
    axes[0].axis('off')

    axes[1].imshow(ahd_slice, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f"AHD-Net v1 (Ours)\n{metrics_dip['psnr']:.2f} dB | {metrics_dip['ssim']:.4f}")
    axes[1].axis('off')

    axes[2].imshow(err_map, cmap='magma', vmin=0, vmax=0.15)
    axes[2].set_title(f"AHD-Net Error Map\nMAE: {np.mean(err_map):.5f}")
    axes[2].axis('off')

    # Bar chart of PSNRs across all 6 models
    model_names = ['Baseline', 'Hybrid', 'CNN Ctrl', 'PW Ctrl', 'Transf.', 'AHD-Net']
    psnrs = [
        hist_models['baseline']['metrics']['psnr'],
        hist_models['hybrid']['metrics']['psnr'],
        hist_models['cnn_control']['metrics']['psnr'],
        hist_models['pointwise_control']['metrics']['psnr'],
        hist_models['transformer']['metrics']['psnr'],
        metrics_dip['psnr'],
    ]
    colors = ['#1f77b4', '#ff7f0e', '#aec7e8', '#ffbb78', '#d62728', '#2ca02c']

    ax_bar = axes[3]
    bars = ax_bar.bar(model_names, psnrs, color=colors)
    ax_bar.set_ylim(26.0, 30.0)
    ax_bar.set_ylabel('PSNR (dB)')
    ax_bar.set_title('Figure 10 Model Comparison')
    ax_bar.tick_params(axis='x', rotation=45)
    for bar in bars:
        yval = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f"{yval:.2f}", ha='center', va='bottom', fontsize=8)

    # Loss convergence curve
    axes[4].plot(loss_history, color='blue', linewidth=1.5)
    axes[4].set_yscale('log')
    axes[4].set_xlabel('Epoch')
    axes[4].set_ylabel('L1 Measurement Loss (log)')
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
    comp_panel_path = os.path.join(out_dir, "figure10_full_benchmark_panel.png")
    plt.savefig(comp_panel_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_print(f"--> Saved full benchmark panel to: {comp_panel_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="AHD-Net Figure 10 Full-Budget Benchmark")
    parser.add_argument('--epochs', type=int, default=1200, help="Training epochs (default: 1200)")
    args = parser.parse_args()
    run_benchmark(epochs=args.epochs)
