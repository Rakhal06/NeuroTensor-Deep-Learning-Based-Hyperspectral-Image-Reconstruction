"""
Full-Budget Benchmark Execution Script for AHD-Net v1 on Figure 12
===================================================================
Application: Figure 12 — 3D PET/CT Radon Compressive Sensing (40 Angles)
Decomposition: 3-Way CP/PARAFAC (Rank R=1000, 128x128x56)
Physics Forward Operator: 40-Angle 2D Radon Transform (0° to 180°)
Noise: Mixed Poisson (tau=100.0) + Gaussian Readout (SNR=2.0), Observation Seed 42
Latent Initialization: Seed 200 (Positional Encodings)
Training Budget: 4,000 Epochs
Temperature Schedule: Cosine Decay (2.0 -> 0.5)
Target Parameters: 4,050,988 (Delta: 0)
Supervision: 100% Unsupervised Measurement L2 (Frobenius) Sinogram Loss
Output: results/extensions/adaptive_hybrid_v1_full/figure12_pet_ct/
"""

import os
import sys
import copy
import time
import math
import json
import random
import warnings
warnings.filterwarnings('ignore')
from typing import Dict, Any, List

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
from deeptensor.physics import operators as lin_inverse
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
                'cnn': float(mean_weights[0]),
                'pointwise': float(mean_weights[1]),
                'attention': float(mean_weights[2]),
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


def run_benchmark(epochs: int = 4000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rank = 1000
    n_inputs = rank
    lr = 1e-3
    nangles = 40
    reg_noise_std = 1.0 / 30.0
    tau_start = 2.0
    tau_min = 0.5
    target_params = 4050988

    out_dir = os.path.join(project_root, 'results', 'extensions', 'adaptive_hybrid_v1_full', 'figure12_pet_ct')
    os.makedirs(out_dir, exist_ok=True)
    log_file = os.path.join(out_dir, "training.log")

    def log_print(msg: str):
        print(msg)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    # Clear log file
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"=== AHD-Net Figure 12 PET/CT Full-Budget Benchmark ({epochs} Epochs) ===\n")

    log_print("=" * 80)
    log_print("AHD-NET V1 FULL-BUDGET BENCHMARK: FIGURE 12 3D PET/CT RADON COMPRESSIVE SENSING")
    log_print(f"Device: {device} ({torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'CPU'})")
    log_print(f"Budget: {epochs} epochs | Rank: R={rank} | LR: {lr} | Angles: {nangles}")
    log_print(f"Noise: Mixed Poisson (tau=100.0) + Gaussian (SNR=2.0) | Seed: 42")
    log_print(f"Temperature Schedule: Cosine Decay {tau_start} -> {tau_min}")
    log_print(f"Output Directory: {out_dir}")
    log_print("=" * 80)

    # 1. Load PET/CT Dataset
    data_path = os.path.join(project_root, 'data', 'pet.mat')
    if not os.path.exists(data_path):
        data_path = os.path.join(project_root, 'data', 'PET.mat')
    data = io.loadmat(data_path)
    cube = data.get('hypercube', data.get('PET', data.get('data', None)))
    if cube is None:
        for k in data:
            if not k.startswith('__') and isinstance(data[k], np.ndarray) and data[k].ndim == 3:
                cube = data[k]
                break

    cube = cube.astype(np.float32)
    cube = cube / cube.max()
    H, W, T = cube.shape
    log_print(f"--> Loaded PET/CT dataset: ({H}, {W}, {T}) (H={H}, W={W}, Axial T={T})")

    # 2. Physics Forward Model: 40-Angle 2D Radon Projection + Detector Noise
    set_seed(42)
    angles = torch.tensor(np.linspace(0, 180, nangles).astype(np.float32), device=device)
    cube_ten = torch.tensor(cube, dtype=torch.float32, device=device).permute(2, 0, 1)[None, ...]  # (1, 56, 128, 128)
    clean_measurements = lin_inverse.radon(cube_ten, angles).detach().cpu().numpy()  # (40, 56, 128)
    noisy_measurements = utils.measure(clean_measurements, 2.0, 100.0)
    meas_ten = torch.tensor(noisy_measurements, dtype=torch.float32, device=device)
    criterion_l2 = losses.L2Norm()

    # Algorithmic initial inversion for baseline observation comparison
    log_print("--> Computing algorithmic initial inversion for baseline observation...")
    rec_alg = utils.get_inp([H, W, T], device=device)
    opt_alg = torch.optim.Adam([rec_alg], lr=0.1)
    for _ in range(100):
        m_est_alg = lin_inverse.radon(rec_alg[None, ...].permute(0, 3, 1, 2), angles)
        l_alg = criterion_l2(meas_ten - m_est_alg)
        opt_alg.zero_grad()
        l_alg.backward()
        opt_alg.step()
    cube_obs = rec_alg.detach().cpu().numpy()
    obs_metrics = compute_metrics(cube, cube_obs)
    log_print(f"--> Degraded Algorithmic Observation: PSNR={obs_metrics['psnr']:.2f} dB, SSIM={obs_metrics['ssim']:.4f}")

    # 3. Build AHD-Net Model under Seed 100
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
    assert delta_params == 0, f"Parameter mismatch! Expected {target_params}, got {total_params}"

    adaptive_block = model.w_net.adaptive_block

    # 4. Latent Initialization under Seed 200
    set_seed(200)
    u_inp = utils.get_1d_posencode_inp(H, rank // 2, device=device)
    v_inp = utils.get_1d_posencode_inp(W, rank // 2, device=device)
    w_inp = utils.get_1d_posencode_inp(T, rank // 2, device=device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + [u_inp, v_inp, w_inp],
        lr=lr
    )

    u_inp_per = u_inp.detach().clone()
    v_inp_per = v_inp.detach().clone()
    w_inp_per = w_inp.detach().clone()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    # 5. Training Loop
    best_loss = float('inf')
    best_epoch = 0
    best_state_dict = None
    best_u = None
    best_v = None
    best_w = None
    best_inloop_recon = None
    loss_history = []
    telemetry_history = []

    model.train()
    start_train_time = time.time()
    log_print("\n--> Starting AHD-Net training (4,000 epochs)...")

    for epoch in range(1, epochs + 1):
        # Update routing temperature via cosine schedule
        tau_cur = get_cosine_temperature(epoch, epochs, tau_start, tau_min)
        adaptive_block.set_temperature(tau_cur)

        # In-loop perturbation (standard DIP training protocol)
        u_pert = u_inp + u_inp_per.normal_() * reg_noise_std
        v_pert = v_inp + v_inp_per.normal_() * reg_noise_std
        w_pert = w_inp + w_inp_per.normal_() * reg_noise_std

        # Forward pass: 3D tensor reconstruction + 40-angle Radon projection
        tensor_estim = model(u_pert, v_pert, w_pert)
        m_est = lin_inverse.radon(tensor_estim[None, ...].permute(0, 3, 1, 2), angles)

        loss = criterion_l2(meas_ten - m_est)
        cur_loss = float(loss.item())
        loss_history.append(cur_loss)

        optimizer.zero_grad()
        loss.backward()

        # Telemetry recording before optimizer step
        if epoch % 100 == 0 or epoch == 1 or epoch == epochs:
            g_norms = compute_branch_gradient_norms(adaptive_block)
            telem = extract_router_telemetry(adaptive_block)
            telem['epoch'] = epoch
            telem['loss'] = cur_loss
            telem['tau'] = tau_cur
            telem['gradient_norms'] = g_norms
            telemetry_history.append(telem)

        optimizer.step()

        # Strict Checkpoint Selection: GOVERNED 100% BY MEASUREMENT L2 LOSS (Zero GT Leakage)
        if cur_loss < best_loss:
            best_loss = cur_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            best_u = u_inp.detach().clone()
            best_v = v_inp.detach().clone()
            best_w = w_inp.detach().clone()
            best_inloop_recon = tensor_estim.detach().cpu().numpy()

        if epoch % 200 == 0 or epoch == 1 or epoch == epochs:
            mw = adaptive_block.last_gates.mean(dim=(0, 1)).cpu().tolist() if adaptive_block.last_gates is not None else [0.33, 0.33, 0.33]
            ent = float(adaptive_block.compute_router_entropy().item())
            elapsed = time.time() - start_train_time
            log_print(
                f"  Epoch {epoch:5d}/{epochs} | Meas L2 Loss: {cur_loss:.5e} | "
                f"Best: {best_loss:.5e} (ep {best_epoch}) | tau: {tau_cur:.3f} | "
                f"H: {ent:.4f} | Weights (CNN/PW/Attn): {mw[0]:.3f}/{mw[1]:.3f}/{mw[2]:.3f} | "
                f"Elapsed: {elapsed:.1f}s"
            )

    total_train_time = time.time() - start_train_time
    peak_vram = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if torch.cuda.is_available() else 0.0

    log_print("\n" + "=" * 80)
    log_print("TRAINING COMPLETE: SAVING CHECKPOINTS & EXECUTING SCIENTIFIC EVALUATION")
    log_print(f"Total Runtime: {total_train_time:.2f}s ({total_train_time/60.0:.2f} min) | Peak VRAM: {peak_vram:.1f} MB")
    log_print(f"Best Measurement L2 Loss: {best_loss:.6e} at Epoch {best_epoch}")
    log_print("=" * 80)

    # Save best checkpoint
    best_ckpt_path = os.path.join(out_dir, "best_checkpoint.pth")
    tau_best = get_cosine_temperature(best_epoch, epochs, tau_start, tau_min)
    save_checkpoint(
        model,
        best_ckpt_path,
        optimizer=optimizer,
        epoch=best_epoch,
        metrics={'loss': best_loss, 'best_loss': best_loss, 'total_time': total_train_time, 'peak_vram_mb': peak_vram},
        config={
            'tau': tau_best,
            'u_inp': best_u.cpu(),
            'v_inp': best_v.cpu(),
            'w_inp': best_w.cpu(),
            'total_params': total_params,
            'rank': rank,
            'nangles': nangles,
            'H': H,
            'W': W,
            'T': T,
        }
    )
    # Save final checkpoint
    final_ckpt_path = os.path.join(out_dir, "final_checkpoint.pth")
    save_checkpoint(
        model,
        final_ckpt_path,
        optimizer=optimizer,
        epoch=epochs,
        metrics={'loss': cur_loss, 'best_loss': best_loss, 'total_time': total_train_time, 'peak_vram_mb': peak_vram},
        config={
            'tau': tau_cur,
            'u_inp': u_inp.detach().cpu(),
            'v_inp': v_inp.detach().cpu(),
            'w_inp': w_inp.detach().cpu(),
            'total_params': total_params,
            'rank': rank,
            'nangles': nangles,
            'H': H,
            'W': W,
            'T': T,
        }
    )
    log_print(f"--> Saved best checkpoint:  {best_ckpt_path}")
    log_print(f"--> Saved final checkpoint: {final_ckpt_path}")

    # 6. Comprehensive Post-Training Evaluation Protocols
    log_print("\n" + "=" * 80)
    log_print("PROTOCOL-DISAMBIGUATED EVALUATION ON PET/CT GROUND TRUTH")
    log_print("=" * 80)

    # Protocol A: Comparable In-Loop Snapshot (Best Epoch)
    metrics_inloop = compute_metrics(cube, best_inloop_recon)

    # Load frozen best checkpoint into fresh model for inference protocols
    eval_model = build_deeptensor_cp_model(
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
        tau=tau_best,
    ).to(device)
    eval_model.load_state_dict(best_state_dict)

    # Protocol B: Clean-Latent DIP Inference Protocol (tau = 0.5000, model.train(), torch.no_grad())
    eval_model.train()
    eval_model.w_net.adaptive_block.set_temperature(0.5)
    with torch.no_grad():
        recon_dip_ten = eval_model(best_u, best_v, best_w)
        recon_dip = recon_dip_ten.detach().cpu().numpy()
    metrics_dip = compute_metrics(cube, recon_dip)

    # Protocol C: Diagnostic Unit Temperature (tau = 1.0000, clean latents)
    eval_model.w_net.adaptive_block.set_temperature(1.0)
    with torch.no_grad():
        recon_tau1_ten = eval_model(best_u, best_v, best_w)
        recon_tau1 = recon_tau1_ten.detach().cpu().numpy()
    metrics_tau1 = compute_metrics(cube, recon_tau1)

    # Protocol D: Eval Mode Running Statistics (model.eval())
    eval_model.eval()
    eval_model.w_net.adaptive_block.set_temperature(0.5)
    with torch.no_grad():
        recon_eval_ten = eval_model(best_u, best_v, best_w)
        recon_eval = recon_eval_ten.detach().cpu().numpy()
    metrics_eval = compute_metrics(cube, recon_eval)

    # Central Slice 28 Disambiguation
    slice_idx = T // 2  # Slice 28
    gt_slice = cube[:, :, slice_idx]
    recon_slice = recon_dip[:, :, slice_idx]
    slice_mse = float(np.mean((gt_slice - recon_slice) ** 2))
    slice_global_psnr = float(10.0 * np.log10(1.0 / (slice_mse + 1e-12)))
    slice_ptp = float(gt_slice.max() - gt_slice.min())
    slice_local_psnr = float(10.0 * np.log10((slice_ptp ** 2) / (slice_mse + 1e-12)))

    log_print(f"Protocol A (Comparable In-Loop Snapshot at Epoch {best_epoch}):")
    log_print(f"  PSNR (Full-Volume)       : {metrics_inloop['psnr']:.4f} dB")
    log_print(f"  SSIM                     : {metrics_inloop['ssim']:.4f}")
    log_print(f"  RMSE                     : {metrics_inloop['rmse']:.6f}")
    log_print(f"  MAE                      : {metrics_inloop['mae']:.6f}")
    log_print(f"  Relative Error           : {metrics_inloop['rel_error']:.6f}")
    log_print("-" * 80)
    log_print(f"Protocol B (Clean-Latent DIP Inference Protocol, tau=0.5000):")
    log_print(f"  PSNR (Full-Volume)       : {metrics_dip['psnr']:.4f} dB")
    log_print(f"  SSIM                     : {metrics_dip['ssim']:.4f}")
    log_print(f"  RMSE                     : {metrics_dip['rmse']:.6f}")
    log_print(f"  MAE                      : {metrics_dip['mae']:.6f}")
    log_print(f"  Relative Error           : {metrics_dip['rel_error']:.6f}")
    log_print(f"  Central Slice 28 (Global): {slice_global_psnr:.4f} dB (ptp=1.0000)")
    log_print(f"  Central Slice 28 (Local) : {slice_local_psnr:.4f} dB (ptp={slice_ptp:.4f})")
    log_print("-" * 80)
    log_print(f"Protocol C (Diagnostic Unit Temperature tau=1.0000):")
    log_print(f"  PSNR (Full-Volume)       : {metrics_tau1['psnr']:.4f} dB")
    log_print(f"  SSIM                     : {metrics_tau1['ssim']:.4f}")
    log_print("-" * 80)
    log_print(f"Protocol D (Eval Mode Running Statistics):")
    log_print(f"  PSNR (Full-Volume)       : {metrics_eval['psnr']:.4f} dB (BN discrepancy: -{metrics_dip['psnr'] - metrics_eval['psnr']:.2f} dB)")
    log_print(f"  SSIM                     : {metrics_eval['ssim']:.4f}")
    log_print("=" * 80)

    # 7. Load Historical Figure 12 Results for Direct Benchmark Comparison
    hist_path = os.path.join(project_root, 'results', 'hybrid', 'figure12_pet_ct', 'figure12_pet_ct_comparison.json')
    with open(hist_path, 'r') as f:
        hist_data = json.load(f)
    hist_models = hist_data['models']

    base_psnr = hist_models['baseline']['metrics']['psnr']

    log_print("\n" + "=" * 80)
    log_print("COMPARATIVE BENCHMARK TABLE: FIGURE 12 3D PET/CT RADON CS (All 6 Architectures)")
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

    # 8. Save Comparison Summary JSON
    comparison_summary = {
        'experiment': 'figure12_pet_ct_full_budget',
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

    comp_file = os.path.join(out_dir, 'figure12_ahd_net_comparison.json')
    with open(comp_file, 'w') as f:
        json.dump(comparison_summary, f, indent=2)
    log_print(f"\n--> Saved comparison summary to: {comp_file}")

    # 9. Generate Visual Diagnostic Panels
    err_map = np.abs(gt_slice - recon_slice)

    # Panel 1: Reconstruction & Error Map
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(gt_slice, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f"Ground Truth (Slice {slice_idx})")
    axes[0].axis('off')

    axes[1].imshow(recon_slice, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f"AHD-Net v1 Recon\nFull PSNR: {metrics_dip['psnr']:.2f} dB | SSIM: {metrics_dip['ssim']:.4f}")
    axes[1].axis('off')

    im2 = axes[2].imshow(err_map, cmap='inferno', vmin=0, vmax=0.10)
    axes[2].set_title(f"Absolute Error Map\nMean Error: {np.mean(err_map):.5f}")
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "figure12_ahd_net_reconstruction.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_print(f"--> Saved reconstruction panel to: {fig_path}")

    # Panel 2: Axial Slice Strip (Slices 7, 14, 21, 28, 35, 42, 49)
    sample_slices = [7, 14, 21, 28, 35, 42, 49]
    fig, axes = plt.subplots(2, len(sample_slices), figsize=(20, 6))
    for col_idx, s_idx in enumerate(sample_slices):
        axes[0, col_idx].imshow(cube[:, :, s_idx], cmap='gray', vmin=0, vmax=1)
        axes[0, col_idx].set_title(f"GT Slice {s_idx}", fontsize=10, fontweight='bold')
        axes[0, col_idx].axis('off')

        axes[1, col_idx].imshow(recon_dip[:, :, s_idx], cmap='gray', vmin=0, vmax=1)
        axes[1, col_idx].set_title(f"AHD-Net Slice {s_idx}", fontsize=10)
        axes[1, col_idx].axis('off')

    plt.tight_layout()
    strip_path = os.path.join(out_dir, "figure12_slice_strip.png")
    plt.savefig(strip_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_print(f"--> Saved axial slice strip to: {strip_path}")

    # Panel 3: Router Trajectory Plot
    epochs_log = [e['epoch'] for e in telemetry_history]
    cnn_log = [e['mean_weights']['cnn'] for e in telemetry_history]
    pw_log = [e['mean_weights']['pointwise'] for e in telemetry_history]
    attn_log = [e['mean_weights']['attention'] for e in telemetry_history]
    ent_log = [e['router_entropy'] for e in telemetry_history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.plot(epochs_log, cnn_log, label='CNN Branch (Local)', color='tab:blue', linewidth=2)
    ax1.plot(epochs_log, pw_log, label='Pointwise Branch (Inter-Channel/Rank)', color='tab:green', linewidth=2)
    ax1.plot(epochs_log, attn_log, label='Attention Branch (Global)', color='tab:red', linewidth=2)
    ax1.axhline(1/3, color='gray', linestyle='--', alpha=0.7, label='Uniform (1/3)')
    ax1.set_ylabel('Mean Routing Weight')
    ax1.set_title(f'AHD-Net v1 Router Trajectory: Figure 12 3D PET/CT Radon CS ({epochs} Epochs)')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_log, ent_log, label='Router Shannon Entropy', color='tab:purple', linewidth=2)
    ax2.axhline(math.log(3.0), color='gray', linestyle='--', alpha=0.7, label='Max Entropy ln(3) ≈ 1.0986')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Entropy (nats)')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    traj_path = os.path.join(out_dir, "figure12_router_trajectory.png")
    plt.savefig(traj_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_print(f"--> Saved router trajectory plot to: {traj_path}")

    # Panel 4: Comprehensive Multi-Model Benchmark Panel
    fig, axes = plt.subplots(1, 6, figsize=(24, 4))
    axes[0].imshow(gt_slice, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f"Ground Truth\nSlice {slice_idx}")
    axes[0].axis('off')

    axes[1].imshow(recon_slice, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f"AHD-Net v1 (Ours)\n{metrics_dip['psnr']:.2f} dB | {metrics_dip['ssim']:.4f}")
    axes[1].axis('off')

    axes[2].imshow(err_map, cmap='inferno', vmin=0, vmax=0.10)
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
    colors = ['#1f77b4', '#ff7f0e', '#aec7e8', '#2ca02c', '#d62728', '#9467bd']

    ax_bar = axes[3]
    bars = ax_bar.bar(model_names, psnrs, color=colors)
    ax_bar.set_ylim(29.0, 31.5)
    ax_bar.set_ylabel('PSNR (dB)')
    ax_bar.set_title('Figure 12 Model Comparison')
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
        colors=['#1f77b4', '#2ca02c', '#d62728'],
        autopct='%1.1f%%',
        startangle=90
    )
    axes[5].set_title(f"Channel Gating Breakdown\nEntropy: {final_router_telem.get('router_entropy', 1.09):.4f}")

    plt.tight_layout()
    bench_path = os.path.join(out_dir, "figure12_full_benchmark_panel.png")
    plt.savefig(bench_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_print(f"--> Saved multi-model benchmark panel to: {bench_path}")

    log_print("\n" + "=" * 80)
    log_print("FIGURE 12 AHD-NET V1 FULL-BUDGET BENCHMARK COMPLETE!")
    log_print("=" * 80)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Full-Budget Benchmark for AHD-Net on Figure 12 PET/CT")
    parser.add_argument('--epochs', type=int, default=4000, help="Number of training epochs (default: 4000)")
    args = parser.parse_args()
    run_benchmark(epochs=args.epochs)
