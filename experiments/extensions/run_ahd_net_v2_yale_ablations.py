"""
AHD-Net v2 Controlled Ablation Study: Figure 10 Extended Yale Face B
===================================================================
Diagnostic Testbed: Extended Yale Face B (192 x 168 x 160, Rank R=2000, 30% Salt & Pepper)
Training Budget: 1,200 Epochs | Seed: Obs 42, Latents 200, Model 100 | L1 Unsupervised Loss

Phase 1 Conditions:
  1. cond1_cnn_only: Clamped CNN branch (w = [1, 0, 0])
  2. cond2_pointwise_only: Clamped Pointwise branch (w = [0, 1, 0])
  3. cond3_attention_only: Clamped Attention branch (w = [0, 0, 1])
  4. cond4_static_equal: Fixed equal tri-branch ensemble (w = [1/3, 1/3, 1/3])
  5. cond5_standard_cagc: Standard AHD-Net v1 reference (tau: 2.0 -> 0.5)

Phase 2 Conditions (Routing Interventions):
  6a. cond6a_temp_sharp: Sharp anneal (tau: 2.0 -> 0.1)
  6b. cond6b_temp_soft: Soft anneal (tau: 2.0 -> 1.0)
  6c. cond6c_temp_const1: Constant unit temperature (tau = 1.0)
  7. cond7_gating_floor15: Gating floor constraint (w_b >= 0.15, tau: 2.0 -> 0.5)
  8. cond8_entropy_reg: Entropy regularization (lambda = 0.01, tau: 2.0 -> 0.5)
"""

import os
import sys
import copy
import time
import math
import json
import random
import argparse
from typing import Dict, Any, List, Optional

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


# Condition configurations definition
CONDITIONS = {
    'cond1_cnn_only': {
        'name': 'CNN-Only Tri-Branch',
        'phase': 1,
        'router_type': 'cnn_only',
        'gating_floor': 0.0,
        'tau_schedule': 'constant',
        'tau_start': 1.0,
        'tau_min': 1.0,
        'entropy_reg': 0.0,
        'description': 'Clamped CNN branch (w = [1, 0, 0]) inside tri-branch block',
    },
    'cond2_pointwise_only': {
        'name': 'Pointwise-Only Tri-Branch',
        'phase': 1,
        'router_type': 'pointwise_only',
        'gating_floor': 0.0,
        'tau_schedule': 'constant',
        'tau_start': 1.0,
        'tau_min': 1.0,
        'entropy_reg': 0.0,
        'description': 'Clamped Pointwise branch (w = [0, 1, 0]) inside tri-branch block',
    },
    'cond3_attention_only': {
        'name': 'Attention-Only Tri-Branch',
        'phase': 1,
        'router_type': 'attention_only',
        'gating_floor': 0.0,
        'tau_schedule': 'constant',
        'tau_start': 1.0,
        'tau_min': 1.0,
        'entropy_reg': 0.0,
        'description': 'Clamped Attention branch (w = [0, 0, 1]) inside tri-branch block',
    },
    'cond4_static_equal': {
        'name': 'Static Equal Tri-Branch',
        'phase': 1,
        'router_type': 'static',
        'gating_floor': 0.0,
        'tau_schedule': 'constant',
        'tau_start': 1.0,
        'tau_min': 1.0,
        'entropy_reg': 0.0,
        'description': 'Fixed uniform ensemble (w = [1/3, 1/3, 1/3]) without learned router',
    },
    'cond5_standard_cagc': {
        'name': 'Standard AHD-Net v1 Reference',
        'phase': 1,
        'router_type': 'cagc',
        'gating_floor': 0.0,
        'tau_schedule': 'cosine',
        'tau_start': 2.0,
        'tau_min': 0.5,
        'entropy_reg': 0.0,
        'description': 'Standard CAGC router with cosine temperature decay (2.0 -> 0.5)',
    },
    'cond6a_temp_sharp': {
        'name': 'Temperature Sharpening',
        'phase': 2,
        'router_type': 'cagc',
        'gating_floor': 0.0,
        'tau_schedule': 'cosine',
        'tau_start': 2.0,
        'tau_min': 0.1,
        'entropy_reg': 0.0,
        'description': 'Aggressive cosine temperature sharpening (2.0 -> 0.1)',
    },
    'cond6b_temp_soft': {
        'name': 'Conservative Temperature',
        'phase': 2,
        'router_type': 'cagc',
        'gating_floor': 0.0,
        'tau_schedule': 'cosine',
        'tau_start': 2.0,
        'tau_min': 1.0,
        'entropy_reg': 0.0,
        'description': 'Conservative cosine temperature decay (2.0 -> 1.0)',
    },
    'cond6c_temp_const1': {
        'name': 'Constant Temperature',
        'phase': 2,
        'router_type': 'cagc',
        'gating_floor': 0.0,
        'tau_schedule': 'constant',
        'tau_start': 1.0,
        'tau_min': 1.0,
        'entropy_reg': 0.0,
        'description': 'Fixed constant unit temperature (tau = 1.0, no decay)',
    },
    'cond7_gating_floor15': {
        'name': 'Gating Floor Constraint (0.15)',
        'phase': 2,
        'router_type': 'cagc',
        'gating_floor': 0.15,
        'tau_schedule': 'cosine',
        'tau_start': 2.0,
        'tau_min': 0.5,
        'entropy_reg': 0.0,
        'description': 'CAGC with w_b >= 0.15 minimum gating floor to prevent attention starvation',
    },
    'cond8_entropy_reg': {
        'name': 'Entropy Regularization (0.01)',
        'phase': 2,
        'router_type': 'cagc',
        'gating_floor': 0.0,
        'tau_schedule': 'cosine',
        'tau_start': 2.0,
        'tau_min': 0.5,
        'entropy_reg': 0.01,
        'description': 'CAGC with entropy maximization loss penalty (lambda = 0.01)',
    },
}


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


def compute_tau(step: int, max_steps: int, schedule: str, tau_start: float, tau_min: float) -> float:
    if schedule == 'constant':
        return tau_start
    elif schedule == 'cosine':
        progress = min(max(float(step) / float(max_steps), 0.0), 1.0)
        return tau_min + 0.5 * (tau_start - tau_min) * (1.0 + math.cos(math.pi * progress))
    else:
        return tau_start


def extract_router_telemetry(adaptive_block: AdaptiveTriBranchBlock) -> Dict[str, Any]:
    with torch.no_grad():
        gates = adaptive_block.last_gates
        if gates is None:
            return {
                'mean_weights': {'cnn': 1.0/3.0, 'pointwise': 1.0/3.0, 'attention': 1.0/3.0},
                'router_entropy': math.log(3.0),
                'channel_entropy': math.log(3.0),
                'current_tau': float(adaptive_block.tau),
                'dominant_counts': {'cnn': 0, 'pointwise': 0, 'attention': 0},
            }
        mean_weights = gates.mean(dim=(0, 1)).cpu().tolist()
        router_entropy = float(adaptive_block.compute_router_entropy().item())
        channel_entropy = float(adaptive_block.compute_channel_entropy().item())
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
            'router_entropy': router_entropy,
            'channel_entropy': channel_entropy,
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
    ]:
        norms = [torch.norm(p.grad) for p in branch.parameters() if p.grad is not None]
        if norms:
            g_norms[name] = float(torch.norm(torch.stack(norms)).item())
        else:
            g_norms[name] = 0.0

    if adaptive_block.router is not None:
        r_norms = [torch.norm(p.grad) for p in adaptive_block.router.parameters() if p.grad is not None]
        g_norms['router'] = float(torch.norm(torch.stack(r_norms)).item()) if r_norms else 0.0
    else:
        g_norms['router'] = 0.0

    return g_norms


def run_ablation_condition(
    cond_id: str,
    cond_cfg: Dict[str, Any],
    clean_tensor: np.ndarray,
    noisy_tensor: np.ndarray,
    base_out_dir: str,
    device: torch.device,
    epochs: int = 1200,
    rank: int = 2000,
    lr: float = 1e-3,
) -> Dict[str, Any]:
    out_dir = os.path.join(base_out_dir, cond_id)
    os.makedirs(out_dir, exist_ok=True)
    log_file_path = os.path.join(out_dir, 'run_log.txt')
    log_f = open(log_file_path, 'w', encoding='utf-8')

    def log_print(msg: str):
        print(msg)
        log_f.write(msg + '\n')
        log_f.flush()

    log_print("=" * 80)
    log_print(f"RUNNING ABLATION CONDITION: {cond_id} — {cond_cfg['name']}")
    log_print(f"Description: {cond_cfg['description']}")
    log_print(f"Router Type: {cond_cfg['router_type']} | Gating Floor: {cond_cfg['gating_floor']}")
    log_print(f"Temperature Schedule: {cond_cfg['tau_schedule']} ({cond_cfg['tau_start']} -> {cond_cfg['tau_min']})")
    log_print(f"Entropy Regularization: lambda = {cond_cfg['entropy_reg']}")
    log_print(f"Epochs: {epochs} | Rank: {rank} | Device: {device}")
    log_print("=" * 80)

    H, W, T = clean_tensor.shape
    noisy_ten = torch.tensor(noisy_tensor, dtype=torch.float32).to(device)
    criterion_l1 = losses.L1Norm()

    # 1. Build Model under Seed 100
    set_seed(100)
    model = build_deeptensor_cp_model(
        rank=rank,
        lengths=(H, W, T),
        u_type='skip1d',
        v_type='skip1d',
        w_type='adaptive_hybrid',
        n_inputs=rank,
        learnable_core=True,
        normalize_core=True,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ffn=512,
        router_type=cond_cfg['router_type'],
        pe_type='sin_pe',
        tau=cond_cfg['tau_start'],
        gating_floor=cond_cfg['gating_floor'],
    ).to(device)

    total_params = count_parameters(model)
    adaptive_block = model.w_net.adaptive_block
    block_params = count_parameters(adaptive_block)
    log_print(f"--> Trainable Parameters: Model={total_params:,} | AdaptiveBlock={block_params:,}")

    # 2. Latent Positional Encoding Inputs under Seed 200
    set_seed(200)
    u_inp = utils.get_1d_posencode_inp(H, rank // 2, device=device)
    v_inp = utils.get_1d_posencode_inp(W, rank // 2, device=device)
    w_inp = utils.get_1d_posencode_inp(T, rank // 2, device=device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + [u_inp, v_inp, w_inp],
        lr=lr
    )

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
    model.train()

    for idx in range(epochs):
        # Update temperature
        cur_tau = compute_tau(
            idx, epochs,
            cond_cfg['tau_schedule'],
            cond_cfg['tau_start'],
            cond_cfg['tau_min'],
        )
        adaptive_block.set_temperature(cur_tau)

        # Forward pass
        recon = model(u_inp, v_inp, w_inp)
        loss_meas = criterion_l1(noisy_ten - recon)

        # Optional Entropy Regularization
        if cond_cfg['entropy_reg'] > 0.0 and cond_cfg['router_type'] == 'cagc':
            ent = adaptive_block.compute_channel_entropy()
            loss = loss_meas - cond_cfg['entropy_reg'] * ent
        else:
            loss = loss_meas

        optimizer.zero_grad()
        loss.backward()

        cur_meas_loss = float(loss_meas.item())
        loss_history.append(cur_meas_loss)

        # Telemetry capture
        if idx == 0 or (idx + 1) % 50 == 0 or idx == epochs - 1:
            grad_norms = compute_branch_gradient_norms(adaptive_block)
            router_telem = extract_router_telemetry(adaptive_block)
            telem_entry = {
                'epoch': idx + 1,
                'meas_loss': cur_meas_loss,
                'total_loss': float(loss.item()),
                'tau': cur_tau,
                'router_entropy': router_telem['router_entropy'],
                'channel_entropy': router_telem['channel_entropy'],
                'mean_weights': router_telem['mean_weights'],
                'dominant_counts': router_telem['dominant_counts'],
                'gradient_norms': grad_norms,
            }
            telemetry_history.append(telem_entry)

            if (idx + 1) % 200 == 0 or idx == 0 or idx == epochs - 1:
                mw = router_telem['mean_weights']
                log_print(
                    f"  [Epoch {idx + 1:4d}/{epochs}] Loss: {cur_meas_loss:.6f} | "
                    f"tau: {cur_tau:.2f} | H_chan: {router_telem['channel_entropy']:.4f} | "
                    f"Gates: CNN={mw['cnn']:.3f}, PW={mw['pointwise']:.3f}, Attn={mw['attention']:.3f}"
                )

        # Strictly unsupervised checkpoint selection: lowest measurement L1 loss
        if cur_meas_loss < best_loss:
            best_loss = cur_meas_loss
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

    log_print(f"--> Optimization Complete: Runtime={total_train_time:.1f}s | Best Meas Loss={best_loss:.6f} at Ep {best_epoch}")

    # 3. Save Checkpoints
    best_ckpt_path = os.path.join(out_dir, "best_checkpoint.pth")
    final_ckpt_path = os.path.join(out_dir, "final_checkpoint.pth")
    tau_best = compute_tau(best_epoch - 1, epochs, cond_cfg['tau_schedule'], cond_cfg['tau_start'], cond_cfg['tau_min'])

    save_checkpoint(
        model,
        best_ckpt_path,
        optimizer=optimizer,
        epoch=best_epoch,
        metrics={'best_loss': best_loss, 'best_epoch': best_epoch, 'peak_vram_mb': peak_vram, 'tau_at_best': tau_best},
        config=best_latents,
    )
    save_checkpoint(
        model,
        final_ckpt_path,
        optimizer=optimizer,
        epoch=epochs,
        metrics={'final_loss': cur_meas_loss, 'best_loss': best_loss, 'peak_vram_mb': peak_vram},
        config={'u_inp': u_inp.detach().cpu(), 'v_inp': v_inp.detach().cpu(), 'w_inp': w_inp.detach().cpu()},
    )

    # 4. Evaluate Metrics
    # Protocol A: In-loop snapshot
    metrics_inloop = compute_metrics(clean_tensor, best_recon)

    # Protocol B: Clean-latent DIP inference (reload best checkpoint)
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

    # Protocol C: Unit temperature tau=1.0 diagnostic
    adaptive_block.set_temperature(1.0)
    with torch.no_grad():
        recon_tau1 = model(
            best_latents['u_inp'].to(device),
            best_latents['v_inp'].to(device),
            best_latents['w_inp'].to(device),
        ).cpu().numpy()
    metrics_tau1 = compute_metrics(clean_tensor, recon_tau1)
    adaptive_block.set_temperature(tau_best)

    # Protocol D: Eval mode running statistics
    model.eval()
    with torch.no_grad():
        recon_eval = model(
            best_latents['u_inp'].to(device),
            best_latents['v_inp'].to(device),
            best_latents['w_inp'].to(device),
        ).cpu().numpy()
    metrics_eval = compute_metrics(clean_tensor, recon_eval)

    log_print(f"--> RESULTS: PSNR (In-Loop)={metrics_inloop['psnr']:.4f} dB | SSIM={metrics_inloop['ssim']:.4f}")
    log_print(f"--> RESULTS: PSNR (Clean DIP)={metrics_dip['psnr']:.4f} dB | SSIM={metrics_dip['ssim']:.4f}")

    # Extract final router telemetry
    final_router_telem = extract_router_telemetry(adaptive_block)

    # Save metrics and telemetry
    metrics_data = {
        'cond_id': cond_id,
        'name': cond_cfg['name'],
        'phase': cond_cfg['phase'],
        'router_type': cond_cfg['router_type'],
        'gating_floor': cond_cfg['gating_floor'],
        'tau_schedule': cond_cfg['tau_schedule'],
        'entropy_reg': cond_cfg['entropy_reg'],
        'total_params': total_params,
        'block_params': block_params,
        'runtime_sec': total_train_time,
        'peak_vram_mb': peak_vram,
        'best_epoch': best_epoch,
        'best_loss': best_loss,
        'tau_at_best': tau_best,
        'metrics_inloop': metrics_inloop,
        'metrics_dip': metrics_dip,
        'metrics_tau1': metrics_tau1,
        'metrics_eval': metrics_eval,
        'final_router_telem': final_router_telem,
    }

    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_data, f, indent=2)

    with open(os.path.join(out_dir, 'telemetry.json'), 'w') as f:
        json.dump(telemetry_history, f, indent=2)

    with open(os.path.join(out_dir, 'loss_history.json'), 'w') as f:
        json.dump(loss_history, f)

    # Save representative middle slice visualization
    mid_idx = T // 2
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(clean_tensor[:, :, mid_idx], cmap='gray', vmin=0, vmax=1)
    axes[0].set_title("Ground Truth (Yale B)")
    axes[0].axis('off')

    axes[1].imshow(noisy_tensor[:, :, mid_idx], cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f"30% Salt & Pepper\nPSNR: 9.87 dB")
    axes[1].axis('off')

    axes[2].imshow(recon_dip[:, :, mid_idx], cmap='gray', vmin=0, vmax=1)
    axes[2].set_title(f"{cond_cfg['name']}\nPSNR: {metrics_dip['psnr']:.2f} dB, SSIM: {metrics_dip['ssim']:.4f}")
    axes[2].axis('off')

    diff = np.abs(clean_tensor[:, :, mid_idx] - recon_dip[:, :, mid_idx])
    im = axes[3].imshow(diff, cmap='magma', vmin=0, vmax=0.3)
    axes[3].set_title("Absolute Error Map")
    axes[3].axis('off')
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'reconstruction_slice.png'), dpi=150)
    plt.close()

    log_f.close()
    return metrics_data


def run_all_conditions(phases: List[int], target_conds: Optional[List[str]] = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_out_dir = os.path.join(project_root, 'results', 'extensions', 'ahd_net_v2_ablations', 'figure10_yale')
    os.makedirs(base_out_dir, exist_ok=True)

    print("=" * 80)
    print("AHD-NET V2 CONTROLLED RESEARCH ABLATION SUITE: FIGURE 10 (YALE FACE B)")
    print(f"Target Phases: {phases} | Device: {device}")
    print("=" * 80)

    # Load Yale dataset
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

    # Generate 30% Salt & Pepper noise under Seed 42
    set_seed(42)
    noisy_tensor = add_salt_and_pepper_noise(clean_tensor, 0.30)
    obs_m = compute_metrics(clean_tensor, noisy_tensor)
    print(f"Loaded Yale Face B {clean_tensor.shape}. Observation Seed 42: PSNR={obs_m['psnr']:.2f} dB, SSIM={obs_m['ssim']:.4f}")

    all_results = {}
    master_summary_path = os.path.join(base_out_dir, 'yale_ablation_master_summary.json')
    if os.path.exists(master_summary_path):
        with open(master_summary_path, 'r') as f:
            all_results = json.load(f).get('conditions', {})

    selected_conditions = {}
    for cid, ccfg in CONDITIONS.items():
        if target_conds is not None and cid not in target_conds:
            continue
        if ccfg['phase'] in phases:
            selected_conditions[cid] = ccfg

    print(f"\nConditions to run ({len(selected_conditions)}): {list(selected_conditions.keys())}\n")

    suite_start = time.time()
    for cid, ccfg in selected_conditions.items():
        cond_res = run_ablation_condition(
            cond_id=cid,
            cond_cfg=ccfg,
            clean_tensor=clean_tensor,
            noisy_tensor=noisy_tensor,
            base_out_dir=base_out_dir,
            device=device,
            epochs=1200,
            rank=2000,
            lr=1e-3,
        )
        all_results[cid] = cond_res

        # Save cumulative master summary
        with open(master_summary_path, 'w') as f:
            json.dump({
                'experiment': 'ahd_net_v2_yale_ablations',
                'dataset': 'Extended Yale Face B (192x168x160)',
                'rank': 2000,
                'epochs': 1200,
                'conditions': all_results,
            }, f, indent=2)

    total_suite_time = time.time() - suite_start
    print("\n" + "=" * 80)
    print(f"ALL SELECTED CONDITIONS COMPLETE! Total Suite Time: {total_suite_time / 60.0:.2f} min")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run AHD-Net v2 Yale Ablation Study")
    parser.add_argument('--phases', nargs='+', type=int, default=[1, 2], help="Phases to execute (1, 2, or both)")
    parser.add_argument('--conditions', nargs='+', type=str, default=None, help="Specific condition IDs to run")
    args = parser.parse_args()

    run_all_conditions(phases=args.phases, target_conds=args.conditions)

