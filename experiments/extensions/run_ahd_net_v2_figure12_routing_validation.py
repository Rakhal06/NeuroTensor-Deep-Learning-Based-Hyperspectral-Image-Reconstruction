"""
AHD-Net v2 Stage 3: Figure 12 PET/CT Radon Compressive Sensing Routing Validation
================================================================================
Application: Figure 12 — 3D PET/CT Radon Compressive Sensing (40 Angles)
Decomposition: 3-Way CP/PARAFAC (Rank R=1000, 128x128x56)
Physics Forward Operator: 40-Angle 2D Radon Transform (0° to 180°)
Noise: Mixed Poisson (tau=100.0) + Gaussian Readout (SNR=2.0), Observation Seed 42
Latent Initialization: Seed 200 (Positional Encodings)
Training Budget: 4,000 Epochs | Optimizer: Adam (lr = 1e-3)
Supervision: 100% Unsupervised Measurement L2 (Frobenius) Sinogram Loss

Conditions:
  Condition A (Static Equal): w = [1/3, 1/3, 1/3], no learned router.
  Condition B (Constant Tau=1.0): Learned CAGC router, fixed constant tau = 1.0 (no decay).
"""

import os
import sys
import copy
import time
import math
import json
import random
import argparse
import warnings
warnings.filterwarnings('ignore')
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
from deeptensor.physics import operators as lin_inverse
from deeptensor.models.factory import (
    build_deeptensor_cp_model,
    count_parameters,
    save_checkpoint,
)
from deeptensor.models.attention import AdaptiveTriBranchBlock
from deeptensor.utils.comparison import compute_metrics, to_2d_slice


CONDITIONS = {
    'cond_a_static_equal': {
        'name': 'Static Equal Fusion',
        'condition_id': 'Condition A',
        'router_type': 'static',
        'tau_schedule': 'constant',
        'tau': 1.0,
        'description': 'Fixed uniform ensemble (w = [1/3, 1/3, 1/3]) without learned router',
    },
    'cond_b_constant_tau1': {
        'name': 'Constant Temperature CAGC',
        'condition_id': 'Condition B',
        'router_type': 'cagc',
        'tau_schedule': 'constant',
        'tau': 1.0,
        'description': 'Learned CAGC router with fixed constant unit temperature (tau = 1.0, no decay)',
    },
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        g_norms[name] = float(torch.norm(torch.stack(norms)).item()) if norms else 0.0

    if adaptive_block.router is not None:
        r_norms = [torch.norm(p.grad) for p in adaptive_block.router.parameters() if p.grad is not None]
        g_norms['router'] = float(torch.norm(torch.stack(r_norms)).item()) if r_norms else 0.0
    else:
        g_norms['router'] = 0.0

    return g_norms


def run_condition(
    cond_id: str,
    cond_cfg: Dict[str, Any],
    cube: np.ndarray,
    meas_ten: torch.Tensor,
    angles: torch.Tensor,
    base_out_dir: str,
    device: torch.device,
    epochs: int = 4000,
    rank: int = 1000,
    lr: float = 1e-3,
    reg_noise_std: float = 1.0 / 30.0,
    nangles: int = 40,
) -> Dict[str, Any]:
    out_dir = os.path.join(base_out_dir, cond_id)
    os.makedirs(out_dir, exist_ok=True)
    log_file = os.path.join(out_dir, "run_log.txt")
    log_f = open(log_file, 'w', encoding='utf-8')

    def log_print(msg: str):
        print(msg)
        log_f.write(msg + '\n')
        log_f.flush()

    log_print("=" * 80)
    log_print(f"RUNNING FIGURE 12 PET/CT ROUTING VALIDATION: {cond_id} — {cond_cfg['name']}")
    log_print(f"Condition: {cond_cfg['condition_id']} | Description: {cond_cfg['description']}")
    log_print(f"Router Type: {cond_cfg['router_type']} | Fixed Tau: {cond_cfg['tau']}")
    log_print(f"Epochs: {epochs} | Rank: {rank} | Angles: {nangles} | Device: {device}")
    log_print("=" * 80)

    H, W, T = cube.shape
    criterion_l2 = losses.L2Norm()

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
        tau=cond_cfg['tau'],
    ).to(device)

    total_params = count_parameters(model)
    adaptive_block = model.w_net.adaptive_block
    block_params = count_parameters(adaptive_block)
    log_print(f"--> Trainable Parameters: Model={total_params:,} | AdaptiveBlock={block_params:,}")

    # 2. Latent Initialization under Seed 200
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

    best_loss = float('inf')
    best_epoch = 0
    best_recon = None
    best_model_state = None
    best_latents = None

    loss_history = []
    telemetry_history = []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    train_start_time = time.time()
    model.train()

    for epoch in range(1, epochs + 1):
        # Set temperature
        adaptive_block.set_temperature(cond_cfg['tau'])

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

        # Telemetry capture
        if epoch % 100 == 0 or epoch == 1 or epoch == epochs:
            g_norms = compute_branch_gradient_norms(adaptive_block)
            router_telem = extract_router_telemetry(adaptive_block)
            telem_entry = {
                'epoch': epoch,
                'meas_loss': cur_loss,
                'tau': cond_cfg['tau'],
                'router_entropy': router_telem['router_entropy'],
                'channel_entropy': router_telem['channel_entropy'],
                'mean_weights': router_telem['mean_weights'],
                'dominant_counts': router_telem['dominant_counts'],
                'gradient_norms': g_norms,
            }
            telemetry_history.append(telem_entry)

            if epoch % 500 == 0 or epoch == 1 or epoch == epochs:
                mw = router_telem['mean_weights']
                dc = router_telem['dominant_counts']
                log_print(
                    f"  [Epoch {epoch:4d}/{epochs}] Loss: {cur_loss:.5e} | Best Loss: {best_loss:.5e} (Ep {best_epoch}) | "
                    f"H: {router_telem['channel_entropy']:.4f} | "
                    f"Gates: CNN={mw['cnn']:.3f}, PW={mw['pointwise']:.3f}, Attn={mw['attention']:.3f} | "
                    f"Dominance: CNN={dc['cnn']}, PW={dc['pointwise']}, Attn={dc['attention']}"
                )

        optimizer.step()

        # Strictly unsupervised checkpoint selection: lowest measurement L2 sinogram loss
        if cur_loss < best_loss:
            best_loss = cur_loss
            best_epoch = epoch
            best_recon = tensor_estim.detach().cpu().numpy()
            best_model_state = copy.deepcopy(model.state_dict())
            best_latents = {
                'u_inp': u_inp.detach().cpu(),
                'v_inp': v_inp.detach().cpu(),
                'w_inp': w_inp.detach().cpu(),
            }

    total_train_time = time.time() - train_start_time
    peak_vram = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if torch.cuda.is_available() else 0.0

    log_print(f"--> Optimization Complete: Runtime={total_train_time:.1f}s | Best Meas Loss={best_loss:.6e} at Ep {best_epoch}")

    # 3. Save Checkpoints
    best_ckpt_path = os.path.join(out_dir, "best_checkpoint.pth")
    final_ckpt_path = os.path.join(out_dir, "final_checkpoint.pth")

    save_checkpoint(
        model,
        best_ckpt_path,
        optimizer=optimizer,
        epoch=best_epoch,
        metrics={'best_loss': best_loss, 'best_epoch': best_epoch, 'peak_vram_mb': peak_vram, 'tau_at_best': cond_cfg['tau']},
        config=best_latents,
    )
    save_checkpoint(
        model,
        final_ckpt_path,
        optimizer=optimizer,
        epoch=epochs,
        metrics={'final_loss': cur_loss, 'best_loss': best_loss, 'peak_vram_mb': peak_vram},
        config={'u_inp': u_inp.detach().cpu(), 'v_inp': v_inp.detach().cpu(), 'w_inp': w_inp.detach().cpu()},
    )

    # 4. Evaluate Metrics (Clean DIP vs In-Loop)
    # Protocol A: In-loop snapshot at best epoch
    metrics_inloop = compute_metrics(cube, best_recon)

    # Protocol B: Clean-latent DIP inference (without random perturbation noise)
    model.load_state_dict(best_model_state)
    adaptive_block.set_temperature(cond_cfg['tau'])
    model.train()
    with torch.no_grad():
        recon_dip = model(
            best_latents['u_inp'].to(device),
            best_latents['v_inp'].to(device),
            best_latents['w_inp'].to(device),
        ).cpu().numpy()
    metrics_dip = compute_metrics(cube, recon_dip)

    log_print(f"--> RESULTS: PSNR (In-Loop)={metrics_inloop['psnr']:.4f} dB | SSIM={metrics_inloop['ssim']:.4f}")
    log_print(f"--> RESULTS: PSNR (Clean DIP)={metrics_dip['psnr']:.4f} dB | SSIM={metrics_dip['ssim']:.4f}")

    final_router_telem = extract_router_telemetry(adaptive_block)

    metrics_data = {
        'cond_id': cond_id,
        'condition_id': cond_cfg['condition_id'],
        'name': cond_cfg['name'],
        'router_type': cond_cfg['router_type'],
        'tau': cond_cfg['tau'],
        'total_params': total_params,
        'block_params': block_params,
        'runtime_sec': total_train_time,
        'peak_vram_mb': peak_vram,
        'best_epoch': best_epoch,
        'best_loss': best_loss,
        'metrics_inloop': metrics_inloop,
        'metrics_dip': metrics_dip,
        'final_router_telem': final_router_telem,
    }

    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_data, f, indent=2)

    with open(os.path.join(out_dir, 'telemetry.json'), 'w') as f:
        json.dump(telemetry_history, f, indent=2)

    with open(os.path.join(out_dir, 'loss_history.json'), 'w') as f:
        json.dump(loss_history, f)

    # Representative slice visualization (transaxial slices 14, 28, 42)
    sample_slices = [14, 28, 42]
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for col_idx, s_idx in enumerate(sample_slices):
        axes[0, col_idx].imshow(cube[:, :, s_idx], cmap='hot', vmin=0, vmax=1)
        axes[0, col_idx].set_title(f"GT Slice {s_idx}")
        axes[0, col_idx].axis('off')

        axes[1, col_idx].imshow(recon_dip[:, :, s_idx], cmap='hot', vmin=0, vmax=1)
        axes[1, col_idx].set_title(f"{cond_cfg['name']}\nPSNR: {metrics_dip['psnr']:.2f} dB")
        axes[1, col_idx].axis('off')

        err = np.abs(cube[:, :, s_idx] - recon_dip[:, :, s_idx])
        im = axes[2, col_idx].imshow(err, cmap='magma', vmin=0, vmax=0.15)
        axes[2, col_idx].set_title("Absolute Error")
        axes[2, col_idx].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'reconstruction_slice_strip.png'), dpi=150)
    plt.close()

    log_f.close()
    return metrics_data


def run_all(target_cond: Optional[str] = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_out_dir = os.path.join(project_root, 'results', 'extensions', 'ahd_net_v2_ablations', 'figure12_pet_ct')
    os.makedirs(base_out_dir, exist_ok=True)

    print("=" * 80)
    print("AHD-NET V2 STAGE 3: FIGURE 12 PET/CT ROUTING VALIDATION")
    print(f"Device: {device} | Output: {base_out_dir}")
    print("=" * 80)

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
    nangles = 40

    # 2. Physics Forward Model: 40-Angle 2D Radon Projection + Detector Noise
    set_seed(42)
    angles = torch.tensor(np.linspace(0, 180, nangles).astype(np.float32), device=device)
    cube_ten = torch.tensor(cube, dtype=torch.float32, device=device).permute(2, 0, 1)[None, ...]  # (1, 56, 128, 128)
    clean_measurements = lin_inverse.radon(cube_ten, angles).detach().cpu().numpy()  # (40, 56, 128)
    noisy_measurements = utils.measure(clean_measurements, 2.0, 100.0)
    meas_ten = torch.tensor(noisy_measurements, dtype=torch.float32, device=device)

    print(f"Loaded PET/CT ({H}, {W}, {T}). Observation Seed 42: Sinogram {meas_ten.shape} across {nangles} angles")

    selected_conditions = CONDITIONS
    if target_cond is not None and target_cond in CONDITIONS:
        selected_conditions = {target_cond: CONDITIONS[target_cond]}

    all_results = {}
    master_summary_path = os.path.join(base_out_dir, 'figure12_routing_validation_summary.json')
    if os.path.exists(master_summary_path):
        with open(master_summary_path, 'r') as f:
            all_results = json.load(f).get('conditions', {})

    suite_start = time.time()
    for cid, ccfg in selected_conditions.items():
        cond_res = run_condition(
            cond_id=cid,
            cond_cfg=ccfg,
            cube=cube,
            meas_ten=meas_ten,
            angles=angles,
            base_out_dir=base_out_dir,
            device=device,
            epochs=4000,
            rank=1000,
            lr=1e-3,
            reg_noise_std=1.0 / 30.0,
            nangles=nangles,
        )
        all_results[cid] = cond_res

        with open(master_summary_path, 'w') as f:
            json.dump({
                'experiment': 'ahd_net_v2_figure12_routing_validation',
                'dataset': 'PET/CT (128x128x56)',
                'rank': 1000,
                'epochs': 4000,
                'conditions': all_results,
            }, f, indent=2)

    total_time = time.time() - suite_start
    print("\n" + "=" * 80)
    print(f"FIGURE 12 ROUTING VALIDATION COMPLETE! Total Runtime: {total_time / 60.0:.2f} min")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run AHD-Net v2 Figure 12 Routing Validation")
    parser.add_argument('--condition', type=str, default=None, help="Run specific condition")
    args = parser.parse_args()

    run_all(target_cond=args.condition)

