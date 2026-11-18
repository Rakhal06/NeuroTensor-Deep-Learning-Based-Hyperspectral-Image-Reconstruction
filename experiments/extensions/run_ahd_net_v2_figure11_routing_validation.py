"""
AHD-Net v2 Stage 2: Figure 11 Video Compressive Sensing Routing Validation
==========================================================================
Application: Figure 11 — Video Compressive Sensing (CACTI)
Dataset: Cat Video (data/cat_video.mat), shape: (128, 128, 64)
Forward Model: 8-Frame Temporal Multiplexing with Random Binary Coded Aperture
Decomposition Rank: R = 32
Noise: Poisson tau = 100, Gaussian SNR = 2 (Observation Seed 42)
Latent Initialization: Seed 200 (Clean Noise Latents, reg_noise_std = 1/30)
Training Budget: 3,000 Epochs | Optimizer: Adam (lr = 1e-3)
Supervision: 100% Unsupervised Measurement L2 Loss

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
from typing import Dict, Any, List, Optional

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


CONDITIONS = {
    'cond_a_static_equal': {
        'name': 'Static Equal Ensemble',
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
    cube_noisy: np.ndarray,
    measurements_ten: torch.Tensor,
    masks_ten: torch.Tensor,
    base_out_dir: str,
    device: torch.device,
    epochs: int = 3000,
    rank: int = 32,
    lr: float = 1e-3,
    reg_noise_std: float = 1.0 / 30.0,
    nframes: int = 8,
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
    log_print(f"RUNNING FIGURE 11 ROUTING VALIDATION: {cond_id} — {cond_cfg['name']}")
    log_print(f"Condition: {cond_cfg['condition_id']} | Description: {cond_cfg['description']}")
    log_print(f"Router Type: {cond_cfg['router_type']} | Fixed Tau: {cond_cfg['tau']}")
    log_print(f"Epochs: {epochs} | Rank: {rank} | Device: {device}")
    log_print("=" * 80)

    H, W, totalframes = cube.shape
    criterion_l2 = losses.L2Norm()

    # 1. Build Model under Seed 100
    set_seed(100)
    model = build_deeptensor_model(
        rank=rank,
        seq_len=totalframes,
        spatial_type='skip',
        spectral_type='adaptive_hybrid',
        n_inputs=rank,
        skip_n33d=64,
        skip_n33u=64,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ffn=512,
        router_type=cond_cfg['router_type'],
        pe_type='sin_pe',
        tau=cond_cfg['tau'],
    ).to(device)

    total_params = count_parameters(model)
    adaptive_block = model.spectral_net.adaptive_block
    block_params = count_parameters(adaptive_block)
    log_print(f"--> Trainable Parameters: Model={total_params:,} | AdaptiveBlock={block_params:,}")

    # 2. Latent Initialization under Seed 200
    set_seed(200)
    im_inp = utils.get_inp([1, rank, H, W], device=device)
    spec_inp = utils.get_inp([1, rank, totalframes], device=device)
    im_inp_per = im_inp.detach().clone()
    spec_inp_per = spec_inp.detach().clone()

    optimizer = torch.optim.Adam(
        list(model.parameters()) + [im_inp, spec_inp],
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
        # Set temperature
        adaptive_block.set_temperature(cond_cfg['tau'])

        # In-loop latent perturbation
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

        # Telemetry capture
        if idx == 0 or (idx + 1) % 100 == 0 or idx == epochs - 1:
            grad_norms = compute_branch_gradient_norms(adaptive_block)
            router_telem = extract_router_telemetry(adaptive_block)
            telem_entry = {
                'epoch': idx + 1,
                'meas_loss': cur_loss,
                'tau': cond_cfg['tau'],
                'router_entropy': router_telem['router_entropy'],
                'channel_entropy': router_telem['channel_entropy'],
                'mean_weights': router_telem['mean_weights'],
                'dominant_counts': router_telem['dominant_counts'],
                'gradient_norms': grad_norms,
            }
            telemetry_history.append(telem_entry)

            if (idx + 1) % 500 == 0 or idx == 0 or idx == epochs - 1:
                mw = router_telem['mean_weights']
                dc = router_telem['dominant_counts']
                log_print(
                    f"  [Epoch {idx + 1:4d}/{epochs}] Loss: {cur_loss:.5e} | Best Loss: {best_loss:.5e} (Ep {best_epoch}) | "
                    f"H: {router_telem['channel_entropy']:.4f} | "
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
        config={'im_inp': im_inp.detach().cpu(), 'spec_inp': spec_inp.detach().cpu()},
    )

    # 4. Evaluate Metrics (Clean DIP vs In-Loop)
    # Protocol A: In-loop snapshot at best epoch
    metrics_inloop = compute_metrics(cube, best_recon)

    # Protocol B: Clean-latent DIP inference (without random perturbation noise)
    model.load_state_dict(best_model_state)
    adaptive_block.set_temperature(cond_cfg['tau'])
    model.train()
    with torch.no_grad():
        out_mat = model(best_latents['im_inp'].to(device), best_latents['spec_inp'].to(device))
        recon_dip = out_mat[0].cpu().numpy().reshape(H, W, totalframes)
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

    # Temporal frame strip visualization (frames 0, 16, 32, 48)
    sample_frames = [0, 16, 32, 48]
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    for col_idx, f_idx in enumerate(sample_frames):
        axes[0, col_idx].imshow(cube[:, :, f_idx], cmap='gray', vmin=0, vmax=1)
        axes[0, col_idx].set_title(f"GT Frame {f_idx}")
        axes[0, col_idx].axis('off')

        axes[1, col_idx].imshow(recon_dip[:, :, f_idx], cmap='gray', vmin=0, vmax=1)
        axes[1, col_idx].set_title(f"{cond_cfg['name']}\nPSNR: {metrics_dip['psnr']:.2f} dB")
        axes[1, col_idx].axis('off')

        err = np.abs(cube[:, :, f_idx] - recon_dip[:, :, f_idx])
        im = axes[2, col_idx].imshow(err, cmap='magma', vmin=0, vmax=0.25)
        axes[2, col_idx].set_title("Absolute Error")
        axes[2, col_idx].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'temporal_frame_strip.png'), dpi=150)
    plt.close()

    log_f.close()
    return metrics_data


def run_all(target_cond: Optional[str] = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_out_dir = os.path.join(project_root, 'results', 'extensions', 'ahd_net_v2_ablations', 'figure11_video_cs')
    os.makedirs(base_out_dir, exist_ok=True)

    print("=" * 80)
    print("AHD-NET V2 STAGE 2: FIGURE 11 VIDEO COMPRESSIVE SENSING ROUTING VALIDATION")
    print(f"Device: {device} | Output: {base_out_dir}")
    print("=" * 80)

    # 1. Load Cat Video
    data_path = os.path.join(project_root, 'data', 'cat_video.mat')
    data = io.loadmat(data_path)
    cube = data['hypercube'].astype(np.float32)
    cube = utils.resize(cube / cube.max(), 1)
    H, W, totalframes = cube.shape
    nframes = 8

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

    obs_m = compute_metrics(cube, cube_noisy)
    print(f"Loaded Cat Video ({H}, {W}, {totalframes}). Observation Seed 42: PSNR={obs_m['psnr']:.2f} dB, SSIM={obs_m['ssim']:.4f}")

    selected_conditions = CONDITIONS
    if target_cond is not None and target_cond in CONDITIONS:
        selected_conditions = {target_cond: CONDITIONS[target_cond]}

    all_results = {}
    master_summary_path = os.path.join(base_out_dir, 'figure11_routing_validation_summary.json')
    if os.path.exists(master_summary_path):
        with open(master_summary_path, 'r') as f:
            all_results = json.load(f).get('conditions', {})

    suite_start = time.time()
    for cid, ccfg in selected_conditions.items():
        cond_res = run_condition(
            cond_id=cid,
            cond_cfg=ccfg,
            cube=cube,
            cube_noisy=cube_noisy,
            measurements_ten=measurements_ten,
            masks_ten=masks_ten,
            base_out_dir=base_out_dir,
            device=device,
            epochs=3000,
            rank=32,
            lr=1e-3,
            reg_noise_std=1.0 / 30.0,
            nframes=8,
        )
        all_results[cid] = cond_res

        with open(master_summary_path, 'w') as f:
            json.dump({
                'experiment': 'ahd_net_v2_figure11_routing_validation',
                'dataset': 'Cat Video (128x128x64)',
                'rank': 32,
                'epochs': 3000,
                'conditions': all_results,
            }, f, indent=2)

    total_time = time.time() - suite_start
    print("\n" + "=" * 80)
    print(f"FIGURE 11 ROUTING VALIDATION COMPLETE! Total Runtime: {total_time / 60.0:.2f} min")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run AHD-Net v2 Figure 11 Routing Validation")
    parser.add_argument('--condition', type=str, default=None, help="Run specific condition")
    args = parser.parse_args()

    run_all(target_cond=args.condition)
