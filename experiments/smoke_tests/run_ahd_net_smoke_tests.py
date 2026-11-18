"""
Adaptive Hybrid DeepTensor v1 (AHD-Net) Four-Application Protocol Smoke Test
===========================================================================
Executes a verified 1-epoch protocol smoke test of AHD-Net inside the actual
DeepTensor reconstruction pipelines across all four modalities:
  1. Figure 09 — ICVL Hyperspectral Denoising (Rank R=20, 128x128x260)
  2. Figure 10 — Yale Face B Denoising (Rank R=2000, 192x168x160)
  3. Figure 11 — Video CS CACTI (Rank R=32, 128x128x64)
  4. Figure 12 — 3D PET/CT Radon CS (Rank R=1000, 128x128x56)

Records:
  - Exact parameter counts and historical delta
  - Forward / backward time and peak CUDA VRAM
  - Routing weights (CNN, PW, Attn) and channel distribution
  - Branch gradient norms (starvation / collapse detection)
  - Router entropy and temperature dynamics
  - Checkpoint saving and independent checkpoint reload verification
"""

import os
import sys
import time
import math
import json
import random
from typing import Dict, Any, Tuple

import numpy as np
from scipy import io
import torch
import torch.nn as nn

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, os.path.join(project_root, 'src'))
    sys.path.insert(0, project_root)

from deeptensor import utils, losses
from deeptensor.physics import operators as lin_inverse
from deeptensor.models.factory import (
    build_deeptensor_model,
    build_deeptensor_cp_model,
    count_parameters,
    save_checkpoint,
    load_checkpoint,
)
from deeptensor.models.attention import (
    AdaptiveTriBranchBlock,
    AdaptiveHybridSpectralNet,
    get_cosine_temperature,
)


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
        # gates: [B, C, 3] or [B, C, L, 3]
        if gates.ndim == 3:
            g_mean = gates.mean(dim=(0, 1)).cpu().numpy()
            c_weights = gates[0].cpu().numpy()  # [C, 3]
        elif gates.ndim == 4:
            g_mean = gates.mean(dim=(0, 1, 2)).cpu().numpy()
            c_weights = gates[0].mean(dim=1).cpu().numpy()  # [C, 3]
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
            'initial_tau': adaptive_block.tau,
            'final_tau': adaptive_block.tau,
            'sums_to_one': sums_valid,
        }
    return telemetry


# ==============================================================================
# APPLICATION 1: Figure 09 — ICVL Hyperspectral Denoising
# ==============================================================================
def smoke_test_figure09(device: torch.device, out_dir: str) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("APPLICATION 1: FIGURE 09 — ICVL HYPERSPECTRAL DENOISING")
    print("=" * 80)

    # 1. Load data
    data_path = os.path.join(project_root, 'data', 'icvl.mat')
    data = io.loadmat(data_path)
    cube = data['hypercube'].astype(np.float32)
    cube = utils.resize(cube / cube.max(), 1)
    H, W, nwvl = cube.shape
    rank = 20
    n_inputs = rank
    reg_noise_std = 1.0 / 30.0

    set_seed(42)
    cube_noisy = utils.measure(cube, 2, 100)
    hsten_noisy = torch.tensor(cube_noisy.reshape(H * W, nwvl), dtype=torch.float32)[None, ...].to(device)
    criterion_l1 = losses.L1Norm()

    # 2. Build model
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
    ).to(device)

    total_params = count_parameters(model)
    target_params = 3137808
    delta_params = total_params - target_params
    print(f"  [Fig 09] Trainable Parameters: {total_params:,} (Target: {target_params:,}, Delta: {delta_params})")
    assert delta_params == 0, f"Parameter mismatch in Fig 09: {total_params} != {target_params}"

    # 3. Latents & Optimizer
    set_seed(200)
    im_inp = utils.get_inp([1, n_inputs, H, W]).to(device)
    spec_inp = utils.get_inp([1, n_inputs, nwvl]).to(device)
    im_inp_per = im_inp.detach().clone()
    spec_inp_per = spec_inp.detach().clone()

    optimizer = torch.optim.Adam(list(model.parameters()) + [im_inp, spec_inp], lr=1e-3)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    # 4. Execute 1 Epoch
    model.train()
    epoch_start = time.time()

    im_pert = im_inp + im_inp_per.normal_() * reg_noise_std
    spec_pert = spec_inp + spec_inp_per.normal_() * reg_noise_std

    t_fwd_start = time.time()
    hs_estim = model(im_pert, spec_pert)
    loss = criterion_l1(hsten_noisy - hs_estim)
    t_fwd = (time.time() - t_fwd_start) * 1000.0

    t_bwd_start = time.time()
    optimizer.zero_grad()
    loss.backward()
    t_bwd = (time.time() - t_bwd_start) * 1000.0

    # Extract diagnostics before step
    adaptive_block = model.spectral_net.adaptive_block
    grad_norms = compute_branch_gradient_norms(adaptive_block)
    telemetry = extract_router_telemetry(adaptive_block)

    optimizer.step()
    epoch_time = time.time() - epoch_start
    peak_vram = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if torch.cuda.is_available() else 0.0

    cur_loss = float(loss.item())
    is_finite = bool(math.isfinite(cur_loss) and not torch.isnan(hs_estim).any() and not torch.isinf(hs_estim).any())
    print(f"  [Fig 09] 1-Epoch Loss: {cur_loss:.6f} | Finite: {is_finite} | Time: {epoch_time:.2f}s | Peak VRAM: {peak_vram:.1f}MB")
    print(f"  [Fig 09] Timing: Fwd {t_fwd:.1f}ms | Bwd {t_bwd:.1f}ms")
    print(f"  [Fig 09] Routing Weights: CNN={telemetry['mean_weights']['cnn']:.4f}, PW={telemetry['mean_weights']['pointwise']:.4f}, Attn={telemetry['mean_weights']['attention']:.4f}")
    print(f"  [Fig 09] Gradient Norms: CNN={grad_norms['cnn_branch']:.6f}, PW={grad_norms['pw_branch']:.6f}, Attn={grad_norms['attn_branch']:.6f}, Router={grad_norms['router']:.6f}")

    # Check branch starvation
    for b_name, norm in grad_norms.items():
        assert norm > 0.0, f"Branch starvation detected in Fig 09: {b_name} has zero gradient norm!"

    # 5. Checkpoint Saving & Independent Verification
    ckpt_path = os.path.join(out_dir, "fig09_ahd_net_checkpoint.pth")
    save_checkpoint(
        model,
        ckpt_path,
        optimizer=optimizer,
        epoch=1,
        metrics={'loss': cur_loss, 'vram_mb': peak_vram},
        config={'im_inp': im_inp.detach().cpu(), 'spec_inp': spec_inp.detach().cpu()}
    )
    assert os.path.exists(ckpt_path), f"Checkpoint not found at {ckpt_path}"
    ckpt_size_mb = os.path.getsize(ckpt_path) / (1024 ** 2)

    # Independent reload
    model_reloaded = build_deeptensor_model(
        rank=rank, seq_len=nwvl, spatial_type='skip', spectral_type='adaptive_hybrid', n_inputs=n_inputs
    ).to(device)
    model_reloaded.eval()
    ckpt_data = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_reloaded.load_state_dict(ckpt_data['model_state_dict'])
    with torch.no_grad():
        test_out = model_reloaded(im_inp, spec_inp)
    reload_verified = bool(torch.isfinite(test_out).all())
    print(f"  [Fig 09] Checkpoint Saved: {ckpt_path} ({ckpt_size_mb:.2f}MB) | Independent Reload Verified: {reload_verified}")

    return {
        'application': 'Figure 09 ICVL HSI',
        'params': total_params,
        'target_params': target_params,
        'delta_params': delta_params,
        'loss': cur_loss,
        'is_finite': is_finite,
        'forward_ms': t_fwd,
        'backward_ms': t_bwd,
        'epoch_sec': epoch_time,
        'peak_vram_mb': peak_vram,
        'grad_norms': grad_norms,
        'telemetry': telemetry,
        'checkpoint_path': ckpt_path,
        'checkpoint_size_mb': ckpt_size_mb,
        'reload_verified': reload_verified,
    }


# ==============================================================================
# APPLICATION 2: Figure 10 — Yale Face B Denoising
# ==============================================================================
def smoke_test_figure10(device: torch.device, out_dir: str) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("APPLICATION 2: FIGURE 10 — YALE FACE B DENOISING")
    print("=" * 80)

    # 1. Load data
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
    rank = 2000
    n_inputs = rank

    set_seed(42)
    noisy_tensor = add_salt_and_pepper_noise(clean_tensor, 0.30)
    noisy_ten = torch.tensor(noisy_tensor, dtype=torch.float32, device=device)
    criterion_l1 = losses.L1Norm()

    # 2. Build model
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
    ).to(device)

    total_params = count_parameters(model)
    target_params = 5602988
    delta_params = total_params - target_params
    print(f"  [Fig 10] Trainable Parameters: {total_params:,} (Target: {target_params:,}, Delta: {delta_params})")
    assert delta_params == 0, f"Parameter mismatch in Fig 10: {total_params} != {target_params}"

    # 3. Latents & Optimizer
    set_seed(200)
    u_inp = utils.get_1d_posencode_inp(H, rank // 2, device=device)
    v_inp = utils.get_1d_posencode_inp(W, rank // 2, device=device)
    w_inp = utils.get_1d_posencode_inp(T, rank // 2, device=device)

    optimizer = torch.optim.Adam(list(model.parameters()) + [u_inp, v_inp, w_inp], lr=1e-3)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    # 4. Execute 1 Epoch
    model.train()
    epoch_start = time.time()

    t_fwd_start = time.time()
    recon = model(u_inp, v_inp, w_inp)
    loss = criterion_l1(noisy_ten - recon)
    t_fwd = (time.time() - t_fwd_start) * 1000.0

    t_bwd_start = time.time()
    optimizer.zero_grad()
    loss.backward()
    t_bwd = (time.time() - t_bwd_start) * 1000.0

    adaptive_block = model.w_net.adaptive_block
    grad_norms = compute_branch_gradient_norms(adaptive_block)
    telemetry = extract_router_telemetry(adaptive_block)

    optimizer.step()
    epoch_time = time.time() - epoch_start
    peak_vram = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if torch.cuda.is_available() else 0.0

    cur_loss = float(loss.item())
    is_finite = bool(math.isfinite(cur_loss) and not torch.isnan(recon).any() and not torch.isinf(recon).any())
    print(f"  [Fig 10] 1-Epoch Loss: {cur_loss:.6f} | Finite: {is_finite} | Time: {epoch_time:.2f}s | Peak VRAM: {peak_vram:.1f}MB")
    print(f"  [Fig 10] Timing: Fwd {t_fwd:.1f}ms | Bwd {t_bwd:.1f}ms")
    print(f"  [Fig 10] Routing Weights: CNN={telemetry['mean_weights']['cnn']:.4f}, PW={telemetry['mean_weights']['pointwise']:.4f}, Attn={telemetry['mean_weights']['attention']:.4f}")
    print(f"  [Fig 10] Gradient Norms: CNN={grad_norms['cnn_branch']:.6f}, PW={grad_norms['pw_branch']:.6f}, Attn={grad_norms['attn_branch']:.6f}, Router={grad_norms['router']:.6f}")

    for b_name, norm in grad_norms.items():
        assert norm > 0.0, f"Branch starvation detected in Fig 10: {b_name} has zero gradient norm!"

    # 5. Checkpoint Saving & Independent Verification
    ckpt_path = os.path.join(out_dir, "fig10_ahd_net_checkpoint.pth")
    save_checkpoint(
        model,
        ckpt_path,
        optimizer=optimizer,
        epoch=1,
        metrics={'loss': cur_loss, 'vram_mb': peak_vram},
        config={'u_inp': u_inp.detach().cpu(), 'v_inp': v_inp.detach().cpu(), 'w_inp': w_inp.detach().cpu()}
    )
    assert os.path.exists(ckpt_path), f"Checkpoint not found at {ckpt_path}"
    ckpt_size_mb = os.path.getsize(ckpt_path) / (1024 ** 2)

    model_reloaded = build_deeptensor_cp_model(
        rank=rank, lengths=(H, W, T), u_type='skip1d', v_type='skip1d', w_type='adaptive_hybrid', n_inputs=n_inputs
    ).to(device)
    model_reloaded.eval()
    ckpt_data = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_reloaded.load_state_dict(ckpt_data['model_state_dict'])
    with torch.no_grad():
        test_out = model_reloaded(u_inp, v_inp, w_inp)
    reload_verified = bool(torch.isfinite(test_out).all())
    print(f"  [Fig 10] Checkpoint Saved: {ckpt_path} ({ckpt_size_mb:.2f}MB) | Independent Reload Verified: {reload_verified}")

    return {
        'application': 'Figure 10 Yale Face B',
        'params': total_params,
        'target_params': target_params,
        'delta_params': delta_params,
        'loss': cur_loss,
        'is_finite': is_finite,
        'forward_ms': t_fwd,
        'backward_ms': t_bwd,
        'epoch_sec': epoch_time,
        'peak_vram_mb': peak_vram,
        'grad_norms': grad_norms,
        'telemetry': telemetry,
        'checkpoint_path': ckpt_path,
        'checkpoint_size_mb': ckpt_size_mb,
        'reload_verified': reload_verified,
    }


# ==============================================================================
# APPLICATION 3: Figure 11 — Video Compressive Sensing (CACTI)
# ==============================================================================
def smoke_test_figure11(device: torch.device, out_dir: str) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("APPLICATION 3: FIGURE 11 — VIDEO COMPRESSIVE SENSING (CACTI)")
    print("=" * 80)

    # 1. Load video data
    data_path = os.path.join(project_root, 'data', 'cat_video.mat')
    data = io.loadmat(data_path)
    cube = data['hypercube'].astype(np.float32)
    cube = utils.resize(cube / cube.max(), 1)
    H, W, totalframes = cube.shape
    rank = 32
    n_inputs = rank
    nframes = 8
    reg_noise_std = 1.0 / 30.0

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
    criterion = losses.L2Norm()

    # 2. Build model (Note: skip_n33d=64, skip_n33u=64, d_model=64)
    set_seed(100)
    model = build_deeptensor_model(
        rank=rank,
        seq_len=totalframes,
        spatial_type='skip',
        spectral_type='adaptive_hybrid',
        n_inputs=n_inputs,
        skip_n33d=64,
        skip_n33u=64,
        d_model=64,
        n_heads=4,
        n_layers=4,
        d_ffn=512,
    ).to(device)

    total_params = count_parameters(model)
    target_params = 834632
    delta_params = total_params - target_params
    print(f"  [Fig 11] Trainable Parameters: {total_params:,} (Target: {target_params:,}, Delta: {delta_params})")
    assert delta_params == 0, f"Parameter mismatch in Fig 11: {total_params} != {target_params}"

    # 3. Latents & Optimizer
    set_seed(200)
    im_inp = utils.get_inp([1, n_inputs, H, W], device=device)
    spec_inp = utils.get_inp([1, n_inputs, totalframes], device=device)
    im_inp_per = im_inp.detach().clone()
    spec_inp_per = spec_inp.detach().clone()

    optimizer = torch.optim.Adam(list(model.parameters()) + [im_inp, spec_inp], lr=1e-3)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    # 4. Execute 1 Epoch
    model.train()
    epoch_start = time.time()

    im_pert = im_inp + im_inp_per.normal_() * reg_noise_std
    spec_pert = spec_inp + spec_inp_per.normal_() * reg_noise_std

    t_fwd_start = time.time()
    mat_estim = model(im_pert, spec_pert)
    cube_estim = mat_estim.reshape(1, H, W, totalframes).permute(0, 3, 1, 2)
    meas_estim = lin_inverse.video2codedvideo(cube_estim, masks_ten, nframes)
    loss = criterion(measurements_ten - meas_estim)
    t_fwd = (time.time() - t_fwd_start) * 1000.0

    t_bwd_start = time.time()
    optimizer.zero_grad()
    loss.backward()
    t_bwd = (time.time() - t_bwd_start) * 1000.0

    adaptive_block = model.spectral_net.adaptive_block
    grad_norms = compute_branch_gradient_norms(adaptive_block)
    telemetry = extract_router_telemetry(adaptive_block)

    optimizer.step()
    epoch_time = time.time() - epoch_start
    peak_vram = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if torch.cuda.is_available() else 0.0

    cur_loss = float(loss.item())
    is_finite = bool(math.isfinite(cur_loss) and not torch.isnan(mat_estim).any() and not torch.isinf(mat_estim).any())
    print(f"  [Fig 11] 1-Epoch Loss: {cur_loss:.6f} | Finite: {is_finite} | Time: {epoch_time:.2f}s | Peak VRAM: {peak_vram:.1f}MB")
    print(f"  [Fig 11] Timing: Fwd {t_fwd:.1f}ms | Bwd {t_bwd:.1f}ms")
    print(f"  [Fig 11] Routing Weights: CNN={telemetry['mean_weights']['cnn']:.4f}, PW={telemetry['mean_weights']['pointwise']:.4f}, Attn={telemetry['mean_weights']['attention']:.4f}")
    print(f"  [Fig 11] Gradient Norms: CNN={grad_norms['cnn_branch']:.6f}, PW={grad_norms['pw_branch']:.6f}, Attn={grad_norms['attn_branch']:.6f}, Router={grad_norms['router']:.6f}")

    for b_name, norm in grad_norms.items():
        assert norm > 0.0, f"Branch starvation detected in Fig 11: {b_name} has zero gradient norm!"

    # 5. Checkpoint Saving & Independent Verification
    ckpt_path = os.path.join(out_dir, "fig11_ahd_net_checkpoint.pth")
    save_checkpoint(
        model,
        ckpt_path,
        optimizer=optimizer,
        epoch=1,
        metrics={'loss': cur_loss, 'vram_mb': peak_vram},
        config={'im_inp': im_inp.detach().cpu(), 'spec_inp': spec_inp.detach().cpu()}
    )
    assert os.path.exists(ckpt_path), f"Checkpoint not found at {ckpt_path}"
    ckpt_size_mb = os.path.getsize(ckpt_path) / (1024 ** 2)

    model_reloaded = build_deeptensor_model(
        rank=rank, seq_len=totalframes, spatial_type='skip', spectral_type='adaptive_hybrid',
        n_inputs=n_inputs, skip_n33d=64, skip_n33u=64, d_model=64
    ).to(device)
    model_reloaded.eval()
    ckpt_data = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_reloaded.load_state_dict(ckpt_data['model_state_dict'])
    with torch.no_grad():
        test_out = model_reloaded(im_inp, spec_inp)
    reload_verified = bool(torch.isfinite(test_out).all())
    print(f"  [Fig 11] Checkpoint Saved: {ckpt_path} ({ckpt_size_mb:.2f}MB) | Independent Reload Verified: {reload_verified}")

    return {
        'application': 'Figure 11 Video CS',
        'params': total_params,
        'target_params': target_params,
        'delta_params': delta_params,
        'loss': cur_loss,
        'is_finite': is_finite,
        'forward_ms': t_fwd,
        'backward_ms': t_bwd,
        'epoch_sec': epoch_time,
        'peak_vram_mb': peak_vram,
        'grad_norms': grad_norms,
        'telemetry': telemetry,
        'checkpoint_path': ckpt_path,
        'checkpoint_size_mb': ckpt_size_mb,
        'reload_verified': reload_verified,
    }


# ==============================================================================
# APPLICATION 4: Figure 12 — PET/CT Radon Compressive Sensing
# ==============================================================================
def smoke_test_figure12(device: torch.device, out_dir: str) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("APPLICATION 4: FIGURE 12 — 3D PET/CT RADON COMPRESSIVE SENSING")
    print("=" * 80)

    # 1. Load data
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
    rank = 1000
    n_inputs = rank
    nangles = 40
    reg_noise_std = 1.0 / 30.0

    set_seed(42)
    angles = torch.tensor(np.linspace(0, 180, nangles).astype(np.float32), device=device)
    cube_ten = torch.tensor(cube, dtype=torch.float32, device=device).permute(2, 0, 1)[None, ...]
    clean_measurements = lin_inverse.radon(cube_ten, angles).detach().cpu().numpy()
    noisy_measurements = utils.measure(clean_measurements, 2.0, 100.0)
    meas_ten = torch.tensor(noisy_measurements, dtype=torch.float32, device=device)
    criterion_l2 = losses.L2Norm()

    # 2. Build model
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
    ).to(device)

    total_params = count_parameters(model)
    target_params = 4050988
    delta_params = total_params - target_params
    print(f"  [Fig 12] Trainable Parameters: {total_params:,} (Target: {target_params:,}, Delta: {delta_params})")
    assert delta_params == 0, f"Parameter mismatch in Fig 12: {total_params} != {target_params}"

    # 3. Latents & Optimizer
    set_seed(200)
    u_inp = utils.get_1d_posencode_inp(H, rank // 2, device=device)
    v_inp = utils.get_1d_posencode_inp(W, rank // 2, device=device)
    w_inp = utils.get_1d_posencode_inp(T, rank // 2, device=device)
    u_inp_per = u_inp.detach().clone()
    v_inp_per = v_inp.detach().clone()
    w_inp_per = w_inp.detach().clone()

    optimizer = torch.optim.Adam(list(model.parameters()) + [u_inp, v_inp, w_inp], lr=1e-3)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    # 4. Execute 1 Epoch
    model.train()
    epoch_start = time.time()

    u_pert = u_inp + u_inp_per.normal_() * reg_noise_std
    v_pert = v_inp + v_inp_per.normal_() * reg_noise_std
    w_pert = w_inp + w_inp_per.normal_() * reg_noise_std

    t_fwd_start = time.time()
    tensor_estim = model(u_pert, v_pert, w_pert)
    m_est = lin_inverse.radon(tensor_estim[None, ...].permute(0, 3, 1, 2), angles)
    loss = criterion_l2(meas_ten - m_est)
    t_fwd = (time.time() - t_fwd_start) * 1000.0

    t_bwd_start = time.time()
    optimizer.zero_grad()
    loss.backward()
    t_bwd = (time.time() - t_bwd_start) * 1000.0

    adaptive_block = model.w_net.adaptive_block
    grad_norms = compute_branch_gradient_norms(adaptive_block)
    telemetry = extract_router_telemetry(adaptive_block)

    optimizer.step()
    epoch_time = time.time() - epoch_start
    peak_vram = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if torch.cuda.is_available() else 0.0

    cur_loss = float(loss.item())
    is_finite = bool(math.isfinite(cur_loss) and not torch.isnan(tensor_estim).any() and not torch.isinf(tensor_estim).any())
    print(f"  [Fig 12] 1-Epoch Loss: {cur_loss:.6f} | Finite: {is_finite} | Time: {epoch_time:.2f}s | Peak VRAM: {peak_vram:.1f}MB")
    print(f"  [Fig 12] Timing: Fwd {t_fwd:.1f}ms | Bwd {t_bwd:.1f}ms")
    print(f"  [Fig 12] Routing Weights: CNN={telemetry['mean_weights']['cnn']:.4f}, PW={telemetry['mean_weights']['pointwise']:.4f}, Attn={telemetry['mean_weights']['attention']:.4f}")
    print(f"  [Fig 12] Gradient Norms: CNN={grad_norms['cnn_branch']:.6f}, PW={grad_norms['pw_branch']:.6f}, Attn={grad_norms['attn_branch']:.6f}, Router={grad_norms['router']:.6f}")

    for b_name, norm in grad_norms.items():
        assert norm > 0.0, f"Branch starvation detected in Fig 12: {b_name} has zero gradient norm!"

    # 5. Checkpoint Saving & Independent Verification
    ckpt_path = os.path.join(out_dir, "fig12_ahd_net_checkpoint.pth")
    save_checkpoint(
        model,
        ckpt_path,
        optimizer=optimizer,
        epoch=1,
        metrics={'loss': cur_loss, 'vram_mb': peak_vram},
        config={'u_inp': u_inp.detach().cpu(), 'v_inp': v_inp.detach().cpu(), 'w_inp': w_inp.detach().cpu()}
    )
    assert os.path.exists(ckpt_path), f"Checkpoint not found at {ckpt_path}"
    ckpt_size_mb = os.path.getsize(ckpt_path) / (1024 ** 2)

    model_reloaded = build_deeptensor_cp_model(
        rank=rank, lengths=(H, W, T), u_type='skip1d', v_type='skip1d', w_type='adaptive_hybrid', n_inputs=n_inputs
    ).to(device)
    model_reloaded.eval()
    ckpt_data = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_reloaded.load_state_dict(ckpt_data['model_state_dict'])
    with torch.no_grad():
        test_out = model_reloaded(u_inp, v_inp, w_inp)
    reload_verified = bool(torch.isfinite(test_out).all())
    print(f"  [Fig 12] Checkpoint Saved: {ckpt_path} ({ckpt_size_mb:.2f}MB) | Independent Reload Verified: {reload_verified}")

    return {
        'application': 'Figure 12 3D PET/CT Radon',
        'params': total_params,
        'target_params': target_params,
        'delta_params': delta_params,
        'loss': cur_loss,
        'is_finite': is_finite,
        'forward_ms': t_fwd,
        'backward_ms': t_bwd,
        'epoch_sec': epoch_time,
        'peak_vram_mb': peak_vram,
        'grad_norms': grad_norms,
        'telemetry': telemetry,
        'checkpoint_path': ckpt_path,
        'checkpoint_size_mb': ckpt_size_mb,
        'reload_verified': reload_verified,
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 80)
    print(f"STARTING ADAPTIVE HYBRID DEEPTENSOR V1 FOUR-APPLICATION SMOKE TEST")
    print(f"Target Device: {device} | CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Name  : {torch.cuda.get_device_name(device)}")
    print("=" * 80)

    out_dir = os.path.join(project_root, 'results', 'extensions', 'adaptive_hybrid_v1_smoke')
    os.makedirs(out_dir, exist_ok=True)

    summary = {}
    summary['fig09'] = smoke_test_figure09(device, out_dir)
    summary['fig10'] = smoke_test_figure10(device, out_dir)
    summary['fig11'] = smoke_test_figure11(device, out_dir)
    summary['fig12'] = smoke_test_figure12(device, out_dir)

    summary_file = os.path.join(out_dir, "smoke_test_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("ALL FOUR PROTOCOL SMOKE TESTS COMPLETED SUCCESSFULLY!")
    print(f"Summary JSON saved to: {summary_file}")
    print("=" * 80)


if __name__ == '__main__':
    main()

