#!/usr/bin/env python
'''
Figure 4 Improved Benchmark: Separable Convolutions Speedup on 3D Volume
Paper: "DeepTensor: Low-Rank Tensor Decomposition with Deep Network Priors" (Section 3.2, Page 4 & Figure 4)

Target Specifications:
- Volume: 3D MRI volume size: 128 x 128 x 150 (2,457,600 voxels), correcting previous 64 x 64 x 32 toy volume.
- Compares:
  1. Three 1D DIP networks: f_U(z_U) (x) f_V(z_V) (x) f_W(z_W)
  2. One 2D DIP + One 1D DIP network: f_UV(z_UV) (x) f_W(z_W)
  3. One full 3D ConvNet DIP network: f_UVW(z_UVW)
- Rationale for Speedup:
  Separable dimensionality splitting reduces convolutional FLOPs from O(H W D K^3 C^2)
  down to O((H + W + D) K C^2), achieving ~2 orders of magnitude speedup on standard GPU hardware.
- Outputs:
  - figure4_improved_benchmark_metrics.json
  - plots/figure4_separable_convolutions_speedup_improved.png
  - progression_time_vs_mse.csv
'''

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from deeptensor import utils, deep_prior

class Conv3D_DIP(nn.Module):
    '''Full 3D Convolutional DIP network for volumetric fitting'''
    def __init__(self, in_channels=16, num_channels=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(num_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(num_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(num_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(num_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(num_channels, 1, kernel_size=3, padding=1)
        )
    def forward(self, x):
        return self.net(x)

def generate_mri_phantom_volume(H=128, W=128, D=150, rank=16, seed=42):
    '''Generates synthetic smooth volumetric low-rank MRI target matching paper dimensions'''
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Generate smooth 1D spatial bases
    t_h = np.linspace(-1, 1, H)[:, None]
    t_w = np.linspace(-1, 1, W)[:, None]
    t_d = np.linspace(-1, 1, D)[:, None]
    
    U = np.zeros((H, rank), dtype=np.float32)
    V = np.zeros((W, rank), dtype=np.float32)
    W_fac = np.zeros((D, rank), dtype=np.float32)
    
    for r in range(rank):
        freq = (r + 1) * 0.8
        U[:, r] = np.sin(np.pi * freq * t_h.squeeze()) * np.exp(-t_h.squeeze()**2 * 2.0)
        V[:, r] = np.cos(np.pi * freq * t_w.squeeze()) * np.exp(-t_w.squeeze()**2 * 2.0)
        W_fac[:, r] = np.sin(np.pi * freq * 0.5 * t_d.squeeze()) * np.exp(-t_d.squeeze()**2 * 1.5)
        
    vol = np.einsum('ir, jr, kr -> ijk', U, V, W_fac)
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
    return vol

def run_benchmark(epochs=100, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print("=" * 80)
    print("FIGURE 4 IMPROVED: SEPARABLE CONVOLUTIONS SPEEDUP BENCHMARK (128 x 128 x 150)")
    print("=" * 80, flush=True)

    H, W, D = 128, 128, 150
    rank = 16
    print(f"Volume Dimensions: {H} x {W} x {D} ({H * W * D:,} voxels) | Rank: {rank} | Device: {device}")

    target_vol = generate_mri_phantom_volume(H, W, D, rank=rank)
    target_ten = torch.tensor(target_vol, device=device, dtype=torch.float32)

    # -------------------------------------------------------------
    # 1. Three 1D DIP Networks
    # -------------------------------------------------------------
    print("\n[1/3] Benchmarking Method 1: Three 1D Networks (f_U (x) f_V (x) f_W)...", flush=True)
    u_inp = utils.get_1d_posencode_inp(H, max(1, rank // 2), device=device)
    v_inp = utils.get_1d_posencode_inp(W, max(1, rank // 2), device=device)
    w_inp = utils.get_1d_posencode_inp(D, max(1, rank // 2), device=device)

    u_net = deep_prior.get_net(rank, 'skip1d', 'reflection', 'linear', n_channels=rank, skip_n33d=32, skip_n33u=32, num_scales=4).to(device)
    v_net = deep_prior.get_net(rank, 'skip1d', 'reflection', 'linear', n_channels=rank, skip_n33d=32, skip_n33u=32, num_scales=4).to(device)
    w_net = deep_prior.get_net(rank, 'skip1d', 'reflection', 'linear', n_channels=rank, skip_n33d=32, skip_n33u=32, num_scales=4).to(device)
    core = torch.ones(rank, device=device, requires_grad=True)

    params_1d = list(u_net.parameters()) + list(v_net.parameters()) + list(w_net.parameters()) + [u_inp, v_inp, w_inp, core]
    opt_3_1d = torch.optim.Adam(params_1d, lr=1e-3)
    loss_3_1d, time_3_1d = [], []

    # Warmup
    for _ in range(5):
        u = u_net(u_inp)[0].permute(1, 0)
        v = v_net(v_inp)[0].permute(1, 0)
        w = w_net(w_inp)[0].permute(1, 0)
        rec = torch.einsum('r, ir, jr, kr -> ijk', core / rank, u, v, w)
        l = ((rec - target_ten) ** 2).mean()
        opt_3_1d.zero_grad()
        l.backward()
        opt_3_1d.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.time()
    for ep in range(epochs):
        u = u_net(u_inp)[0].permute(1, 0)
        v = v_net(v_inp)[0].permute(1, 0)
        w = w_net(w_inp)[0].permute(1, 0)
        rec = torch.einsum('r, ir, jr, kr -> ijk', core / rank, u, v, w)
        l = ((rec - target_ten) ** 2).mean()
        opt_3_1d.zero_grad()
        l.backward()
        opt_3_1d.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        time_3_1d.append(time.time() - t0)
        loss_3_1d.append(l.item())

    total_time_3_1d = time.time() - t0
    params_count_3_1d = sum(p.numel() for p in u_net.parameters()) + sum(p.numel() for p in v_net.parameters()) + sum(p.numel() for p in w_net.parameters())
    vram_3_1d = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if torch.cuda.is_available() else 0.0
    print(f"  Three 1D Networks: {total_time_3_1d:.2f}s ({total_time_3_1d / epochs * 1000:.1f} ms/epoch) | Final MSE: {loss_3_1d[-1]:.4e} | Params: {params_count_3_1d:,} | VRAM: {vram_3_1d:.1f} MB", flush=True)

    del u_net, v_net, w_net, u_inp, v_inp, w_inp, core, opt_3_1d, rec, l
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -------------------------------------------------------------
    # 2. One 2D + One 1D DIP Networks
    # -------------------------------------------------------------
    print("\n[2/3] Benchmarking Method 2: One 2D + One 1D Network (f_UV (x) f_W)...", flush=True)
    uv_inp = utils.get_inp([1, rank, H, W], device=device)
    w_inp2 = utils.get_1d_posencode_inp(D, max(1, rank // 2), device=device)

    uv_net = deep_prior.get_net(rank, 'skip', 'reflection', 'bilinear', n_channels=rank, skip_n33d=32, skip_n33u=32, num_scales=4).to(device)
    w_net2 = deep_prior.get_net(rank, 'skip1d', 'reflection', 'linear', n_channels=rank, skip_n33d=32, skip_n33u=32, num_scales=4).to(device)

    params_2d_1d = list(uv_net.parameters()) + list(w_net2.parameters()) + [uv_inp, w_inp2]
    opt_2d_1d = torch.optim.Adam(params_2d_1d, lr=1e-3)
    loss_2d_1d, time_2d_1d = [], []

    # Warmup
    for _ in range(5):
        uv = uv_net(uv_inp).reshape(1, rank, H * W).permute(0, 2, 1)
        w = w_net2(w_inp2)[0].permute(1, 0)
        rec = torch.bmm(uv, w.T[None, ...]).reshape(H, W, D)
        l = ((rec - target_ten) ** 2).mean()
        opt_2d_1d.zero_grad()
        l.backward()
        opt_2d_1d.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.time()
    for ep in range(epochs):
        uv = uv_net(uv_inp).reshape(1, rank, H * W).permute(0, 2, 1)
        w = w_net2(w_inp2)[0].permute(1, 0)
        rec = torch.bmm(uv, w.T[None, ...]).reshape(H, W, D)
        l = ((rec - target_ten) ** 2).mean()
        opt_2d_1d.zero_grad()
        l.backward()
        opt_2d_1d.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        time_2d_1d.append(time.time() - t0)
        loss_2d_1d.append(l.item())

    total_time_2d_1d = time.time() - t0
    params_count_2d_1d = sum(p.numel() for p in uv_net.parameters()) + sum(p.numel() for p in w_net2.parameters())
    vram_2d_1d = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if torch.cuda.is_available() else 0.0
    print(f"  One 2D + One 1D:  {total_time_2d_1d:.2f}s ({total_time_2d_1d / epochs * 1000:.1f} ms/epoch) | Final MSE: {loss_2d_1d[-1]:.4e} | Params: {params_count_2d_1d:,} | VRAM: {vram_2d_1d:.1f} MB", flush=True)

    del uv_net, w_net2, uv_inp, w_inp2, opt_2d_1d, rec, l
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -------------------------------------------------------------
    # 3. One Full 3D DIP Network
    # -------------------------------------------------------------
    print("\n[3/3] Benchmarking Method 3: Full 3D ConvNet DIP (f_UVW)...", flush=True)
    z_3d = torch.randn(1, 16, H, W, D, device=device)
    net_3d = Conv3D_DIP(in_channels=16, num_channels=32).to(device)
    opt_3d = torch.optim.Adam(list(net_3d.parameters()) + [z_3d], lr=1e-3)
    loss_3d, time_3d = [], []

    # Warmup
    for _ in range(5):
        rec_3d = net_3d(z_3d)[0, 0]
        l = ((rec_3d - target_ten) ** 2).mean()
        opt_3d.zero_grad()
        l.backward()
        opt_3d.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.time()
    for ep in range(epochs):
        rec_3d = net_3d(z_3d)[0, 0]
        l = ((rec_3d - target_ten) ** 2).mean()
        opt_3d.zero_grad()
        l.backward()
        opt_3d.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        time_3d.append(time.time() - t0)
        loss_3d.append(l.item())

    total_time_3d = time.time() - t0
    params_count_3d = sum(p.numel() for p in net_3d.parameters())
    vram_3d = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if torch.cuda.is_available() else 0.0
    print(f"  Full 3D ConvNet:  {total_time_3d:.2f}s ({total_time_3d / epochs * 1000:.1f} ms/epoch) | Final MSE: {loss_3d[-1]:.4e} | Params: {params_count_3d:,} | VRAM: {vram_3d:.1f} MB", flush=True)

    del net_3d, z_3d, opt_3d, rec_3d, l
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -------------------------------------------------------------
    # Save Metrics & Plot Figure 4
    # -------------------------------------------------------------
    out_dir = os.path.join(project_root, 'results', 'baseline', 'table01_synthetic')
    plots_dir = os.path.join(out_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    speedup_1d_vs_3d = total_time_3d / (total_time_3_1d + 1e-8)
    speedup_2d_vs_3d = total_time_3d / (total_time_2d_1d + 1e-8)

    print("\n" + "=" * 80)
    print("FIGURE 4 BENCHMARK SUMMARY (128 x 128 x 150):")
    print("=" * 80)
    print(f"Method 1 (Three 1D Nets):     Runtime: {total_time_3_1d:6.2f}s | Speedup vs 3D: {speedup_1d_vs_3d:5.1f}x")
    print(f"Method 2 (One 2D + One 1D):   Runtime: {total_time_2d_1d:6.2f}s | Speedup vs 3D: {speedup_2d_vs_3d:5.1f}x")
    print(f"Method 3 (Full 3D ConvNet):   Runtime: {total_time_3d:6.2f}s | Speedup vs 3D:  1.0x (Reference)")
    print("=" * 80)

    # Plot
    plt.figure(figsize=(8, 5.5))
    plt.plot(time_3_1d, loss_3_1d, label=f'Three 1D Networks ({total_time_3_1d:.2f}s, {speedup_1d_vs_3d:.1f}x faster)', color='navy', linewidth=2.5)
    plt.plot(time_2d_1d, loss_2d_1d, label=f'One 2D + One 1D Network ({total_time_2d_1d:.2f}s, {speedup_2d_vs_3d:.1f}x faster)', color='forestgreen', linewidth=2.2)
    plt.plot(time_3d, loss_3d, label=f'Full 3D ConvNet ({total_time_3d:.2f}s, Reference)', color='crimson', linestyle='--', linewidth=2.2)
    plt.yscale('log')
    plt.xlabel('Training Time (seconds)', fontsize=11)
    plt.ylabel('Mean Squared Error (MSE)', fontsize=11)
    plt.title('Figure 4: Computational Speedup from Separable Convolutions\n(Volume Size: $128 \\times 128 \\times 150$, CUDA)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()

    fig4_plot_path = os.path.join(plots_dir, 'figure4_separable_convolutions_speedup_improved.png')
    plt.savefig(fig4_plot_path, dpi=200)
    plt.close()
    print(f"Saved improved Figure 4 plot to: {fig4_plot_path}")

    res = {
        'volume_dimensions': [H, W, D],
        'total_voxels': H * W * D,
        'decomposition_rank': rank,
        'epochs': epochs,
        'three_1d': {'runtime_sec': total_time_3_1d, 'ms_per_epoch': total_time_3_1d / epochs * 1000, 'final_mse': loss_3_1d[-1], 'params': params_count_3_1d, 'vram_mb': vram_3_1d},
        'one_2d_one_1d': {'runtime_sec': total_time_2d_1d, 'ms_per_epoch': total_time_2d_1d / epochs * 1000, 'final_mse': loss_2d_1d[-1], 'params': params_count_2d_1d, 'vram_mb': vram_2d_1d},
        'full_3d': {'runtime_sec': total_time_3d, 'ms_per_epoch': total_time_3d / epochs * 1000, 'final_mse': loss_3d[-1], 'params': params_count_3d, 'vram_mb': vram_3d},
        'speedup_three_1d_vs_3d': speedup_1d_vs_3d,
        'speedup_2d_1d_vs_3d': speedup_2d_vs_3d
    }

    metrics_json_path = os.path.join(out_dir, 'figure4_improved_benchmark_metrics.json')
    with open(metrics_json_path, 'w') as fp:
        json.dump(res, fp, indent=2)
    print(f"Saved benchmark metrics JSON to: {metrics_json_path}")

    # Progression CSV
    csv_path = os.path.join(out_dir, 'figure4_progression_time_vs_mse.csv')
    with open(csv_path, 'w') as f:
        f.write("epoch,time_3_1d,mse_3_1d,time_2d_1d,mse_2d_1d,time_3d,mse_3d\n")
        for i in range(epochs):
            f.write(f"{i+1},{time_3_1d[i]:.4f},{loss_3_1d[i]:.6e},{time_2d_1d[i]:.4f},{loss_2d_1d[i]:.6e},{time_3d[i]:.4f},{loss_3d[i]:.6e}\n")
    print(f"Saved progression CSV to: {csv_path}")

    return res

if __name__ == '__main__':
    run_benchmark()

