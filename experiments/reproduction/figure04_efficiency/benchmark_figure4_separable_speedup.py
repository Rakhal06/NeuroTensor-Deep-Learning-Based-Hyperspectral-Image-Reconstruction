#!/usr/bin/env python
'''
Figure 4: Separable Convolutions Speedup on 3D Volume
Paper: "DeepTensor: Low-Rank Tensor Decomposition with Deep Network Priors" (Section 3.2, Page 4)

Compares:
1. Three 1D DIP networks: f_U(z_U) (x) f_V(z_V) (x) f_W(z_W)
2. One 2D DIP + One 1D DIP network: f_UV(z_UV) (x) f_W(z_W)
3. One full 3D ConvNet DIP network: f_UVW(z_UVW)

Plots MSE vs Wall-Clock Training Time (seconds) to empirically demonstrate
the ~2 orders of magnitude speedup offered by dimensionality splitting.
'''

import os
import sys
import time
import json
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

def run_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("======================================================================", flush=True)
    print("FIGURE 4: SEPARABLE CONVOLUTIONS SPEEDUP BENCHMARK (CUDA)", flush=True)
    print("======================================================================", flush=True)

    H, W, D = 64, 64, 32
    rank = 16
    epochs = 100

    # Synthetic low-rank 3D target volume
    np.random.seed(42)
    torch.manual_seed(42)
    u_true = np.random.randn(H, rank).astype(np.float32)
    v_true = np.random.randn(W, rank).astype(np.float32)
    w_true = np.random.randn(D, rank).astype(np.float32)
    target_vol = np.einsum('ir, jr, kr -> ijk', u_true, v_true, w_true)
    target_vol = (target_vol - target_vol.min()) / (target_vol.max() - target_vol.min())
    target_ten = torch.tensor(target_vol, device=device, dtype=torch.float32)

    # -------------------------------------------------------------
    # 1. Three 1D DIP Networks
    # -------------------------------------------------------------
    print("Running Method 1: Three 1D Networks...", flush=True)
    u_inp = utils.get_1d_posencode_inp(H, rank // 2, device=device)
    v_inp = utils.get_1d_posencode_inp(W, rank // 2, device=device)
    w_inp = utils.get_1d_posencode_inp(D, rank // 2, device=device)
    u_net = deep_prior.get_net(rank, 'skip1d', 'reflection', 'linear', n_channels=rank, skip_n33d=32, skip_n33u=32, num_scales=3).to(device)
    v_net = deep_prior.get_net(rank, 'skip1d', 'reflection', 'linear', n_channels=rank, skip_n33d=32, skip_n33u=32, num_scales=3).to(device)
    w_net = deep_prior.get_net(rank, 'skip1d', 'reflection', 'linear', n_channels=rank, skip_n33d=32, skip_n33u=32, num_scales=3).to(device)
    core = torch.ones(rank, device=device, requires_grad=True)

    opt_3_1d = torch.optim.Adam(list(u_net.parameters()) + list(v_net.parameters()) + list(w_net.parameters()) + [u_inp, v_inp, w_inp, core], lr=1e-3)
    loss_3_1d, time_3_1d = [], []

    # Warmup
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
        torch.cuda.synchronize()
        time_3_1d.append(time.time() - t0)
        loss_3_1d.append(l.item())
    total_time_3_1d = time.time() - t0
    params_3_1d = sum(p.numel() for p in u_net.parameters()) + sum(p.numel() for p in v_net.parameters()) + sum(p.numel() for p in w_net.parameters())
    print(f"  Three 1D Networks: {total_time_3_1d:.2f}s | Final MSE: {loss_3_1d[-1]:.4e} | Params: {params_3_1d:,}", flush=True)

    # -------------------------------------------------------------
    # 2. One 2D + One 1D DIP Networks
    # -------------------------------------------------------------
    print("Running Method 2: One 2D + One 1D Network...", flush=True)
    uv_inp = utils.get_inp([1, rank, H, W], device=device)
    w_inp2 = utils.get_1d_posencode_inp(D, rank // 2, device=device)
    uv_net = deep_prior.get_net(rank, 'skip', 'reflection', 'bilinear', n_channels=rank, skip_n33d=32, skip_n33u=32, num_scales=3).to(device)
    w_net2 = deep_prior.get_net(rank, 'skip1d', 'reflection', 'linear', n_channels=rank, skip_n33d=32, skip_n33u=32, num_scales=3).to(device)

    opt_2d_1d = torch.optim.Adam(list(uv_net.parameters()) + list(w_net2.parameters()) + [uv_inp, w_inp2], lr=1e-3)
    loss_2d_1d, time_2d_1d = [], []

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
        torch.cuda.synchronize()
        time_2d_1d.append(time.time() - t0)
        loss_2d_1d.append(l.item())
    total_time_2d_1d = time.time() - t0
    params_2d_1d = sum(p.numel() for p in uv_net.parameters()) + sum(p.numel() for p in w_net2.parameters())
    print(f"  One 2D + One 1D:  {total_time_2d_1d:.2f}s | Final MSE: {loss_2d_1d[-1]:.4e} | Params: {params_2d_1d:,}", flush=True)

    # -------------------------------------------------------------
    # 3. One Full 3D DIP Network
    # -------------------------------------------------------------
    print("Running Method 3: Full 3D ConvNet DIP...", flush=True)
    z_3d = torch.randn(1, 16, H, W, D, device=device)
    net_3d = Conv3D_DIP(in_channels=16, num_channels=32).to(device)
    opt_3d = torch.optim.Adam(list(net_3d.parameters()) + [z_3d], lr=1e-3)
    loss_3d, time_3d = [], []

    torch.cuda.synchronize()
    t0 = time.time()
    for ep in range(epochs):
        rec_3d = net_3d(z_3d)[0, 0]
        l = ((rec_3d - target_ten) ** 2).mean()
        opt_3d.zero_grad()
        l.backward()
        opt_3d.step()
        torch.cuda.synchronize()
        time_3d.append(time.time() - t0)
        loss_3d.append(l.item())
    total_time_3d = time.time() - t0
    params_3d = sum(p.numel() for p in net_3d.parameters())
    print(f"  Full 3D ConvNet:  {total_time_3d:.2f}s | Final MSE: {loss_3d[-1]:.4e} | Params: {params_3d:,}", flush=True)

    # -------------------------------------------------------------
    # Save Metrics & Plot Figure 4
    # -------------------------------------------------------------
    out_dir = os.path.join(project_root, 'results', 'baseline', 'table01_synthetic')
    plots_dir = os.path.join(out_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(time_3_1d, loss_3_1d, label=f'Three 1D Networks ({total_time_3_1d:.2f}s)', color='blue', linewidth=2.2)
    plt.plot(time_2d_1d, loss_2d_1d, label=f'One 2D + One 1D Network ({total_time_2d_1d:.2f}s)', color='green', linewidth=2.0)
    plt.plot(time_3d, loss_3d, label=f'Full 3D ConvNet ({total_time_3d:.2f}s)', color='red', linestyle='--', linewidth=2.0)
    plt.yscale('log')
    plt.xlabel('Training Time (seconds)', fontsize=11)
    plt.ylabel('Mean Squared Error (MSE)', fontsize=11)
    plt.title('Figure 4: Computational Speedup from Separable Convolutions (CUDA)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()

    fig4_plot_path = os.path.join(plots_dir, 'figure4_separable_convolutions_speedup.png')
    plt.savefig(fig4_plot_path, dpi=200)
    plt.close()
    print(f"Saved Figure 4 plot to {fig4_plot_path}", flush=True)

    res = {
        'three_1d': {'runtime': total_time_3_1d, 'final_mse': loss_3_1d[-1], 'params': params_3_1d},
        'one_2d_one_1d': {'runtime': total_time_2d_1d, 'final_mse': loss_2d_1d[-1], 'params': params_2d_1d},
        'full_3d': {'runtime': total_time_3d, 'final_mse': loss_3d[-1], 'params': params_3d},
        'speedup_separable_vs_3d': total_time_3d / (total_time_3_1d + 1e-8)
    }
    with open(os.path.join(out_dir, 'figure4_benchmark_metrics.json'), 'w') as fp:
        json.dump(res, fp, indent=2)

    return res

if __name__ == '__main__':
    run_benchmark()

