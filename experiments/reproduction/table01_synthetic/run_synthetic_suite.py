#!/usr/bin/env python
'''
Synthetic Experiments Suite: Figures 3, 4, 8 and Table 1
Paper: "DeepTensor: Low-Rank Tensor Decomposition with Deep Network Priors"
- Figure 3: Inductive bias rejects noise (Signal vs. Noise fit convergence)
- Figure 4: Separable convolutions speedup (2D+1D vs. 3D DIP)
- Figure 8: Stopping criterion analysis (DIP vs. Deep Decoder across noise levels)
- Table 1: Learning rate schedule grid (Fixed, Step, Exp, Cosine across LRs)
'''

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure root directory and modules are on sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from deeptensor import models, utils, losses, deep_prior, deep_decoder

# ==============================================================================
# FIGURE 3: INDUCTIVE BIAS (SIGNAL VS NOISE FITTING)
# ==============================================================================
def run_figure3(args):
    print("\n======================================================================")
    print("FIGURE 3: INDUCTIVE BIAS REJECTS NOISE (SIGNAL VS NOISE)")
    print("======================================================================")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    nrows, ncols, rank = 64, 64, 20
    epochs = args.fig3_epochs

    # Generate clean signal and iid Gaussian noise (sigma=0.01)
    np.random.seed(42)
    torch.manual_seed(42)
    mat_clean, _ = utils.get_matrix(nrows, ncols, rank, 'gaussian', 'piecewise', 0.0, 1000)
    mat_clean = mat_clean / mat_clean.max()
    noise = np.random.randn(nrows, ncols).astype(np.float32) * 0.01
    mat_noisy = mat_clean + noise

    mat_clean_ten = torch.tensor(mat_clean, device=device)[None, ...]
    noise_ten = torch.tensor(noise, device=device)[None, ...]
    mat_noisy_ten = torch.tensor(mat_noisy, device=device)[None, ...]

    u_net = deep_prior.get_net(rank, 'skip1d', 'reflection', upsample_mode='linear',
                               skip_n33d=64, skip_n33u=64, num_scales=5, n_channels=rank).to(device)
    v_net = deep_prior.get_net(rank, 'skip1d', 'reflection', upsample_mode='linear',
                               skip_n33d=64, skip_n33u=64, num_scales=5, n_channels=rank).to(device)
    u_inp = utils.get_inp([1, rank, nrows], device=device)
    v_inp = utils.get_inp([1, rank, ncols], device=device)

    optimizer = torch.optim.Adam(list(u_net.parameters()) + list(v_net.parameters()) + [u_inp, v_inp], lr=1e-3)
    criterion = losses.L2Norm()

    signal_error = []
    noise_error = []

    print(f"Training Rank-{rank} DeepTensor for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        u_out = u_net(u_inp).permute(0, 2, 1)
        v_out = v_net(v_inp)
        mat_estim = torch.bmm(u_out, v_out)

        loss = criterion(mat_estim - mat_noisy_ten)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            s_err = ((mat_estim - mat_clean_ten) ** 2).mean().item()
            n_err = ((mat_estim - noise_ten) ** 2).mean().item()
            signal_error.append(s_err)
            noise_error.append(n_err)

        if epoch % (epochs // 5 if epochs >= 5 else 1) == 0:
            print(f"Epoch [{epoch:4d}/{epochs:4d}] | Signal MSE: {s_err:.4e} | Noise MSE: {n_err:.4e}")

    # Plot Figure 3
    plt.figure(figsize=(6, 4.5))
    plt.plot(signal_error, label='Fit to Signal', color='blue', linewidth=2)
    plt.plot(noise_error, label='Fit to Noise', color='red', linestyle='--', linewidth=2)
    plt.yscale('log')
    plt.xlabel('Training Epochs')
    plt.ylabel('Mean Squared Error (Log Scale)')
    plt.title('Figure 3: Inductive Bias of DeepTensor Rejects Noise')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig3_path = os.path.join(args.output_dir, 'plots', 'figure3_inductive_bias.png')
    plt.savefig(fig3_path, dpi=200)
    plt.close()
    print(f"Saved Figure 3 plot to {fig3_path}")
    return {'signal_mse_final': signal_error[-1], 'noise_mse_final': noise_error[-1]}

# ==============================================================================
# FIGURE 4: SEPARABLE CONVOLUTIONS SPEEDUP
# ==============================================================================
def run_figure4(args):
    print("\n======================================================================")
    print("FIGURE 4: SEPARABLE CONVOLUTIONS SPEEDUP (2D+1D VS 3D)")
    print("======================================================================")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    H, W, D = 32, 32, 32
    rank = 16
    epochs = args.fig4_epochs

    vol = np.random.rand(1, 1, H, W, D).astype(np.float32)
    vol_ten = torch.tensor(vol, device=device)

    # 1. Separable 2D+1D DIP
    u_net = deep_prior.get_net(rank, 'skip', 'reflection', upsample_mode='bilinear',
                               skip_n33d=32, skip_n33u=32, num_scales=3, n_channels=rank).to(device)
    v_net = deep_prior.get_net(rank, 'skip1d', 'reflection', upsample_mode='linear',
                               skip_n33d=32, skip_n33u=32, num_scales=3, n_channels=rank).to(device)
    u_inp = utils.get_inp([1, rank, H, W], device=device)
    v_inp = utils.get_inp([1, rank, D], device=device)
    opt_sep = torch.optim.Adam(list(u_net.parameters()) + list(v_net.parameters()) + [u_inp, v_inp], lr=1e-3)

    t0 = time.time()
    for _ in range(epochs):
        u_out = u_net(u_inp).reshape(1, rank, H * W).permute(0, 2, 1)
        v_out = v_net(v_inp)
        rec = torch.bmm(u_out, v_out).reshape(1, 1, H, W, D)
        loss = ((rec - vol_ten) ** 2).mean()
        opt_sep.zero_grad()
        loss.backward()
        opt_sep.step()
    time_sep = time.time() - t0
    sep_params = sum(p.numel() for p in u_net.parameters()) + sum(p.numel() for p in v_net.parameters())

    # 2. Monolithic 3D Conv Net
    conv3d_net = torch.nn.Sequential(
        torch.nn.Conv3d(1, 32, kernel_size=3, padding=1),
        torch.nn.BatchNorm3d(32),
        torch.nn.LeakyReLU(0.2),
        torch.nn.Conv3d(32, 32, kernel_size=3, padding=1),
        torch.nn.BatchNorm3d(32),
        torch.nn.LeakyReLU(0.2),
        torch.nn.Conv3d(32, 1, kernel_size=3, padding=1)
    ).to(device)
    opt_3d = torch.optim.Adam(conv3d_net.parameters(), lr=1e-3)
    z_3d = utils.get_inp([1, 1, H, W, D], device=device)

    t0 = time.time()
    for _ in range(epochs):
        rec_3d = conv3d_net(z_3d)
        loss3d = ((rec_3d - vol_ten) ** 2).mean()
        opt_3d.zero_grad()
        loss3d.backward()
        opt_3d.step()
    time_3d = time.time() - t0
    conv3d_params = sum(p.numel() for p in conv3d_net.parameters())

    speedup = time_3d / (time_sep + 1e-8)
    print(f"Separable (2D+1D DIP): {time_sep:.2f}s ({sep_params:,} params)")
    print(f"Monolithic 3D ConvNet: {time_3d:.2f}s ({conv3d_params:,} params)")
    print(f"Observed Speedup: {speedup:.1f}x")

    return {
        'separable_runtime': time_sep,
        'monolithic_runtime': time_3d,
        'speedup': speedup,
        'separable_params': sep_params,
        'monolithic_params': conv3d_params
    }

# ==============================================================================
# FIGURE 8: STOPPING CRITERION (DIP VS DEEP DECODER)
# ==============================================================================
def run_figure8(args):
    print("\n======================================================================")
    print("FIGURE 8: STOPPING CRITERION (DIP VS DEEP DECODER)")
    print("======================================================================")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    nrows, ncols, rank = 64, 64, 32
    epochs = args.fig8_epochs
    sigma = 0.2

    np.random.seed(42)
    torch.manual_seed(42)
    mat_clean, _ = utils.get_matrix(nrows, ncols, rank, 'gaussian', 'gaussian', 0.0, 1000)
    mat_clean = mat_clean / mat_clean.max()
    mat_noisy = mat_clean + np.random.randn(nrows, ncols).astype(np.float32) * sigma

    mat_clean_ten = torch.tensor(mat_clean, device=device)[None, ...]
    mat_noisy_ten = torch.tensor(mat_noisy, device=device)[None, ...]

    # Over-parameterized DIP
    u_dip = deep_prior.get_net(rank, 'skip1d', 'reflection', upsample_mode='linear',
                               skip_n33d=64, skip_n33u=64, num_scales=5, n_channels=rank).to(device)
    v_dip = deep_prior.get_net(rank, 'skip1d', 'reflection', upsample_mode='linear',
                               skip_n33d=64, skip_n33u=64, num_scales=5, n_channels=rank).to(device)
    u_dip_inp = utils.get_inp([1, rank, nrows], device=device)
    v_dip_inp = utils.get_inp([1, rank, ncols], device=device)
    opt_dip = torch.optim.Adam(list(u_dip.parameters()) + list(v_dip.parameters()) + [u_dip_inp, v_dip_inp], lr=1e-3)

    # Under-parameterized Deep Decoder
    u_dd = deep_decoder.decodernw1d(rank, [64, 64, 64]).to(device)
    v_dd = deep_decoder.decodernw1d(rank, [64, 64, 64]).to(device)
    u_dd_inp = utils.get_inp([1, 64, nrows // 8], device=device)
    v_dd_inp = utils.get_inp([1, 64, ncols // 8], device=device)
    opt_dd = torch.optim.Adam(list(u_dd.parameters()) + list(v_dd.parameters()) + [u_dd_inp, v_dd_inp], lr=1e-3)

    dip_errors = []
    dd_errors = []

    for epoch in range(1, epochs + 1):
        # DIP step
        u_o = u_dip(u_dip_inp).permute(0, 2, 1)
        v_o = v_dip(v_dip_inp)
        rec_dip = torch.bmm(u_o, v_o)
        l_dip = ((rec_dip - mat_noisy_ten) ** 2).mean()
        opt_dip.zero_grad()
        l_dip.backward()
        opt_dip.step()

        # DD step
        u_d = u_dd(u_dd_inp).permute(0, 2, 1)
        v_d = v_dd(v_dd_inp)
        rec_dd = torch.bmm(u_d, v_d)
        l_dd = ((rec_dd - mat_noisy_ten) ** 2).mean()
        opt_dd.zero_grad()
        l_dd.backward()
        opt_dd.step()

        with torch.no_grad():
            dip_errors.append(((rec_dip - mat_clean_ten) ** 2).mean().item())
            dd_errors.append(((rec_dd - mat_clean_ten) ** 2).mean().item())

    # Plot Figure 8(b)
    plt.figure(figsize=(6, 4.5))
    plt.plot(dip_errors, label='Over-parameterized (DIP)', color='red', linewidth=2)
    plt.plot(dd_errors, label='Under-parameterized (Deep Decoder)', color='blue', linewidth=2)
    plt.yscale('log')
    plt.xlabel('Training Epochs')
    plt.ylabel('Reconstruction Error (MSE)')
    plt.title(f'Figure 8(b): Stopping Behavior (Noise std = {sigma})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig8_path = os.path.join(args.output_dir, 'plots', 'figure8_stopping_criterion.png')
    plt.savefig(fig8_path, dpi=200)
    plt.close()
    print(f"Saved Figure 8 plot to {fig8_path}")

    return {
        'dip_min_error': float(np.min(dip_errors)),
        'dip_optimal_epoch': int(np.argmin(dip_errors) + 1),
        'dd_min_error': float(np.min(dd_errors)),
        'dd_optimal_epoch': int(np.argmin(dd_errors) + 1)
    }

# ==============================================================================
# TABLE 1: LEARNING RATE SCHEDULER GRID
# ==============================================================================
def run_table1(args):
    print("\n======================================================================")
    print("TABLE 1: EFFECT OF LEARNING RATE SCHEDULE")
    print("======================================================================")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    nrows, ncols, rank = 64, 64, 32
    epochs = args.tab1_epochs
    noise_snr = 0.2

    np.random.seed(42)
    mat_clean, _ = utils.get_matrix(nrows, ncols, rank, 'gaussian', 'gaussian', 0.0, 1000)
    mat_clean = mat_clean / mat_clean.max()
    mat_noisy = mat_clean + np.random.randn(nrows, ncols).astype(np.float32) * noise_snr
    mat_noisy_ten = torch.tensor(mat_noisy, device=device)[None, ...]

    svd_psnr = float(utils.psnr(mat_clean, utils.lr_decompose(mat_noisy, rank)))
    print(f"Baseline SVD PSNR: {svd_psnr:.2f} dB (Paper reported: 14.4 dB)")

    schedulers = ['Fixed', 'Step', 'Exponential', 'Cosine annealing']
    max_lrs = [1e-2, 1e-3, 1e-4]
    n_repeats = args.tab1_repeats

    grid_results = {}

    for lr in max_lrs:
        lr_str = f"10^{int(np.log10(lr))}"
        grid_results[lr_str] = {}
        for sched_name in schedulers:
            runs = []
            for r in range(n_repeats):
                torch.manual_seed(42 + r * 10)
                u_net = deep_prior.get_net(rank, 'skip1d', 'reflection', upsample_mode='linear',
                                           skip_n33d=64, skip_n33u=64, num_scales=5, n_channels=rank).to(device)
                v_net = deep_prior.get_net(rank, 'skip1d', 'reflection', upsample_mode='linear',
                                           skip_n33d=64, skip_n33u=64, num_scales=5, n_channels=rank).to(device)
                u_inp = utils.get_inp([1, rank, nrows], device=device)
                v_inp = utils.get_inp([1, rank, ncols], device=device)
                optimizer = torch.optim.Adam(list(u_net.parameters()) + list(v_net.parameters()) + [u_inp, v_inp], lr=lr)

                if sched_name == 'Fixed':
                    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: 1.0)
                elif sched_name == 'Step':
                    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(epochs // 2), gamma=0.99)
                elif sched_name == 'Exponential':
                    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)
                elif sched_name == 'Cosine annealing':
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

                best_mse = float('inf')
                best_recon = None

                for epoch in range(1, epochs + 1):
                    u_out = u_net(u_inp).permute(0, 2, 1)
                    v_out = v_net(v_inp)
                    recon = torch.bmm(u_out, v_out)
                    loss = ((recon - mat_noisy_ten) ** 2).mean()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    loss_val = loss.item()
                    if loss_val < best_mse:
                        best_mse = loss_val
                        best_recon = recon.detach().clone()

                psnr = float(utils.psnr(mat_clean, best_recon.cpu().numpy()))
                runs.append(psnr)

            mean_p = float(np.mean(runs))
            std_p = float(np.std(runs))
            grid_results[lr_str][sched_name] = {'mean': mean_p, 'std': std_p, 'runs': runs}
            print(f"LR {lr_str} | {sched_name:<16} | PSNR: {mean_p:.1f} ± {std_p:.1f} dB")

    return {'svd_psnr': svd_psnr, 'grid': grid_results}

def main():
    parser = argparse.ArgumentParser(description="Synthetic Experiments Suite (Fig 3, 4, 8 and Table 1)")
    parser.add_argument('--fig3_epochs', type=int, default=1500)
    parser.add_argument('--fig4_epochs', type=int, default=200)
    parser.add_argument('--fig8_epochs', type=int, default=1000)
    parser.add_argument('--tab1_epochs', type=int, default=600)
    parser.add_argument('--tab1_repeats', type=int, default=3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default=os.path.join(project_root, 'results', 'baseline', 'table01_synthetic'))
    args = parser.parse_args()

    os.makedirs(os.path.join(args.output_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'images'), exist_ok=True)

    fig3_res = run_figure3(args)
    fig4_res = run_figure4(args)
    fig8_res = run_figure8(args)
    tab1_res = run_table1(args)

    master_results = {
        'figure3': fig3_res,
        'figure4': fig4_res,
        'figure8': fig8_res,
        'table1': tab1_res
    }

    with open(os.path.join(args.output_dir, 'synthetic_suite_metrics.json'), 'w') as fp:
        json.dump(master_results, fp, indent=2)

    print("\nSuccessfully executed and saved all synthetic experiments (Fig 3, 4, 8, Table 1)!")

if __name__ == '__main__':
    main()

