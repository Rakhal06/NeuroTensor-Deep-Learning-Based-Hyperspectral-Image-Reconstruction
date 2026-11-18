#!/usr/bin/env python
'''
Table 2: DeepTensor NMF on MNIST & CIFAR-10
Paper: "DeepTensor: Low-Rank Tensor Decomposition with Deep Network Priors" (Section 3.3, Page 7)

Evaluates DeepTensor Nonnegative Matrix Factorization against standard NMF (Lee & Seung, 2001)
under severe Rician noise (input PSNR ~ 4.8 dB) across three nonnegativity activations:
1. Softplus
2. Absolute Value
3. ReLU
'''

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.decomposition import NMF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure root directory and modules are on sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from deeptensor import models, utils, losses, deep_prior, deep_decoder

def get_data(dataset_name, n_samples=2048, data_root='./data'):
    os.makedirs(data_root, exist_ok=True)
    if dataset_name.upper() == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
        loader = DataLoader(dataset, batch_size=n_samples, shuffle=False)
        data = next(iter(loader))[0].numpy()[:n_samples]
        # data shape: (N, 1, 28, 28)
    elif dataset_name.upper() == 'CIFAR':
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])
        dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
        loader = DataLoader(dataset, batch_size=n_samples, shuffle=False)
        data = next(iter(loader))[0].numpy()[:n_samples]
        # data shape: (N, 1, 32, 32)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Normalize data to [0, 1]
    data = data.astype(np.float32)
    data -= data.min()
    data /= (data.max() + 1e-8)

    # Add Rician noise: n = 0.3 * (0.3*z1 + z2^2)
    np.random.seed(42)
    z1 = np.random.randn(*data.shape).astype(np.float32)
    z2 = np.random.randn(*data.shape).astype(np.float32)
    rician_noise = 0.3 * (0.3 * z1 + (z2 ** 2))
    noisy_data = np.clip(data + rician_noise, 0.0, None)

    input_psnr = utils.psnr(data, noisy_data)
    print(f"[{dataset_name}] Loaded {n_samples} samples | Shape: {data.shape} | Input Noisy PSNR: {input_psnr:.2f} dB")
    return data, noisy_data, input_psnr

def run_standard_nmf(data, noisy_data, n_components=10):
    N, C, H, W = data.shape
    X_noisy = noisy_data.reshape(N, H * W)
    X_clean = data.reshape(N, H * W)

    model = NMF(n_components=n_components, init='nndsvda', max_iter=500, random_state=42)
    W_mat = model.fit_transform(X_noisy)
    H_mat = model.components_

    recon = np.dot(W_mat, H_mat).reshape(data.shape)
    psnr_val = utils.psnr(data, recon)
    return float(psnr_val), recon

def run_deeptensor_nmf(data, noisy_data, act_u, act_v, act_name, args):
    N, C, H, W = data.shape
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    n_clusters = args.n_clusters
    epochs = args.epochs
    lr = args.learning_rate

    # Data to device
    data_ten = torch.tensor(data, device=device)
    noisy_ten = torch.tensor(noisy_data, device=device)

    # Networks: 1D DIP for U (sample memberships), 2D DIP for V (spatial basis centroids)
    u_net = deep_prior.get_net(
        n_clusters, "skip1d", "reflection",
        upsample_mode="linear",
        skip_n33d=args.init_nconv,
        skip_n33u=args.init_nconv,
        num_scales=5,
        n_channels=n_clusters
    ).to(device)

    v_net = deep_prior.get_net(
        n_clusters, "skip", "reflection",
        upsample_mode="bilinear",
        skip_n33d=args.init_nconv,
        skip_n33u=args.init_nconv,
        num_scales=4,
        n_channels=n_clusters
    ).to(device)

    u_inp = utils.get_inp([1, n_clusters, N], device=device)
    v_inp = utils.get_inp([1, n_clusters, H, W], device=device)

    u_net.train()
    v_net.train()

    optimizer = torch.optim.Adam(list(u_net.parameters()) + list(v_net.parameters()) + [u_inp, v_inp], lr=lr)
    criterion = losses.L2Norm()

    best_loss = float('inf')
    best_recon = None
    loss_history = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        u_out = act_u(u_net(u_inp))       # (1, n_clusters, N)
        v_out = act_v(v_net(v_inp))       # (1, n_clusters, H, W)

        v_mat = v_out.reshape(1, n_clusters, H * W)
        u_mat = u_out.permute(0, 2, 1)    # (1, N, n_clusters)

        recon_flat = torch.bmm(u_mat, v_mat) # (1, N, H*W)
        recon = recon_flat.reshape(1, N, H, W).permute(1, 0, 2, 3) # (N, 1, H, W)

        loss_data = criterion(recon - noisy_ten)
        loss_reg = 0.01 * (v_out.abs().mean() + u_out.abs().mean())
        loss = loss_data + loss_reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if loss_val < best_loss:
            best_loss = loss_val
            best_recon = recon.detach().clone()

        if epoch % args.log_interval == 0 or epoch == epochs:
            cur_psnr = utils.psnr(data, recon.detach().cpu().numpy())
            print(f"[{act_name}] Epoch [{epoch:4d}/{epochs:4d}] | Loss: {loss_val:.4e} | PSNR: {cur_psnr:.2f} dB")

    elapsed_time = time.time() - start_time
    recon_final = best_recon.cpu().numpy()
    final_psnr = float(utils.psnr(data, recon_final))
    rmse = float(np.sqrt(np.mean((data - recon_final) ** 2)))

    return {
        'psnr': final_psnr,
        'rmse': rmse,
        'runtime': elapsed_time,
        'loss_history': loss_history,
        'recon': recon_final
    }

def main():
    parser = argparse.ArgumentParser(description="Table 2: DeepTensor NMF Reproduction")
    parser.add_argument('--dataset', type=str, default='all', choices=['MNIST', 'CIFAR', 'all'])
    parser.add_argument('--n_samples', type=int, default=2048)
    parser.add_argument('--n_clusters', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--init_nconv', type=int, default=128)
    parser.add_argument('--n_runs', type=int, default=3, help="Number of repetitions for mean and std")
    parser.add_argument('--log_interval', type=int, default=250)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default=os.path.join(project_root, 'results', 'baseline', 'table02_nmf'))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    images_dir = os.path.join(args.output_dir, 'images')
    plots_dir = os.path.join(args.output_dir, 'plots')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    datasets_to_run = ['MNIST', 'CIFAR'] if args.dataset == 'all' else [args.dataset]
    activations = {
        'softplus': (torch.nn.Softplus(), torch.nn.Softplus()),
        'abs': (torch.abs, torch.abs),
        'relu': (torch.relu, torch.relu)
    }

    results_table = {}

    for dname in datasets_to_run:
        print(f"\n======================================================================")
        print(f"TABLE 2 NMF BENCHMARK: {dname}")
        print(f"======================================================================")
        data, noisy_data, in_psnr = get_data(dname, n_samples=args.n_samples)

        # Baseline Standard NMF
        nmf_psnr, nmf_recon = run_standard_nmf(data, noisy_data, n_components=args.n_clusters)
        print(f"Standard NMF PSNR: {nmf_psnr:.2f} dB")

        results_table[dname] = {
            'input_psnr': float(in_psnr),
            'standard_nmf': float(nmf_psnr),
            'deeptensor': {}
        }

        # Run DeepTensor with Softplus, Abs, ReLU
        for act_name, (act_u, act_v) in activations.items():
            run_psnrs = []
            for run_idx in range(args.n_runs):
                print(f"\n--- Running DeepTensor [{dname}] [{act_name.upper()}] (Run {run_idx+1}/{args.n_runs}) ---")
                res = run_deeptensor_nmf(data, noisy_data, act_u, act_v, f"{act_name}_run{run_idx+1}", args)
                run_psnrs.append(res['psnr'])

            mean_psnr = float(np.mean(run_psnrs))
            std_psnr = float(np.std(run_psnrs))
            results_table[dname]['deeptensor'][act_name] = {
                'mean_psnr': mean_psnr,
                'std_psnr': std_psnr,
                'runs': run_psnrs
            }
            print(f"[{dname}] DeepTensor {act_name.upper()}: {mean_psnr:.2f} ± {std_psnr:.2f} dB")

    # Save JSON and Text Summary
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as fp:
        json.dump(results_table, fp, indent=2)

    summary_text = f"""======================================================================
TABLE 2: NMF REPRODUCTION RESULTS (Mean ± Std over {args.n_runs} runs)
======================================================================
DATASET       ACTIVATION FUNCTION    REPRODUCED (OURS)       PAPER REPORTED
----------------------------------------------------------------------
MNIST         Softplus               {results_table['MNIST']['deeptensor']['softplus']['mean_psnr']:.2f} ± {results_table['MNIST']['deeptensor']['softplus']['std_psnr']:.2f} dB       7.40 ± 0.03 dB
MNIST         Abs. value             {results_table['MNIST']['deeptensor']['abs']['mean_psnr']:.2f} ± {results_table['MNIST']['deeptensor']['abs']['std_psnr']:.2f} dB       7.30 ± 0.01 dB
MNIST         ReLU                   {results_table['MNIST']['deeptensor']['relu']['mean_psnr']:.2f} ± {results_table['MNIST']['deeptensor']['relu']['std_psnr']:.2f} dB       7.60 ± 0.15 dB
MNIST         Standard NMF           {results_table['MNIST']['standard_nmf']:.2f} dB                 7.50 dB
----------------------------------------------------------------------
CIFAR-10      Softplus               {results_table['CIFAR']['deeptensor']['softplus']['mean_psnr']:.2f} ± {results_table['CIFAR']['deeptensor']['softplus']['std_psnr']:.2f} dB       5.70 ± 0.03 dB
CIFAR-10      Abs. value             {results_table['CIFAR']['deeptensor']['abs']['mean_psnr']:.2f} ± {results_table['CIFAR']['deeptensor']['abs']['std_psnr']:.2f} dB       5.70 ± 0.02 dB
CIFAR-10      ReLU                   {results_table['CIFAR']['deeptensor']['relu']['mean_psnr']:.2f} ± {results_table['CIFAR']['deeptensor']['relu']['std_psnr']:.2f} dB       6.30 ± 0.04 dB
CIFAR-10      Standard NMF           {results_table['CIFAR']['standard_nmf']:.2f} dB                 5.60 dB
======================================================================
"""
    print(summary_text)
    with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as fp:
        fp.write(summary_text)

if __name__ == '__main__':
    main()

