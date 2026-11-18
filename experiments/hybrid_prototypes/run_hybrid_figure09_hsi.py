#!/usr/bin/env python
"""
Hybrid CNN-Transformer Prototype: Figure 09 Hyperspectral Denoising (ICVL)
==========================================================================
Evaluates Baseline CNN vs. Hybrid Attention vs. Spectral Transformer vs. Control
on hyperspectral cube denoising (128x128x260, Rank R=20, SNR=2, tau=100).
"""

import sys
import os
import argparse
import time
import copy
import random

import numpy as np
from scipy import io
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from deeptensor import utils, losses
from deeptensor.models.factory import (
    build_deeptensor_model,
    count_parameters,
    save_checkpoint,
)
from deeptensor.utils.comparison import SideBySideComparator


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_single_variant(variant: str, epochs: int, cube: np.ndarray, cube_noisy: np.ndarray,
                       device: torch.device, out_dir: str):
    H, W, nwvl = cube.shape
    rank = 20
    n_inputs = rank
    lr = 1e-3
    reg_noise_std = 1.0 / 30.0

    hsmat = cube.reshape(H * W, nwvl)
    hsten_noisy = torch.tensor(cube_noisy.reshape(H * W, nwvl), dtype=torch.float32)[None, ...].to(device)
    criterion_l1 = losses.L1Norm()

    # Model architecture selection
    if variant == 'baseline':
        spectral_type = 'skip1d'
    elif variant == 'hybrid':
        spectral_type = 'hybrid_attention'
    elif variant == 'transformer':
        spectral_type = 'spectral_transformer'
    elif variant in ('control', 'transformer_control'):
        spectral_type = 'control_transformer'
    elif variant in ('cnn_control', 'matched_cnn'):
        spectral_type = 'cnn_control'
    elif variant in ('pointwise_control', 'hybrid_control'):
        spectral_type = 'hybrid_control'
    else:
        raise ValueError(f"Unknown variant: {variant}")

    # Deterministic model initialization
    set_seed(100)
    model = build_deeptensor_model(
        rank=rank,
        seq_len=nwvl,
        spatial_type='skip',
        spectral_type=spectral_type,
        n_inputs=n_inputs,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ffn=512,
    ).to(device)

    set_seed(200)
    im_inp = utils.get_inp([1, n_inputs, H, W]).to(device)
    spec_inp = utils.get_inp([1, n_inputs, nwvl]).to(device)

    model.train()
    optimizer = torch.optim.Adam(
        list(model.parameters()) + [im_inp, spec_inp],
        lr=lr
    )

    im_inp_per = im_inp.detach().clone()
    spec_inp_per = spec_inp.detach().clone()

    best_loss = float('inf')
    best_recon = None
    best_model_state = None
    loss_history = []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    start_time = time.time()
    for idx in range(epochs):
        im_pert = im_inp + im_inp_per.normal_() * reg_noise_std
        spec_pert = spec_inp + spec_inp_per.normal_() * reg_noise_std

        hs_estim = model(im_pert, spec_pert)
        loss = criterion_l1(hsten_noisy - hs_estim)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cur_loss = loss.item()
        loss_history.append(cur_loss)

        if cur_loss < best_loss:
            best_loss = cur_loss
            best_recon = hs_estim.detach().squeeze(0).cpu().numpy().reshape(H, W, nwvl)
            best_model_state = copy.deepcopy(model.state_dict())

        if (idx + 1) % 500 == 0 or idx == epochs - 1:
            print(f"    [{variant}] Epoch {idx + 1:4d}/{epochs} | Loss: {cur_loss:.6f} | Best Loss: {best_loss:.6f}")

    runtime = time.time() - start_time
    peak_vram = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if torch.cuda.is_available() else 0.0
    total_params = count_parameters(model)

    # Save checkpoint with weights corresponding to best unsupervised loss
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    ckpt_path = os.path.join(out_dir, f"{variant}_checkpoint.pth")
    save_checkpoint(model, ckpt_path, optimizer=optimizer, epoch=epochs, metrics={'final_loss': cur_loss, 'best_loss': best_loss})

    return {
        'recon': best_recon,
        'params': total_params,
        'runtime_sec': runtime,
        'peak_vram_mb': peak_vram,
        'loss_history': loss_history,
    }


def main():
    parser = argparse.ArgumentParser(description="Hybrid CNN-Transformer Prototype for Figure 09 HSI")
    parser.add_argument('--variant', type=str, default='all', choices=['baseline', 'hybrid', 'transformer', 'control', 'cnn_control', 'pointwise_control', 'all'])
    parser.add_argument('--smoke_test', action='store_true', help="Run 10 epochs for fast verification")
    parser.add_argument('--epochs', type=int, default=None)
    args = parser.parse_args()

    out_dir = os.path.join(project_root, 'results', 'hybrid', 'figure09_hyperspectral')
    os.makedirs(out_dir, exist_ok=True)

    epochs = 10 if args.smoke_test else (args.epochs or 5000)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== Running Figure 09 Hybrid Prototype on {device} (Epochs: {epochs}) ===")

    # Load ICVL data
    data_path = os.path.join(project_root, 'data', 'icvl.mat')
    data = io.loadmat(data_path)
    cube = data['hypercube'].astype(np.float32)
    cube = utils.resize(cube / cube.max(), 1)
    H, W, nwvl = cube.shape

    set_seed(42)
    cube_noisy = utils.measure(cube, 2, 100)

    comparator = SideBySideComparator("figure09_icvl_hsi", out_dir)
    variants = ['baseline', 'hybrid', 'cnn_control', 'pointwise_control', 'transformer'] if args.variant == 'all' else [args.variant]

    for v in variants:
        print(f"--> Training variant: {v}...")
        res = run_single_variant(v, epochs, cube, cube_noisy, device, out_dir)
        comparator.add_model_result(
            name=v,
            recon=res['recon'],
            params=res['params'],
            runtime_sec=res['runtime_sec'],
            peak_vram_mb=res['peak_vram_mb'],
            loss_history=res['loss_history'],
        )
        print(f"    Completed {v}: params={res['params']:,}, time={res['runtime_sec']:.2f}s, VRAM={res['peak_vram_mb']:.1f}MB")

    # Generate comparison summary and panels
    summary = comparator.generate_comparison(
        gt=cube,
        observation=cube_noisy,
        slice_idx=nwvl // 2,
        save_figures=True,
    )
    print(f"=== Figure 09 Hybrid Evaluation Complete ===")
    print(f"Summary saved to: {summary['json_path']}")
    print(f"Panel saved to: {summary.get('figure_path', 'N/A')}")


if __name__ == '__main__':
    main()

