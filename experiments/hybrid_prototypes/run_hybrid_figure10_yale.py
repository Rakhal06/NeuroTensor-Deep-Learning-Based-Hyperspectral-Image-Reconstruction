#!/usr/bin/env python
"""
Hybrid CNN-Transformer Prototype: Figure 10 Yale Face B Tensor Denoising
========================================================================
Evaluates Baseline CNN vs. Hybrid Attention vs. Spectral Transformer vs. Control
on 3-Way PARAFAC tensor denoising (192x168x160, Rank R=2000, 30% salt & pepper noise).
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
    build_deeptensor_cp_model,
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


def add_salt_and_pepper_noise(tensor, noise_ratio=0.30):
    noisy = tensor.copy()
    mask = np.random.rand(*tensor.shape)
    salt = mask < (noise_ratio / 2.0)
    pepper = (mask >= (noise_ratio / 2.0)) & (mask < noise_ratio)
    noisy[salt] = 1.0
    noisy[pepper] = 0.0
    return noisy


def run_single_variant(variant: str, epochs: int, clean_tensor: np.ndarray, noisy_tensor: np.ndarray,
                       device: torch.device, out_dir: str):
    H, W, T = clean_tensor.shape
    rank = 2000
    n_inputs = rank
    lr = 1e-3

    noisy_ten = torch.tensor(noisy_tensor, dtype=torch.float32).to(device)
    criterion_l1 = losses.L1Norm()

    # Model architecture selection for mode 3 (temporal/slice factor W)
    if variant == 'baseline':
        w_type = 'skip1d'
    elif variant == 'hybrid':
        w_type = 'hybrid_attention'
    elif variant == 'transformer':
        w_type = 'spectral_transformer'
    elif variant in ('control', 'transformer_control'):
        w_type = 'control_transformer'
    elif variant in ('cnn_control', 'matched_cnn'):
        w_type = 'cnn_control'
    elif variant in ('pointwise_control', 'hybrid_control'):
        w_type = 'hybrid_control'
    else:
        raise ValueError(f"Unknown variant: {variant}")

    set_seed(100)
    model = build_deeptensor_cp_model(
        rank=rank,
        lengths=(H, W, T),
        u_type='skip1d',
        v_type='skip1d',
        w_type=w_type,
        n_inputs=n_inputs,
        learnable_core=True,
        normalize_core=True,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ffn=512,
    ).to(device)

    set_seed(200)
    u_inp = utils.get_1d_posencode_inp(H, rank // 2, device=device)
    v_inp = utils.get_1d_posencode_inp(W, rank // 2, device=device)
    w_inp = utils.get_1d_posencode_inp(T, rank // 2, device=device)

    model.train()
    optimizer = torch.optim.Adam(
        list(model.parameters()) + [u_inp, v_inp, w_inp],
        lr=lr
    )

    best_loss = float('inf')
    best_recon = None
    best_model_state = None
    loss_history = []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    start_time = time.time()
    for idx in range(epochs):
        recon = model(u_inp, v_inp, w_inp)
        loss = criterion_l1(noisy_ten - recon)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cur_loss = loss.item()
        loss_history.append(cur_loss)

        if cur_loss < best_loss:
            best_loss = cur_loss
            best_recon = recon.detach().cpu().numpy()
            best_model_state = copy.deepcopy(model.state_dict())

        if (idx + 1) % 200 == 0 or idx == epochs - 1:
            print(f"    [{variant}] Epoch {idx + 1:4d}/{epochs} | Loss: {cur_loss:.6f} | Best Loss: {best_loss:.6f}")

    runtime = time.time() - start_time
    peak_vram = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if torch.cuda.is_available() else 0.0
    total_params = count_parameters(model)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    ckpt_path = os.path.join(out_dir, f"{variant}_checkpoint.pth")
    save_checkpoint(
        model,
        ckpt_path,
        optimizer=optimizer,
        epoch=epochs,
        metrics={'final_loss': cur_loss, 'best_loss': best_loss},
        config={
            'u_inp': u_inp.detach().cpu(),
            'v_inp': v_inp.detach().cpu(),
            'w_inp': w_inp.detach().cpu(),
        }
    )

    return {
        'recon': best_recon,
        'params': total_params,
        'runtime_sec': runtime,
        'peak_vram_mb': peak_vram,
        'loss_history': loss_history,
    }


def main():
    parser = argparse.ArgumentParser(description="Hybrid CNN-Transformer Prototype for Figure 10 Yale Face B")
    parser.add_argument('--variant', type=str, default='all', choices=['baseline', 'hybrid', 'transformer', 'control', 'cnn_control', 'pointwise_control', 'all'])
    parser.add_argument('--smoke_test', action='store_true', help="Run 10 epochs for fast verification")
    parser.add_argument('--epochs', type=int, default=None)
    args = parser.parse_args()

    out_dir = os.path.join(project_root, 'results', 'hybrid', 'figure10_yale_denoising')
    os.makedirs(out_dir, exist_ok=True)

    epochs = 10 if args.smoke_test else (args.epochs or 1200)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== Running Figure 10 Hybrid Prototype on {device} (Epochs: {epochs}) ===")

    # Load Yale Face B
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

    set_seed(42)
    noisy_tensor = add_salt_and_pepper_noise(clean_tensor, 0.30)

    comparator = SideBySideComparator("figure10_yale_b", out_dir)
    variants = ['baseline', 'hybrid', 'cnn_control', 'pointwise_control', 'transformer'] if args.variant == 'all' else [args.variant]

    for v in variants:
        print(f"--> Training variant: {v}...")
        res = run_single_variant(v, epochs, clean_tensor, noisy_tensor, device, out_dir)
        comparator.add_model_result(
            name=v,
            recon=res['recon'],
            params=res['params'],
            runtime_sec=res['runtime_sec'],
            peak_vram_mb=res['peak_vram_mb'],
            loss_history=res['loss_history'],
        )
        print(f"    Completed {v}: params={res['params']:,}, time={res['runtime_sec']:.2f}s, VRAM={res['peak_vram_mb']:.1f}MB")

    summary = comparator.generate_comparison(
        gt=clean_tensor,
        observation=noisy_tensor,
        slice_idx=T // 2,
        save_figures=True,
    )
    print(f"=== Figure 10 Hybrid Evaluation Complete ===")
    print(f"Summary saved to: {summary['json_path']}")
    print(f"Panel saved to: {summary.get('figure_path', 'N/A')}")


if __name__ == '__main__':
    main()
