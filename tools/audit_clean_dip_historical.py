"""
Unified Clean DIP vs In-Loop Evaluation Script for Historical Models
====================================================================
Evaluates all historical checkpoints under both Clean DIP inference (train mode,
no input perturbation) and compares directly with in-loop snapshot metrics.
"""

import sys
import os
import json
import numpy as np
from scipy import io
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, os.path.join(project_root, 'src'))
    sys.path.insert(0, project_root)

from deeptensor import utils
from deeptensor.models.factory import build_deeptensor_cp_model, build_deeptensor_model
from deeptensor.utils.comparison import compute_metrics


def audit_figure10(device='cpu'):
    print("\n" + "=" * 80)
    print("FIGURE 10: YALE FACE B — CLEAN DIP EVALUATION OF ALL MODELS")
    print("=" * 80)
    mat = io.loadmat(os.path.join(project_root, 'data', 'yale_b.mat'))
    cube = mat['faces'].astype(np.float32)
    cube = cube / cube.max()

    with open(os.path.join(project_root, 'results', 'hybrid', 'figure10_yale_denoising', 'figure10_yale_b_comparison.json')) as f:
        hist_data = json.load(f)

    print(f"{'Model':<18} | {'In-Loop PSNR':<13} | {'Clean DIP PSNR':<14} | {'Delta (Clean - Loop)':<20} | {'Clean SSIM':<10}")
    print("-" * 80)

    for m_name in ['baseline', 'cnn_control', 'pointwise_control', 'hybrid', 'transformer']:
        ckpt_path = os.path.join(project_root, 'results', 'hybrid', 'figure10_yale_denoising', f'{m_name}_checkpoint.pth')
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        w_type = 'skip1d' if m_name == 'baseline' else ('hybrid_attention' if m_name == 'hybrid' else ('cnn_control' if m_name == 'cnn_control' else ('hybrid_control' if m_name == 'pointwise_control' else 'spectral_transformer')))
        model = build_deeptensor_cp_model(
            rank=2000, lengths=(192, 168, 160), u_type='skip1d', v_type='skip1d', w_type=w_type,
            n_inputs=2000, learnable_core=True, normalize_core=True, d_model=128, n_heads=4, n_layers=4, d_ffn=512
        ).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.train()
        u = ckpt['config']['u_inp'].to(device)
        v = ckpt['config']['v_inp'].to(device)
        w = ckpt['config']['w_inp'].to(device)
        with torch.no_grad():
            recon = model(u, v, w).cpu().numpy()
        m_clean = compute_metrics(cube, recon)
        m_inloop = hist_data['models'][m_name]['metrics']
        delta = m_clean['psnr'] - m_inloop['psnr']
        print(f"{m_name:<18} | {m_inloop['psnr']:<13.4f} | {m_clean['psnr']:<14.4f} | {delta:<+20.4f} | {m_clean['ssim']:<10.4f}")


def audit_figure11(device='cpu'):
    print("\n" + "=" * 80)
    print("FIGURE 11: VIDEO CS — CLEAN DIP EVALUATION OF ALL MODELS")
    print("=" * 80)
    mat = io.loadmat(os.path.join(project_root, 'data', 'cat_video.mat'))
    cube = mat['hypercube'].astype(np.float32)
    cube = utils.resize(cube / cube.max(), 1)
    H, W, totalframes = cube.shape

    with open(os.path.join(project_root, 'results', 'hybrid', 'figure11_video_cs', 'figure11_video_cs_comparison.json')) as f:
        hist_data = json.load(f)

    print(f"{'Model':<18} | {'In-Loop PSNR':<13} | {'Clean DIP PSNR':<14} | {'Delta (Clean - Loop)':<20} | {'Clean SSIM':<10}")
    print("-" * 80)

    for m_name in ['baseline', 'cnn_control', 'pointwise_control', 'hybrid', 'transformer']:
        ckpt_path = os.path.join(project_root, 'results', 'hybrid', 'figure11_video_cs', f'{m_name}_checkpoint.pth')
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        spectral_type = 'skip1d' if m_name == 'baseline' else ('hybrid_attention' if m_name == 'hybrid' else ('cnn_control' if m_name == 'cnn_control' else ('hybrid_control' if m_name == 'pointwise_control' else 'spectral_transformer')))
        model = build_deeptensor_model(
            rank=32, seq_len=totalframes, spatial_type='skip', spectral_type=spectral_type,
            n_inputs=32, skip_n33d=64, skip_n33u=64, d_model=128, n_heads=4, n_layers=4, d_ffn=512
        ).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.train()
        im_inp = ckpt['im_inp'].to(device)
        spec_inp = ckpt['spec_inp'].to(device)
        with torch.no_grad():
            out_mat = model(im_inp, spec_inp)
            recon = out_mat[0].cpu().numpy().reshape(H, W, totalframes)
        m_clean = compute_metrics(cube, recon)
        m_inloop = hist_data['models'][m_name]['metrics']
        delta = m_clean['psnr'] - m_inloop['psnr']
        print(f"{m_name:<18} | {m_inloop['psnr']:<13.4f} | {m_clean['psnr']:<14.4f} | {delta:<+20.4f} | {m_clean['ssim']:<10.4f}")


def audit_figure12(device='cpu'):
    print("\n" + "=" * 80)
    print("FIGURE 12: 3D PET/CT RADON CS — CLEAN DIP EVALUATION OF ALL MODELS")
    print("=" * 80)
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

    with open(os.path.join(project_root, 'results', 'hybrid', 'figure12_pet_ct', 'figure12_pet_ct_comparison.json')) as f:
        hist_data = json.load(f)

    print(f"{'Model':<18} | {'In-Loop PSNR':<13} | {'Clean DIP PSNR':<14} | {'Delta (Clean - Loop)':<20} | {'Clean SSIM':<10}")
    print("-" * 80)

    for m_name in ['baseline', 'hybrid', 'cnn_control', 'pointwise_control', 'transformer']:
        ckpt_path = os.path.join(project_root, 'results', 'hybrid', 'figure12_pet_ct', f'{m_name}_checkpoint.pth')
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        w_type = 'skip1d' if m_name == 'baseline' else ('hybrid_attention' if m_name == 'hybrid' else ('cnn_control' if m_name == 'cnn_control' else ('hybrid_control' if m_name == 'pointwise_control' else 'spectral_transformer')))
        model = build_deeptensor_cp_model(
            rank=1000, lengths=(H, W, T), u_type='skip1d', v_type='skip1d', w_type=w_type,
            n_inputs=1000, learnable_core=True, normalize_core=True, d_model=128, n_heads=4, n_layers=4, d_ffn=512
        ).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.train()
        u = ckpt['u_inp'].to(device)
        v = ckpt['v_inp'].to(device)
        w = ckpt['w_inp'].to(device)
        with torch.no_grad():
            recon = model(u, v, w).cpu().numpy()
        m_clean = compute_metrics(cube, recon)
        m_inloop = hist_data['models'][m_name]['metrics']
        delta = m_clean['psnr'] - m_inloop['psnr']
        print(f"{m_name:<18} | {m_inloop['psnr']:<13.4f} | {m_clean['psnr']:<14.4f} | {delta:<+20.4f} | {m_clean['ssim']:<10.4f}")


if __name__ == '__main__':
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    audit_figure10(dev)
    audit_figure11(dev)
    audit_figure12(dev)

