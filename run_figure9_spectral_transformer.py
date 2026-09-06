#!/usr/bin/env python

import sys
import os
import argparse
import time
import copy
import random

import numpy as np
from scipy import io
from skimage.metrics import structural_similarity as ssim_func

import torch

sys.path.append('modules')
import utils
import losses
import deep_prior
import spectral
from spectral_transformer import SpectralTransformer, ControlSpectralTransformer, count_parameters

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description='Transformer-Dominant Spectral Generator DeepTensor Extension')
    parser.add_argument('--variant', type=str, choices=['baseline', 'transformer', 'control'], required=True)
    parser.add_argument('--smoke_test', action='store_true')
    args = parser.parse_args()

    os.makedirs('results_spectral_transformer', exist_ok=True)
    
    # --------------------------------------------------
    # Configuration (Identical to frozen baseline)
    # --------------------------------------------------
    expname = 'icvl'
    rank = 20
    n_inputs = rank
    init_nconv = 128
    scaling = 1
    tau = 100
    noise_snr = 2
    
    learning_rate = 1e-3
    epochs = 10 if args.smoke_test else 5000
    reg_noise_std = 1.0/30.0
    exp_weight = 0.99
    
    # --------------------------------------------------
    # Data & Observation Fairness
    # --------------------------------------------------
    set_seed(42) # Global seed 42 for data loading and measure()
    
    data = io.loadmat('data/%s.mat'%expname)
    cube = data['hypercube'].astype(np.float32)
    wavelengths = data['wavelengths'].astype(np.float32).ravel()
    cube = utils.resize(cube/cube.max(), scaling)
    
    H, W, nwvl = cube.shape
    
    hsmat = cube.reshape(H*W, nwvl)
    cube_noisy = utils.measure(cube, noise_snr, tau)
    
    hsten = torch.tensor(hsmat)[None, ...].cuda()
    hsten_noisy = torch.tensor(cube_noisy.reshape(H*W, nwvl))[None, ...].cuda()
    
    # Log baseline observation PSNR to verify fairness
    obs_psnr = utils.psnr(cube, cube_noisy)
    if args.smoke_test and args.variant == 'baseline':
        with open('results_spectral_transformer/sanity_check_baseline.txt', 'w') as f:
            f.write(f"Baseline observation PSNR: {obs_psnr:.4f} dB\n")
    
    criterion_l1 = losses.L1Norm()
    
    # --------------------------------------------------
    # RNG Fairness & Model Construction
    # --------------------------------------------------
    
    # Seed precisely for im_net
    set_seed(100)
    im_net = deep_prior.get_net(n_inputs, 'skip', 'reflection',
                                upsample_mode='bilinear',
                                skip_n33d=128,
                                skip_n33u=128,
                                num_scales=5,
                                n_channels=rank).cuda()
    im_inp = utils.get_inp([1, n_inputs, H, W])
    
    # CRITICAL RNG BOUNDARY FIX
    # Generate spec_inp BEFORE constructing spec_net so different parameter initialization does not consume latent rng sequence
    set_seed(200)
    spec_inp = utils.get_inp([1, n_inputs, nwvl])
    
    if args.variant == 'baseline':
        spec_net = deep_prior.get_net(n_inputs, 'skip1d', 'reflection',
                                    upsample_mode='linear',
                                    skip_n33d=128,
                                    skip_n33u=128,
                                    num_scales=5,
                                    n_channels=rank).cuda()
    elif args.variant == 'transformer':
        spec_net = SpectralTransformer(d_model=128, n_heads=4, n_layers=4, d_ffn=512, seq_len=nwvl, rank=rank).cuda()
    elif args.variant == 'control':
        spec_net = ControlSpectralTransformer(d_model=128, n_heads=4, n_layers=4, d_ffn=512, seq_len=nwvl, rank=rank).cuda()
    
    # --------------------------------------------------
    # Parameter and BIT-EQUALITY fairness checks
    # --------------------------------------------------
    if args.smoke_test:
        N_im = count_parameters(im_net)
        
        # Calculate spec_net params
        if args.variant == 'baseline':
            N_spec = count_parameters(spec_net)
        else:
            N_spec = count_parameters(spec_net)
            
        total_params = N_im + N_spec
        
        # Save hashes/sums to verify bit-equality across variants
        im_inp_sum = im_inp.sum().item()
        spec_inp_sum = spec_inp.sum().item()
        
        # Check ALiBi trainable params
        alibi_trainable = 0
        if args.variant != 'baseline':
            alibi_trainable = sum(p.numel() for n, p in spec_net.named_parameters() if 'alibi' in n and p.requires_grad)
        
        with open(f'results_spectral_transformer/{args.variant}_summary.txt', 'w') as f:
            f.write(f"Variant: {args.variant}\n")
            f.write(f"im_net params: {N_im}\n")
            f.write(f"spec_net params: {N_spec}\n")
            f.write(f"Total params: {total_params}\n")
            f.write(f"im_inp sum: {im_inp_sum:.6f}\n")
            f.write(f"spec_inp sum: {spec_inp_sum:.6f}\n")
            f.write(f"ALiBi trainable params: {alibi_trainable}\n")
            f.write(f"spec_inp shape: {spec_inp.shape}\n")
    
    # --------------------------------------------------
    # Training Preparation
    # --------------------------------------------------
    im_net.train()
    spec_net.train()
    
    net_params = list(im_net.parameters()) + list(spec_net.parameters())
    inp_params = [im_inp] + [spec_inp]
    params = net_params + inp_params
    
    optimizer = torch.optim.Adam(lr=learning_rate, params=params)
    
    im_inp_per = im_inp.detach().clone()
    spec_inp_per = spec_inp.detach().clone()
    
    hs_estim_avg = None
    best_loss = float('inf')
    best_hs_estim = None
    
    loss_array = np.zeros(epochs)
    
    # --------------------------------------------------
    # Training Loop
    # --------------------------------------------------
    torch.cuda.reset_peak_memory_stats()
    
    start_time = time.time()
    for idx in range(epochs):
        im_inp_perturbed = im_inp + im_inp_per.normal_()*reg_noise_std
        spec_inp_perturbed = spec_inp + spec_inp_per.normal_()*reg_noise_std
        
        U_img = im_net(im_inp_perturbed)
        
        if args.smoke_test and args.variant == 'transformer' and idx == 5:
            V, attn_weights = spec_net(spec_inp_perturbed, need_weights=True)
            np.save('results_spectral_transformer/attention_weights_smoke.npy', attn_weights.detach().cpu().numpy())
        else:
            V = spec_net(spec_inp_perturbed)
            
        U = U_img.reshape(-1, rank, H*W).permute(0, 2, 1)
        hs_estim = torch.bmm(U, V)
        
        loss = criterion_l1(hsten_noisy - hs_estim)
        loss_array[idx] = loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_hs_estim = copy.deepcopy(hs_estim.detach().squeeze())
            
        if hs_estim_avg is None:
            hs_estim_avg = hs_estim.detach()
        else:
            hs_estim_avg = hs_estim_avg * exp_weight + hs_estim.detach() * (1 - exp_weight)
            
        if idx % 100 == 0 or (args.smoke_test and idx == epochs - 1):
            print(f'Epoch {idx}/{epochs}, Loss: {loss.item():.6f}')
            
    end_time = time.time()
    runtime = end_time - start_time
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    
    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------
    rec_cube = best_hs_estim.reshape(H, W, nwvl).cpu().numpy()
    
    psnr_val = utils.psnr(cube, rec_cube)
    rmse_val = np.sqrt(np.mean((cube - rec_cube)**2))
    
    ssim_val = 0
    for ch in range(nwvl):
        ssim_val += ssim_func(cube[:,:,ch], rec_cube[:,:,ch], data_range=1.0)
    ssim_val /= nwvl

    print(f"\nFinal Results ({args.variant}):")
    print(f"PSNR: {psnr_val:.4f} dB")
    print(f"SSIM: {ssim_val:.4f}")
    print(f"RMSE: {rmse_val:.6f}")
    print(f"Runtime: {runtime:.2f} s")
    print(f"Peak VRAM: {peak_vram_mb:.2f} MB")
    
    if args.smoke_test:
        with open(f'results_spectral_transformer/{args.variant}_summary.txt', 'a') as f:
            f.write(f"PSNR: {psnr_val:.4f} dB\n")
            f.write(f"SSIM: {ssim_val:.4f}\n")
            f.write(f"RMSE: {rmse_val:.6f}\n")
            f.write(f"Runtime: {runtime:.2f} s\n")
            f.write(f"Peak VRAM: {peak_vram_mb:.2f} MB\n")
            f.write(f"Final Loss: {loss_array[-1]:.6f}\n")
            f.write(f"V shape: {V.shape}\n")
            f.write(f"hs_estim shape: {hs_estim.shape}\n")
    else:
        # Full run outputs
        np.savetxt(f'results_spectral_transformer/loss_{args.variant}.csv', loss_array, delimiter=',')
        
        with open(f'results_spectral_transformer/{args.variant}_full_summary.txt', 'w') as f:
            f.write(f"Variant: {args.variant}\n")
            f.write(f"PSNR: {psnr_val:.4f} dB\n")
            f.write(f"SSIM: {ssim_val:.4f}\n")
            f.write(f"RMSE: {rmse_val:.6f}\n")
            f.write(f"Runtime: {runtime:.2f} s\n")
            f.write(f"Peak VRAM: {peak_vram_mb:.2f} MB\n")
            if args.variant != 'baseline':
                f.write(f"spec_net params: {count_parameters(spec_net)}\n")
        
        if args.variant == 'transformer':
            _, attn_weights = spec_net(spec_inp.detach(), need_weights=True)
            np.save('results_spectral_transformer/attention_weights.npy', attn_weights.detach().cpu().numpy())

if __name__ == '__main__':
    main()
