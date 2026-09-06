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
from attention_spectral import HybridSpectralNet, count_parameters, estimate_flops

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description='Hybrid CNN-Transformer DeepTensor Extension')
    parser.add_argument('--variant', type=str, choices=['baseline', 'hybrid', 'control'], required=True)
    parser.add_argument('--smoke_test', action='store_true')
    args = parser.parse_args()

    os.makedirs('results_hybrid', exist_ok=True)
    
    # --------------------------------------------------
    # Configuration (Identical to run_figure9.py)
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
        with open('results_hybrid/sanity_check_baseline.txt', 'w') as f:
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
    
    # Seed precisely for spec_net
    set_seed(200)
    spec_net_base = deep_prior.get_net(n_inputs, 'skip1d', 'reflection',
                                upsample_mode='linear',
                                skip_n33d=128,
                                skip_n33u=128,
                                num_scales=5,
                                n_channels=rank).cuda()
    spec_inp = utils.get_inp([1, n_inputs, nwvl])
    
    # Apply variant wrapper
    if args.variant == 'baseline':
        spec_net = spec_net_base
    else:
        spec_net = HybridSpectralNet(spec_net_base, variant=args.variant).cuda()
    
    # --------------------------------------------------
    # Parameter and FLOPs checks
    # --------------------------------------------------
    if args.smoke_test:
        N_im = count_parameters(im_net)
        N_spec = count_parameters(spec_net)
        total_params = N_im + N_spec
        
        flops = estimate_flops(spec_net, variant=args.variant) if args.variant != 'baseline' else 0
        
        with open(f'results_hybrid/{args.variant}_summary.txt', 'w') as f:
            f.write(f"Variant: {args.variant}\n")
            f.write(f"im_net params: {N_im}\n")
            f.write(f"spec_net params: {N_spec}\n")
            f.write(f"Total params: {total_params}\n")
            if args.variant != 'baseline':
                f.write(f"Extra FLOPs estimated: {flops}\n")
    
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
        
        if args.smoke_test and args.variant == 'hybrid' and idx == 5:
            V, attn_weights = spec_net(spec_inp_perturbed, need_weights=True)
            np.save('results_hybrid/attention_weights_hybrid_smoke.npy', attn_weights.detach().cpu().numpy())
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
            
    end_time = time.time()
    runtime = end_time - start_time
    max_vram = torch.cuda.max_memory_allocated() / (1024**2) # in MB
    
    # --------------------------------------------------
    # Post-Training Attention Diagnostic
    # --------------------------------------------------
    if not args.smoke_test and args.variant == 'hybrid':
        with torch.no_grad():
            _, attn_weights = spec_net(spec_inp, need_weights=True)
            np.save('results_hybrid/attention_weights_hybrid.npy', attn_weights.detach().cpu().numpy())
            
    # --------------------------------------------------
    # Validation & Logging
    # --------------------------------------------------
    if args.smoke_test:
        with open(f'results_hybrid/{args.variant}_summary.txt', 'a') as f:
            f.write(f"Max VRAM (MB): {max_vram:.2f}\n")
            f.write(f"Final loss: {loss_array[-1]:.4f}\n")
            # Shape checks
            f.write(f"V shape: {list(V.shape)}\n")
            f.write(f"hs_estim shape: {list(hs_estim.shape)}\n")
            f.write("Finite checks: " + ("Pass" if torch.isfinite(loss) else "Fail") + "\n")
            
            # Verify BatchNorm train mode propagation
            bn_train = True
            for m in spec_net.modules():
                if isinstance(m, torch.nn.BatchNorm1d):
                    if not m.training:
                        bn_train = False
                        break
            f.write(f"BatchNorms in train mode: {bn_train}\n")
            
    else:
        # Full run validation
        cube_estim = best_hs_estim.cpu().numpy().reshape(H, W, nwvl)
        final_psnr = utils.psnr(cube, cube_estim)
        data_range = cube.max() - cube.min()
        final_ssim = ssim_func(cube, cube_estim, channel_axis=-1, data_range=data_range)
        final_rmse = np.sqrt(np.mean((cube - cube_estim)**2))
        
        N_im = count_parameters(im_net)
        N_spec = count_parameters(spec_net)
        total_params = N_im + N_spec
        
        print(f"[{args.variant}] Final PSNR: {final_psnr:.4f} dB, SSIM: {final_ssim:.4f}, RMSE: {final_rmse:.6f}")
        print(f"[{args.variant}] Runtime: {runtime:.2f} s, Peak VRAM: {max_vram:.2f} MB")
        
        with open(f'results_hybrid/{args.variant}_full_summary.txt', 'w') as f:
            f.write(f"Variant: {args.variant}\n")
            f.write(f"PSNR: {final_psnr:.4f}\n")
            f.write(f"SSIM: {final_ssim:.4f}\n")
            f.write(f"RMSE: {final_rmse:.6f}\n")
            f.write(f"Runtime (s): {runtime:.2f}\n")
            f.write(f"Peak VRAM (MB): {max_vram:.2f}\n")
            f.write(f"Total Params: {total_params}\n")
            
        np.savetxt(f'results_hybrid/loss_{args.variant}.csv', loss_array, delimiter=',')

if __name__ == '__main__':
    main()
