#!/usr/bin/env python
#


'''
    This script replicates the DeepTensor results in Figure 6. The accuracy is
    different from the paper as hyperspectral image was downsampled.
'''

import argparse
import os

import sys
import tqdm
import copy
import time

import numpy as np
from scipy import io
from skimage.metrics import structural_similarity as ssim_func

import torch

import matplotlib.pyplot as plt
plt.gray()
import cv2

sys.path.append('modules')

import models
import utils
import losses
import spectral
import deep_prior
import deep_decoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extension 4: Noise Robustness')
    parser.add_argument('--noise_type', type=str, choices=['mixed', 'gaussian', 'poisson'], required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=5000)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    expname = 'icvl'
    rank = 20
    n_inputs = rank
    init_nconv = 128
    num_channels_up = 5
    nettype = 'dip'
    
    scaling = 1
    tau = 100
    noise_snr = 2
    
    # Learning constants
    learning_rate = 1e-3
    epochs = 5000
    reg_noise_std = 1.0/30.0
    exp_weight = 0.99
    
    # Load data
    data = io.loadmat('data/%s.mat'%expname)
    cube = data['hypercube'].astype(np.float32)
    wavelengths = data['wavelengths'].astype(np.float32).ravel()
    cube = utils.resize(cube/cube.max(), scaling)
    
    
    H, W, nwvl = cube.shape

    if rank == 'max':
        rank = nwvl - 1
        n_inputs = rank
    
    hsmat = cube.reshape(H*W, nwvl)
    
    # --- CALIBRATION BLOCK ---
    # We must match the PSNR of the original mixed baseline
    np.random.seed(args.seed)
    cube_mixed_ref = utils.measure(cube, noise_snr=2, tau=100)
    target_psnr = utils.psnr(cube, cube_mixed_ref)
    
    # Restore seed for generation
    np.random.seed(args.seed)
    
    if args.noise_type == 'mixed':
        tau = 100
        noise_snr = 2
        cube_noisy = utils.measure(cube, noise_snr, tau)
        actual_psnr = utils.psnr(cube, cube_noisy)
        calib_info = "N/A"
    elif args.noise_type == 'gaussian':
        tau = float('inf')
        low, high = 0.0, 100.0
        for _ in range(50):
            mid = (low + high) / 2
            np.random.seed(args.seed)
            cube_g = utils.measure(cube, noise_snr=mid, tau=tau)
            psnr_g = utils.psnr(cube, cube_g)
            if psnr_g > target_psnr:
                low = mid
            else:
                high = mid
        noise_snr = mid
        np.random.seed(args.seed)
        cube_noisy = utils.measure(cube, noise_snr, tau)
        actual_psnr = utils.psnr(cube, cube_noisy)
        calib_info = f"snr={noise_snr:.6f}"
    elif args.noise_type == 'poisson':
        noise_snr = 0
        low, high = 1e-3, 10000.0
        for _ in range(50):
            mid = (low + high) / 2
            np.random.seed(args.seed)
            cube_p = utils.measure(cube, noise_snr=0, tau=mid)
            psnr_p = utils.psnr(cube, cube_p)
            if psnr_p < target_psnr:
                low = mid
            else:
                high = mid
        tau = mid
        np.random.seed(args.seed)
        cube_noisy = utils.measure(cube, noise_snr, tau)
        actual_psnr = utils.psnr(cube, cube_noisy)
        calib_info = f"tau={tau:.6f}"

    print(f"[{args.noise_type.upper()}] Target PSNR: {target_psnr:.4f} | Actual PSNR: {actual_psnr:.4f}")
    
    epochs = args.epochs

    
    hsten = torch.tensor(hsmat)[None, ...].cuda()
    hsten_noisy = torch.tensor(cube_noisy.reshape(H*W, nwvl))[None, ...].cuda()
    
    criterion_l1 = losses.L1Norm()
    
    loss_array = np.zeros(epochs)
    mse_array = np.zeros(epochs)

    # Generate networks
    if nettype == 'unet':
        im_net = models.UNetND(n_inputs, rank, 2, init_nconv).cuda()    
        spec_net = models.UNetND(n_inputs, rank, 1, init_nconv).cuda()
        
        im_inp = utils.get_inp([1, n_inputs, H, W])
        spec_inp = utils.get_inp([1, n_inputs, nwvl])
    elif nettype == 'dip':
        im_net = deep_prior.get_net(n_inputs, 'skip', 'reflection',
                                    upsample_mode='bilinear',
                                    skip_n33d=128,
                                    skip_n33u=128,
                                    num_scales=5,
                                    n_channels=rank).cuda()
        spec_net = deep_prior.get_net(n_inputs, 'skip1d', 'reflection',
                                    upsample_mode='linear',
                                    skip_n33d=128,
                                    skip_n33u=128,
                                    num_scales=5,
                                    n_channels=rank).cuda()
        im_inp = utils.get_inp([1, n_inputs, H, W])
        spec_inp = utils.get_inp([1, n_inputs, nwvl])
    elif nettype == 'dd':
        nchans = [init_nconv]*num_channels_up
        im_net = deep_decoder.decodernw(rank).cuda()
        spec_net = deep_decoder.decodernw1d(rank).cuda()
        
        H1 = H // pow(2, num_channels_up)
        W1 = W // pow(2, num_channels_up)
        nwvl1 = nwvl // pow(2, num_channels_up)
        im_inp = utils.get_inp([1, init_nconv, H1, W1])
        spec_inp = utils.get_inp([1, init_nconv, nwvl1])
    
    # Switch to training mode
    im_net.train()
    spec_net.train()
    
    net_params = list(im_net.parameters()) + list(spec_net.parameters())
    inp_params = [im_inp] + [spec_inp]
    
    im_inp_per = im_inp.detach().clone()
    spec_inp_per = spec_inp.detach().clone()
    
    hs_estim_avg = None
    best_loss = float('inf')
    best_epoch = 0
    
    params = net_params + inp_params
    optimizer = torch.optim.Adam(lr=learning_rate, params=params)

    tic = time.time()
    tbar = tqdm.tqdm(range(epochs))
    for idx in tbar:        
        # Perturb inputs
        im_inp_perturbed = im_inp + im_inp_per.normal_()*reg_noise_std
        spec_inp_perturbed = spec_inp + spec_inp_per.normal_()*reg_noise_std
        
        U_img = im_net(im_inp_perturbed)
        V = spec_net(spec_inp_perturbed)
        
        U = U_img.reshape(-1, rank, H*W).permute(0, 2, 1)
        
        hs_estim = torch.bmm(U, V)
        
        loss = criterion_l1(hsten_noisy - hs_estim)
        loss_array[idx] = loss.item()
        mse_array[idx] = ((hs_estim - hsten)**2).mean().item()
        
        tbar.set_description('%.4e'%mse_array[idx])
        tbar.refresh()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_epoch = idx
            best_hs_estim = copy.deepcopy(hs_estim.detach().squeeze())
        
            # Averaging as per original code    
            if hs_estim_avg is None:
                hs_estim_avg = hs_estim.detach()
            else:
                hs_estim_avg = exp_weight*hs_estim_avg +\
                            (1 - exp_weight)*hs_estim.detach()
        

            

    toc = time.time()
    runtime = toc - tic
    
    cube_estim = best_hs_estim.cpu().numpy().reshape(H, W, nwvl)
    cube_lr = spectral.lr_decompose(cube_noisy, rank)
    
    data_range = cube.max() - cube.min()
    final_psnr = utils.psnr(cube, cube_estim)
    final_ssim = ssim_func(cube, cube_estim, channel_axis=-1, data_range=data_range)
    final_rmse = np.sqrt(((cube - cube_estim)**2).mean())
    
    psnr_gain = final_psnr - actual_psnr
    
    # Calculate noise-intensity correlation
    noise_magnitude = np.abs(cube_noisy - cube)
    correlation = np.corrcoef(cube.flatten(), noise_magnitude.flatten())[0, 1]
    
    vram_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
    
    res_dir = "results_robustness"
    os.makedirs(res_dir, exist_ok=True)
    
    res_file = os.path.join(res_dir, f"summary_{args.noise_type}.txt")
    with open(res_file, 'w') as f:
        f.write(f"Noise Type: {args.noise_type}\n")
        f.write(f"Noise Parameters: tau={tau}, noise_snr={noise_snr} ({calib_info})\n")
        f.write(f"Target Input PSNR: {target_psnr:.4f} dB\n")
        f.write(f"Actual Input PSNR: {actual_psnr:.4f} dB\n")
        f.write(f"Reconstruction PSNR: {final_psnr:.4f} dB\n")
        f.write(f"Reconstruction SSIM: {final_ssim:.6f}\n")
        f.write(f"Reconstruction RMSE: {final_rmse:.6f}\n")
        f.write(f"PSNR Gain: {psnr_gain:.4f} dB\n")
        f.write(f"Runtime: {runtime:.2f} sec\n")
        f.write(f"Peak VRAM: {vram_peak:.2f} MB\n")
        f.write(f"Noise-Intensity Correlation: {correlation:.6f}\n")

    print(f"Results saved to {res_file}")

