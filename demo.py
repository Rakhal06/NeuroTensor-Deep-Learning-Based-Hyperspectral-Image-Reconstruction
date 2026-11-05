import sys
import argparse
import time
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from demo_backend import DemoBackend

def print_header(title):
    print("=" * 50)
    print(f"{title:^50}")
    print("=" * 50)

def print_section(title):
    print("-" * 50)
    print(f"{title}")
    print("-" * 50)

def display_preprocessing(backend):
    print_section("INPUT")
    print(f"Dataset: {backend.expname}")
    print(f"Original shape: ({backend.H}, {backend.W}, {backend.nwvl})")
    print(f"dtype: {backend.cube.dtype}")
    print(f"Device: {backend.device}")
    print_section("PREPROCESSING")
    print(f"Normalization: Div by max, scaling = {backend.scaling}")
    print(f"Noise SNR: {backend.noise_snr} dB")
    print(f"Unfolded shape: ({backend.H * backend.W}, {backend.nwvl})")

def display_network(rank, result=None, device="cuda"):
    print_section("NETWORK")
    print("Spatial network: Deep Image Prior (Skip), 5 scales, 128 channels")
    print("Spectral network: Deep Image Prior (Skip1D), 5 scales, 128 channels")
    print(f"Rank capacity: {rank}")
    print(f"Device: {device}")

def display_factors(res):
    print_section("FACTORS")
    U = res['U']
    V = res['V']
    print(f"U shape: [spatial_dimension x rank] = [{U.shape[1]} x {U.shape[2]}]")
    print(f"V shape: [rank x spectral_dimension] = [{V.shape[1]} x {V.shape[2]}]")

def display_reconstruction(res):
    print_section("RECONSTRUCTION")
    hs_estim = res['hs_estim']
    print("U x V")
    print(f"Shape: [{hs_estim.shape[1]} x {hs_estim.shape[2]}]")
    
def display_optimization(res):
    print_section("OPTIMIZATION")
    print(f"Optimizer: {res['optimizer_info']}")
    print(f"Final loss: {res.get('loss', 0.0):.6e}")

def display_metrics(res):
    print_section("METRICS")
    metrics = res['metrics']
    print(f"PSNR: {metrics['PSNR']:.2f} dB")
    print(f"SSIM: {metrics['SSIM']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.6f}")

def save_visuals(backend, res, method_name):
    method_dir = method_name.lower().replace(' ', '_')
    out_dir = f"results/{method_dir}"
    os.makedirs(out_dir, exist_ok=True)
    
    cube = backend.cube
    cube_noisy = backend.cube_noisy
    cube_estim = res['reconstruction']
    
    wvl_idx = backend.nwvl // 2
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(cube[..., wvl_idx], cmap='gray')
    axes[0].set_title('REFERENCE')
    axes[0].axis('off')
    
    axes[1].imshow(cube_noisy[..., wvl_idx], cmap='gray')
    axes[1].set_title('INPUT / NOISY')
    axes[1].axis('off')
    
    axes[2].imshow(cube_estim[..., wvl_idx], cmap='gray')
    axes[2].set_title(f'{method_name.upper()} RECONSTRUCTION')
    axes[2].axis('off')
    
    plt.tight_layout()
    fig_path = os.path.join(out_dir, 'reconstruction_comparison.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()
    
    # Error Map
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(cube[..., wvl_idx], cmap='gray')
    axes[0].set_title('REFERENCE')
    axes[0].axis('off')
    
    axes[1].imshow(cube_estim[..., wvl_idx], cmap='gray')
    axes[1].set_title('RECONSTRUCTION')
    axes[1].axis('off')
    
    err = np.abs(cube[..., wvl_idx] - cube_estim[..., wvl_idx])
    im = axes[2].imshow(err, cmap='hot')
    axes[2].set_title('ERROR MAP')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    
    err_path = os.path.join(out_dir, 'error_map.png')
    plt.savefig(err_path, dpi=150)
    plt.close()
    print(f"Saved visual evidence to {out_dir}/")
    
    # Save metrics
    import json, csv
    metrics = res['metrics']
    
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    with open(os.path.join(out_dir, 'metrics.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        for k, v in metrics.items():
            writer.writerow([k, v])
            
    # Save network summary
    with open(os.path.join(out_dir, 'network_summary.txt'), 'w') as f:
        f.write("Spatial network parameters: ~1.2M\n")
        f.write("Spectral network parameters: ~1.2M\n")
        f.write("Input dims: [1, n_inputs, H, W] and [1, n_inputs, nwvl]\n")
        f.write("Output dims: [1, R_MAX, H, W] and [1, R_MAX, nwvl]\n")

def run_method(backend, mode, method_id):
    if method_id == 1:
        method_name = "Baseline"
        print_header("DEEPTENSOR IMPLEMENTATION DEMO")
        print(f"METHOD:\n{method_name}")
        display_preprocessing(backend)
        res = backend.run_baseline(mode=mode)
        display_network(res['rank'], device=backend.device)
        display_factors(res)
        display_reconstruction(res)
        display_optimization(res)
        display_metrics(res)
        save_visuals(backend, res, method_name)
        return res
        
    elif method_id == 2:
        method_name = "Adaptive Rank Selection"
        print_header("DEEPTENSOR IMPLEMENTATION DEMO")
        print(f"METHOD:\n{method_name}")
        display_preprocessing(backend)
        res = backend.run_adaptive(mode=mode)
        display_network("30 (R_MAX)", device=backend.device)
        display_factors(res)
        display_reconstruction(res)
        display_optimization(res)
        
        print_section("ADAPTIVE RANK")
        rh = res['rank_history']
        print(f"Initial rank: 4")
        print(f"Selected best rank (used for demo): {res['rank']}")
        if rh:
            print(f"Highest active rank reached: {rh[-1][2]}")
        print(f"Number of rank changes: {len(rh)}")
        for e, old_r, new_r in rh:
            print(f"  - Epoch {e}: {old_r} -> {new_r}")
            
        display_metrics(res)
        save_visuals(backend, res, method_name)
        
        # Save rank history plot
        if rh:
            epochs = [0] + [x[0] for x in rh]
            ranks = [4] + [x[2] for x in rh]
            plt.figure(figsize=(6, 4))
            plt.step(epochs, ranks, where='post')
            plt.title('Active Rank Evolution')
            plt.xlabel('Epoch')
            plt.ylabel('Active Rank')
            plt.grid(True)
            plt.savefig(f"results/{method_name.lower().replace(' ', '_')}/rank_history.png")
            plt.close()
            
        return res
        
    elif method_id == 3:
        method_name = "Factor Group-Sparsity"
        print_header("DEEPTENSOR IMPLEMENTATION DEMO")
        print(f"METHOD:\n{method_name}")
        display_preprocessing(backend)
        res = backend.run_sparsity(mode=mode)
        display_network("30 (R_MAX)", device=backend.device)
        display_factors(res)
        display_reconstruction(res)
        display_optimization(res)
        
        print_section("GROUP-SPARSITY")
        print(f"Rmax: 30")
        print(f"lambda: {res['lambda_val']}")
        print(f"Primary Effective Rank: {res['rank']}")
        g_r = res['g_r']
        # Show top 5 components
        top_idx = np.argsort(g_r)[::-1][:5]
        print("Top 5 Components (by g_r):")
        for i in top_idx:
            if g_r[i] > 1e-6:
                print(f"  Component {i+1:2d}: norm = {g_r[i]:.4e}")
                
        display_metrics(res)
        save_visuals(backend, res, method_name)
        return res

def run_comparison(backend, mode):
    print_header("THREE-WAY COMPARISON")
    print("Running Baseline...")
    res_b = backend.run_baseline(mode=mode)
    print("Running Adaptive Rank...")
    res_a = backend.run_adaptive(mode=mode)
    print("Running Factor Group-Sparsity...")
    res_s = backend.run_sparsity(mode=mode)
    
    print("\n-------------------------------------------------------------------------")
    print(f"{'METHOD':<15} {'PSNR':<8} {'SSIM':<8} {'RMSE':<8} {'RANK':<8} {'EFF.RANK':<8}")
    print("-------------------------------------------------------------------------")
    
    mb = res_b['metrics']
    print(f"{'Baseline':<15} {mb['PSNR']:<8.2f} {mb['SSIM']:<8.4f} {mb['RMSE']:<8.4f} {res_b['rank']:<8} {'-':<8}")
    
    ma = res_a['metrics']
    print(f"{'Adaptive':<15} {ma['PSNR']:<8.2f} {ma['SSIM']:<8.4f} {ma['RMSE']:<8.4f} {res_a['rank']:<8} {'-':<8}")
    
    ms = res_s['metrics']
    print(f"{'Group Sparse':<15} {ms['PSNR']:<8.2f} {ms['SSIM']:<8.4f} {ms['RMSE']:<8.4f} {'30(Max)':<8} {res_s['rank']:<8}")
    print("-------------------------------------------------------------------------")
    
    out_dir = "results/comparison"
    os.makedirs(out_dir, exist_ok=True)
    
    cube = backend.cube
    cube_noisy = backend.cube_noisy
    wvl_idx = backend.nwvl // 2
    
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    axes[0].imshow(cube[..., wvl_idx], cmap='gray'); axes[0].set_title('REFERENCE'); axes[0].axis('off')
    axes[1].imshow(cube_noisy[..., wvl_idx], cmap='gray'); axes[1].set_title('NOISY INPUT'); axes[1].axis('off')
    axes[2].imshow(res_b['reconstruction'][..., wvl_idx], cmap='gray'); axes[2].set_title('BASELINE'); axes[2].axis('off')
    axes[3].imshow(res_a['reconstruction'][..., wvl_idx], cmap='gray'); axes[3].set_title('ADAPTIVE'); axes[3].axis('off')
    axes[4].imshow(res_s['reconstruction'][..., wvl_idx], cmap='gray'); axes[4].set_title('GROUP-SPARSE'); axes[4].axis('off')
    
    plt.tight_layout()
    fig_path = os.path.join(out_dir, 'all_methods_comparison.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\nSaved visual comparison to {out_dir}/")

def main_menu():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default=None, choices=['demo', 'full', 'compare'])
    parser.add_argument('--method', type=str, default=None, choices=['baseline', 'adaptive', 'sparsity'])
    args = parser.parse_args()
    
    if args.mode == 'full' and args.method:
        backend = DemoBackend(expname='icvl')
        method_map = {'baseline': 1, 'adaptive': 2, 'sparsity': 3}
        run_method(backend, 'full', method_map[args.method])
        return
    elif args.mode == 'compare':
        backend = DemoBackend(expname='icvl')
        run_comparison(backend, 'demo')
        return
    elif args.mode == 'demo' and args.method:
        backend = DemoBackend(expname='icvl')
        method_map = {'baseline': 1, 'adaptive': 2, 'sparsity': 3}
        run_method(backend, 'demo', method_map[args.method])
        return
        
    while True:
        print("\n")
        print("==================================================")
        print("        DEEPTENSOR: SELECT INPUT IMAGE")
        print("==================================================")
        print("Available datasets (trained checkpoints):")
        print("[1] ICVL Hyperspectral Image (icvl.mat)")
        print("[Q] Exit")
        print("==================================================")
        choice_img = input("Select an input sample: ").strip().lower()
        
        if choice_img == 'q':
            break
        elif choice_img == '1':
            expname = 'icvl'
        else:
            print("Invalid choice. Please select 1 or Q.")
            continue
            
        backend = DemoBackend(expname=expname)
            
        while True:
            print("\n")
            print("==================================================")
            print("        DEEPTENSOR: SELECT METHOD")
            print("==================================================")
            print("[1] DeepTensor Baseline")
            print("[2] Adaptive Rank Selection")
            print("[3] Factor Group-Sparsity")
            print("[4] Compare All Three")
            print("[B] Back to Input Selection")
            print("==================================================")
            choice = input("Select an option: ").strip().lower()
            
            if choice == '1':
                run_method(backend, 'demo', 1)
            elif choice == '2':
                run_method(backend, 'demo', 2)
            elif choice == '3':
                run_method(backend, 'demo', 3)
            elif choice == '4':
                run_comparison(backend, 'demo')
            elif choice == 'b':
                break
            else:
                print("Invalid choice. Try again.")

if __name__ == "__main__":

    main_menu()
