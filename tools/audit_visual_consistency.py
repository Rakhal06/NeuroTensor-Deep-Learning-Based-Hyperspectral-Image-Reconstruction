import os
import json
import numpy as np
import torch
from scipy import io

from deeptensor.utils.metrics import calculate_psnr, calculate_ssim
from deeptensor.utils.comparison import to_2d_slice

def inspect_all():
    experiments = [
        ('Figure 09', 'figure09_hyperspectral', 'figure09_icvl_hsi_comparison.json', 'data/icvl.mat', 'hypercube', 130), # nwvl=260 or 31?
        ('Figure 10', 'figure10_yale_denoising', 'figure10_yale_b_comparison.json', 'data/yale_b.mat', 'faces', 80),
        ('Figure 11', 'figure11_video_cs', 'figure11_video_cs_comparison.json', 'data/cat_video.mat', 'hypercube', 32),
        ('Figure 12', 'figure12_pet_ct', 'figure12_pet_ct_comparison.json', 'data/pet.mat', 'hypercube', 28),
    ]

    for fig, dir_name, json_name, mat_path, mat_key, exp_slice_idx in experiments:
        print(f"\n=======================================================")
        print(f"=== {fig}: {dir_name} ===")
        print(f"=======================================================")
        
        # Load data
        mat = io.loadmat(mat_path)
        cube = mat[mat_key].astype(np.float32)
        if fig == 'Figure 10':
            cube = cube / 255.0
        else:
            cube = cube / cube.max()
        
        full_ptp = float(np.ptp(cube))
        full_min, full_max = float(cube.min()), float(cube.max())
        print(f"Ground Truth Shape: {cube.shape}")
        print(f"GT Full Volume Min: {full_min:.4f}, Max: {full_max:.4f}, PTP (Range): {full_ptp:.4f}")
        
        # Slice info
        # Let's find slice_idx
        if fig in ('Figure 09', 'Figure 10', 'Figure 11', 'Figure 12'):
            slice_idx = cube.shape[-1] // 2 if cube.shape[-1] < min(cube.shape[0], cube.shape[1]) or cube.ndim==3 else 0
            if fig == 'Figure 09':
                slice_idx = cube.shape[2] // 2
            elif fig == 'Figure 10':
                slice_idx = cube.shape[2] // 2
            elif fig == 'Figure 11':
                slice_idx = cube.shape[2] // 2
            elif fig == 'Figure 12':
                slice_idx = cube.shape[2] // 2
        
        gt_slice = to_2d_slice(cube, slice_idx)
        slice_ptp = float(np.ptp(gt_slice))
        slice_min, slice_max = float(gt_slice.min()), float(gt_slice.max())
        print(f"Slice Index: {slice_idx} (shape: {gt_slice.shape})")
        print(f"Slice Min: {slice_min:.4f}, Max: {slice_max:.4f}, Slice PTP: {slice_ptp:.4f}")
        print(f"Ratio 20*log10(full_ptp / slice_ptp): {20 * np.log10(full_ptp / slice_ptp):.2f} dB")
        print(f"Ratio 20*log10(1.0 / slice_ptp):      {20 * np.log10(1.0 / slice_ptp):.2f} dB")
        
        # Load JSON summary
        res_dir = os.path.join('results', 'hybrid', dir_name)
        json_path = os.path.join(res_dir, json_name)
        with open(json_path) as f:
            summary = json.load(f)
            
        print("\n--- Model Metrics Comparison ---")
        models = ['baseline', 'hybrid', 'cnn_control', 'pointwise_control', 'transformer']
        for m in models:
            if m not in summary['models']:
                continue
            rep_m = summary['models'][m]['metrics']
            
            # Load checkpoint to get recon
            ckpt_path = os.path.join(res_dir, f"{m}_checkpoint.pth")
            if not os.path.exists(ckpt_path):
                print(f"  {m:18s}: Checkpoint missing")
                continue
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            
            # Does checkpoint store best_recon?
            if 'best_recon' in ckpt:
                recon = ckpt['best_recon']
            else:
                # Reconstruct from model and saved inputs
                if fig == 'Figure 10':
                    from deeptensor.models.factory import build_deeptensor_cp_model
                    type_map = {
                        'baseline': 'skip1d',
                        'hybrid': 'hybrid_attention',
                        'cnn_control': 'cnn_control',
                        'pointwise_control': 'hybrid_control',
                        'transformer': 'spectral_transformer',
                    }
                    model = build_deeptensor_cp_model(
                        rank=2000, lengths=cube.shape, u_type='skip1d', v_type='skip1d', w_type=type_map[m],
                        n_inputs=2000, learnable_core=True, normalize_core=True, d_model=128, n_heads=4, n_layers=4, d_ffn=512
                    ).to('cuda' if torch.cuda.is_available() else 'cpu')
                    model.load_state_dict(ckpt['model_state_dict'])
                    model.train()
                    cfg = ckpt.get('config', {})
                    dev = next(model.parameters()).device
                    with torch.no_grad():
                        recon = model(cfg['u_inp'].to(dev), cfg['v_inp'].to(dev), cfg['w_inp'].to(dev)).detach().cpu().numpy()
                elif fig == 'Figure 09':
                    from deeptensor.models.factory import build_deeptensor_model
                    variants_9 = {
                        'baseline': 'skip1d',
                        'hybrid': 'hybrid_attention',
                        'cnn_control': 'cnn_control',
                        'pointwise_control': 'hybrid_control',
                        'transformer': 'spectral_transformer',
                    }
                    H, W, nwvl = cube.shape
                    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
                    model = build_deeptensor_model(
                        rank=20, seq_len=nwvl, spatial_type='skip', spectral_type=variants_9[m],
                        n_inputs=20, d_model=128, n_heads=4, n_layers=4, d_ffn=512
                    ).to(dev)
                    model.load_state_dict(ckpt['model_state_dict'])
                    model.eval()
                    from deeptensor import utils
                    torch.manual_seed(200)
                    np.random.seed(200)
                    im_inp = utils.get_inp([1, 20, H, W]).to(dev)
                    spec_inp = utils.get_inp([1, 20, nwvl]).to(dev)
                    with torch.no_grad():
                        recon = model(im_inp, spec_inp).squeeze(0).cpu().numpy().reshape(H, W, nwvl)
                else:
                    recon = None
                
            print(f"  {m:18s}: Saved Report PSNR = {rep_m['psnr']:.4f} dB | SSIM = {rep_m['ssim']:.4f} | MAE = {rep_m['mae']:.6f}")
            if recon is not None:
                # Full volume metrics with range 1.0 vs full_ptp
                psnr_vol_1 = calculate_psnr(cube, recon, data_range=1.0)
                psnr_vol_ptp = calculate_psnr(cube, recon, data_range=full_ptp)
                mae_vol = float(np.mean(np.abs(cube - recon)))
                
                # Slice metrics with local slice range vs 1.0 vs full_ptp
                recon_slice = to_2d_slice(recon, slice_idx)
                psnr_sl_local = calculate_psnr(gt_slice, recon_slice, data_range=slice_ptp)
                psnr_sl_1 = calculate_psnr(gt_slice, recon_slice, data_range=1.0)
                psnr_sl_ptp = calculate_psnr(gt_slice, recon_slice, data_range=full_ptp)
                mae_sl = float(np.mean(np.abs(gt_slice - recon_slice)))
                
                print(f"    [Full Volume] PSNR(range=1.0): {psnr_vol_1:.4f} dB | PSNR(range=ptp): {psnr_vol_ptp:.4f} dB | Vol MAE: {mae_vol:.6f}")
                print(f"    [Single Slice] Local Range ({slice_ptp:.4f}): {psnr_sl_local:.4f} dB | Range 1.0: {psnr_sl_1:.4f} dB | Range ptp: {psnr_sl_ptp:.4f} dB | Slice MAE: {mae_sl:.6f}")
                print(f"    Discrepancy (Vol PSNR - Slice Local PSNR): {psnr_vol_1 - psnr_sl_local:+.4f} dB")
                print(f"    Discrepancy (Vol MAE - Slice MAE): {mae_vol - mae_sl:+.6f}")

if __name__ == '__main__':
    inspect_all()
