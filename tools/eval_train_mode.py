import sys, os
sys.path.insert(0, os.path.abspath('.'))
import torch
import numpy as np
from scipy import io
from deeptensor import utils
from deeptensor.models.factory import build_deeptensor_model
from deeptensor.utils.comparison import compute_metrics

data_path = 'data/icvl.mat'
data = io.loadmat(data_path)
cube = data['hypercube'].astype(np.float32)
cube = utils.resize(cube / cube.max(), 1)
H, W, nwvl = cube.shape

ckpt_path = 'results/extensions/adaptive_hybrid_v1_full/figure09_hyperspectral/best_checkpoint.pth'
ckpt = torch.load(ckpt_path, map_location='cuda', weights_only=False)

model = build_deeptensor_model(
    rank=20,
    seq_len=nwvl,
    spatial_type='skip',
    spectral_type='adaptive_hybrid',
    n_inputs=20,
    d_model=128,
    n_heads=4,
    n_layers=4,
    d_ffn=512,
    router_type='cagc',
    pe_type='sin_pe',
    tau=0.5,
).to('cuda')
model.load_state_dict(ckpt['model_state_dict'])

im_inp = ckpt['config']['im_inp'].to('cuda')
spec_inp = ckpt['config']['spec_inp'].to('cuda')

# In train mode (standard DIP inference mode)
model.train()
with torch.no_grad():
    out = model(im_inp, spec_inp)
    recon = out[0].cpu().numpy().reshape(H, W, nwvl)
    m = compute_metrics(cube, recon)
    print("Train mode (Clean latents):")
    print(f"  PSNR: {m['psnr']:.4f} dB, SSIM: {m['ssim']:.4f}, RMSE: {m['rmse']:.6f}, MAE: {m['mae']:.6f}, RelErr: {m['rel_error']:.6f}")

    # Check loss on noisy observation
    cube_noisy = utils.measure(cube, 2, 100)
    hsten_noisy = torch.tensor(cube_noisy.reshape(H * W, nwvl), dtype=torch.float32)[None, ...].to('cuda')
    l1_loss = torch.abs(hsten_noisy - out).mean().item()
    print(f"  Unsupervised L1 Loss: {l1_loss:.6f}")

