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

base_ckpt_path = 'results/hybrid/figure09_hyperspectral/baseline_checkpoint.pth'
ckpt = torch.load(base_ckpt_path, map_location='cuda', weights_only=False)

print('Baseline Checkpoint epoch:', ckpt['epoch'])
print('Baseline Checkpoint metrics:', ckpt['metrics'])
print('Baseline Checkpoint config keys:', ckpt['config'].keys() if 'config' in ckpt else 'No config')

model = build_deeptensor_model(
    rank=20,
    seq_len=nwvl,
    spatial_type='skip',
    spectral_type='skip1d',
    n_inputs=20,
).to('cuda')
model.load_state_dict(ckpt['model_state_dict'])

# In run_hybrid_figure09_hsi.py, latents were generated with seed 200:
from experiments.hybrid_prototypes.run_hybrid_figure09_hsi import set_seed
set_seed(200)
im_inp = utils.get_inp([1, 20, H, W]).to('cuda')
spec_inp = utils.get_inp([1, 20, nwvl]).to('cuda')

# But during training, im_inp was optimized in optimizer!
# Did baseline_checkpoint save optimized latents?
if 'optimizer_state_dict' in ckpt:
    print('Baseline checkpoint has optimizer_state_dict, num params:', len(ckpt['optimizer_state_dict']['state']))

