import os
import json
import torch

experiments = [
    ('Figure 09', 'figure09_hyperspectral', 'figure09_icvl_hsi_comparison.json'),
    ('Figure 10', 'figure10_yale_denoising', 'figure10_yale_b_comparison.json'),
    ('Figure 11', 'figure11_video_cs', 'figure11_video_cs_comparison.json'),
    ('Figure 12', 'figure12_pet_ct', 'figure12_pet_ct_comparison.json'),
]

for fig, dir_name, json_name in experiments:
    p = os.path.join('results', 'hybrid', dir_name, json_name)
    with open(p) as f:
        d = json.load(f)
    print(f"\n=======================================================")
    print(f"=== {fig}: {dir_name} ===")
    print(f"=======================================================")
    for m, v in d['models'].items():
        params = v.get('params')
        print(f"  {m:18s}: params = {params:,}" if params is not None else f"  {m:18s}: params = None")
        
    # Also inspect checkpoint parameter counts directly
    res_dir = os.path.join('results', 'hybrid', dir_name)
    for m in ['baseline', 'hybrid', 'cnn_control', 'pointwise_control', 'transformer']:
        ckpt_p = os.path.join(res_dir, f"{m}_checkpoint.pth")
        if os.path.exists(ckpt_p):
            ckpt = torch.load(ckpt_p, map_location='cpu', weights_only=False)
            sd = ckpt['model_state_dict']
            n_p = sum(p.numel() for p in sd.values())
            print(f"  [CHECKPOINT {m:18s}] state_dict params = {n_p:,}")

