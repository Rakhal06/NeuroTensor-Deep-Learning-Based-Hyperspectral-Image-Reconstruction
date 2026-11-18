import os
import json

ROOT = 'e:/Rakhal_Tensor'

orig_fp = os.path.join(ROOT, 'paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/original_baseline/figure13b_metrics.json')
impr_fp = os.path.join(ROOT, 'paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/improved_run_fig13b/figure13b_improved_metrics.json')

print("=== FIGURE 13(B) ORIGINAL BASELINE METRICS ===")
if os.path.exists(orig_fp):
    with open(orig_fp, 'r') as f:
        print(f.read())
else:
    print("Orig metrics not found in original_baseline")

print("\n=== FIGURE 13(B) IMPROVED RUN METRICS ===")
if os.path.exists(impr_fp):
    with open(impr_fp, 'r') as f:
        print(f.read())
else:
    print("Improved metrics not found in improved_run_fig13b")

