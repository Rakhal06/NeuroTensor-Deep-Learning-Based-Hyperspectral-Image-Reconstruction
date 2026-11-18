import os
import json

ROOT = 'e:/Rakhal_Tensor'

print("=== CHECKING FIGURE 9 METRICS ===")
# 1. Check RESULTS-BASELINE/figure9/metrics.txt
with open(os.path.join(ROOT, 'RESULTS-BASELINE/figure9/metrics.txt'), 'r') as f:
    print("RESULTS-BASELINE/figure9/metrics.txt:")
    print(f.read())

# 2. Check paper_reproduction/cnn_baseline/05_hyperspectral_denoising/metrics.json
fp = os.path.join(ROOT, 'paper_reproduction/cnn_baseline/05_hyperspectral_denoising/metrics.json')
if os.path.exists(fp):
    with open(fp, 'r') as f:
        print("\npaper_reproduction/cnn_baseline/05_hyperspectral_denoising/metrics.json:")
        print(f.read())

# 3. Check previous_neurotensor_experiments/01_icvl_cnn_baseline
for root, dirs, files in os.walk(os.path.join(ROOT, 'previous_neurotensor_experiments')):
    for f in files:
        if f.endswith('.json') or f.endswith('.txt'):
            p = os.path.join(root, f)
            with open(p, 'r', errors='ignore') as sfp:
                txt = sfp.read()
                if '46.28' in txt or '43.43' in txt or '43.4' in txt:
                    print(f"\nFound in {os.path.relpath(p, ROOT)}:")
                    for line in txt.splitlines():
                        if any(k in line for k in ['46.28', '43.43', '43.40', 'psnr', 'PSNR']):
                            print(f"  {line.strip()}")

# 4. Check results/baseline
for root, dirs, files in os.walk(os.path.join(ROOT, 'results')):
    for f in files:
        if f.endswith('.json') or f.endswith('.txt'):
            p = os.path.join(root, f)
            with open(p, 'r', errors='ignore') as sfp:
                txt = sfp.read()
                if '46.28' in txt or '43.43' in txt:
                    print(f"\nFound in {os.path.relpath(p, ROOT)}:")
                    for line in txt.splitlines():
                        if any(k in line for k in ['46.28', '43.43', 'psnr', 'PSNR']):
                            print(f"  {line.strip()}")

