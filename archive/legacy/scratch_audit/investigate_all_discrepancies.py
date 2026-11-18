import os
import json
import scipy.io as io
import numpy as np

ROOT = 'e:/Rakhal_Tensor'

def check_file(rel_path):
    p = os.path.join(ROOT, rel_path)
    return os.path.exists(p)

print("=" * 60)
print("1. YALE FACE B DIMENSIONS (192x168x64 vs 192x168x160)")
print("=" * 60)
yale_mat = io.loadmat(os.path.join(ROOT, 'data/yale_b.mat'))
faces = yale_mat['faces']
print(f"data/yale_b.mat 'faces' shape: {faces.shape}, dtype: {faces.dtype}")

# Check run_figure10_yale_tensor.py
with open(os.path.join(ROOT, 'paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/run_figure10_yale_tensor.py'), 'r', encoding='utf-8') as f:
    lines = f.readlines()
for i, l in enumerate(lines[:100]):
    if 'loadmat' in l or 'shape' in l or 'faces' in l:
        print(f"  run_figure10 line {i+1}: {l.strip()}")

# Check run_figure10_yale_tensor_improved.py
with open(os.path.join(ROOT, 'paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/run_figure10_yale_tensor_improved.py'), 'r', encoding='utf-8') as f:
    lines = f.readlines()
for i, l in enumerate(lines[:120]):
    if 'loadmat' in l or 'shape' in l or 'faces' in l:
        print(f"  run_figure10_improved line {i+1}: {l.strip()}")

print("\n" + "=" * 60)
print("2. PET DATASET DIMENSIONS (128x128x150 vs 128x128x56)")
print("=" * 60)
pet_mat = io.loadmat(os.path.join(ROOT, 'data/pet.mat'))
hypercube = pet_mat['hypercube']
print(f"data/pet.mat 'hypercube' shape: {hypercube.shape}, dtype: {hypercube.dtype}")

# Check run_figure12.py
with open(os.path.join(ROOT, 'run_figure12.py'), 'r', encoding='utf-8') as f:
    lines = f.readlines()
for i, l in enumerate(lines[:100]):
    if 'loadmat' in l or 'shape' in l or 'pet' in l.lower():
        print(f"  run_figure12 line {i+1}: {l.strip()}")

# Check run_figure12_improved.py
with open(os.path.join(ROOT, 'paper_reproduction/cnn_baseline/08_ct_reconstruction/run_figure12_improved.py'), 'r', encoding='utf-8') as f:
    lines = f.readlines()
for i, l in enumerate(lines[:100]):
    if 'loadmat' in l or 'shape' in l or 'pet' in l.lower():
        print(f"  run_figure12_improved line {i+1}: {l.strip()}")

print("\n" + "=" * 60)
print("3. FIGURE 9 PSNR (43.43 dB vs 46.28 dB)")
print("=" * 60)
# Check metrics.txt and reports
fig9_baseline_dir = os.path.join(ROOT, 'RESULTS-BASELINE/figure9')
if os.path.exists(fig9_baseline_dir):
    for fn in os.listdir(fig9_baseline_dir):
        if fn.endswith('.txt') or fn.endswith('.json'):
            fp = os.path.join(fig9_baseline_dir, fn)
            with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                print(f"  RESULTS-BASELINE/figure9/{fn}:")
                for line in content.splitlines()[:15]:
                    print(f"    {line}")

# Search codebase for 46.28
print("\nSearching codebase for '46.28'...")
for root, dirs, files in os.walk(ROOT):
    if '.venv' in root or '.git' in root: continue
    for f in files:
        if f.endswith(('.md', '.txt', '.json', '.py')):
            fp = os.path.join(root, f)
            try:
                with open(fp, 'r', encoding='utf-8', errors='ignore') as sfp:
                    txt = sfp.read()
                    if '46.28' in txt or '46.2' in txt:
                        rel = os.path.relpath(fp, ROOT)
                        print(f"  [Found 46.28 match]: {rel}")
            except Exception:
                pass

print("\n" + "=" * 60)
print("4. FIGURE 11 ORIGINAL BASELINE METRIC DISCREPANCIES")
print("=" * 60)
# Check all metrics for Figure 11
fig11_paths = [
    'paper_reproduction/cnn_baseline/07_video_compressive_sensing/original_baseline',
    'paper_reproduction/cnn_baseline/07_video_compressive_sensing/improved_run',
    'paper_reproduction/cnn_baseline/07_video_compressive_sensing'
]
for p in fig11_paths:
    abs_p = os.path.join(ROOT, p)
    if os.path.exists(abs_p):
        for fn in os.listdir(abs_p):
            if fn.endswith(('.json', '.txt', '.csv', '.md')):
                fp = os.path.join(abs_p, fn)
                if os.path.isfile(fp):
                    print(f"\nFile: {p}/{fn}:")
                    with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                        for l in lines[:25]:
                            if any(k in l.lower() for k in ['psnr', 'ssim', 'tv', 'deeptensor', 'baseline', 'improved', 'metric']):
                                print(f"    {l.strip()}")

print("\n" + "=" * 60)
print("5. FIGURE 2 ACCURACY (85.71% vs 88.67%)")
print("=" * 60)
fig2_baseline_dir = os.path.join(ROOT, 'RESULTS-BASELINE/figure2')
if os.path.exists(fig2_baseline_dir):
    for fn in os.listdir(fig2_baseline_dir):
        if fn.endswith(('.txt', '.json')):
            fp = os.path.join(fig2_baseline_dir, fn)
            print(f"\nRESULTS-BASELINE/figure2/{fn}:")
            with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
                print(f.read().strip())

# Check run_figure2.py
with open(os.path.join(ROOT, 'run_figure2.py'), 'r', encoding='utf-8') as f:
    txt = f.read()
    if '85.' in txt or '88.' in txt or 'accuracy' in txt.lower():
        print("Matches in run_figure2.py for accuracy:")
        for line in txt.splitlines():
            if 'acc' in line.lower() or 'accuracy' in line.lower() or 'print' in line.lower() and '%' in line:
                print(f"  {line.strip()}")

# Search codebase for 85.71 and 88.67
print("\nSearching codebase for '85.7' and '88.6'...")
for root, dirs, files in os.walk(ROOT):
    if '.venv' in root or '.git' in root: continue
    for f in files:
        if f.endswith(('.md', '.txt', '.json', '.py')):
            fp = os.path.join(root, f)
            try:
                with open(fp, 'r', encoding='utf-8', errors='ignore') as sfp:
                    t = sfp.read()
                    if '85.7' in t or '85.71' in t:
                        print(f"  [Found 85.71]: {os.path.relpath(fp, ROOT)}")
                    if '88.6' in t or '88.67' in t:
                        print(f"  [Found 88.67]: {os.path.relpath(fp, ROOT)}")
            except Exception:
                pass

print("\n" + "=" * 60)
print("6. FIGURE 13(B) ORIGINAL & IMPROVED METRIC PROVENANCE")
print("=" * 60)
yale_sweep_dir = os.path.join(ROOT, 'paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising')
for sub in ['original_baseline', 'improved_run_fig13b', '.']:
    sd = os.path.normpath(os.path.join(yale_sweep_dir, sub))
    if os.path.exists(sd):
        for fn in os.listdir(sd):
            if '13b' in fn.lower() or 'sweep' in fn.lower():
                fp = os.path.join(sd, fn)
                if os.path.isfile(fp):
                    print(f"\nFile: {os.path.relpath(fp, ROOT)}:")
                    with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
                        for l in f.readlines()[:20]:
                            print(f"  {l.strip()}")

