import os
import json
import scipy.io as io
import numpy as np

ROOT = 'e:/Rakhal_Tensor'

print("=== DISCREPANCY 1: YALE FACE B DIMENSIONS ===")
yale_path = os.path.join(ROOT, 'data', 'yale_b.mat')
if os.path.exists(yale_path):
    m = io.loadmat(yale_path)
    print("Keys in data/yale_b.mat:", [k for k in m.keys() if not k.startswith('__')])
    for k in m.keys():
        if not k.startswith('__'):
            arr = m[k]
            print(f"  Key '{k}': shape {arr.shape}, dtype {arr.dtype}")
else:
    print("data/yale_b.mat NOT FOUND")

# Check CroppedYale
cropped_yale_dir = os.path.join(ROOT, 'data', 'CroppedYale')
if os.path.exists(cropped_yale_dir):
    subdirs = [d for d in os.listdir(cropped_yale_dir) if os.path.isdir(os.path.join(cropped_yale_dir, d))]
    print(f"data/CroppedYale exists: {len(subdirs)} subject directories")
    if subdirs:
        s0 = os.path.join(cropped_yale_dir, subdirs[0])
        pgms = [f for f in os.listdir(s0) if f.endswith('.pgm')]
        print(f"  Example subject {subdirs[0]}: {len(pgms)} PGM images")

# Check scripts mentioning 160 or 64
for root, dirs, files in os.walk(ROOT):
    if '.venv' in root or '.git' in root: continue
    for f in files:
        if f.endswith('.py') or f.endswith('.json') or f.endswith('.yaml') or f.endswith('.md'):
            fp = os.path.join(root, f)
            try:
                with open(fp, 'r', encoding='utf-8', errors='ignore') as sfp:
                    txt = sfp.read()
                    if '160' in txt and ('yale' in txt.lower() or '192' in txt):
                        rel = os.path.relpath(fp, ROOT)
                        print(f"  [Found 160 mention] {rel}")
            except Exception:
                pass

print("\n=== DISCREPANCY 2: PET DATASET DIMENSIONS ===")
pet_path = os.path.join(ROOT, 'data', 'pet.mat')
if os.path.exists(pet_path):
    m = io.loadmat(pet_path)
    print("Keys in data/pet.mat:", [k for k in m.keys() if not k.startswith('__')])
    for k in m.keys():
        if not k.startswith('__'):
            arr = m[k]
            print(f"  Key '{k}': shape {arr.shape}, dtype {arr.dtype}")
else:
    print("data/pet.mat NOT FOUND")

# Check scripts mentioning 150 or 56
for root, dirs, files in os.walk(ROOT):
    if '.venv' in root or '.git' in root: continue
    for f in files:
        if f.endswith('.py') or f.endswith('.json') or f.endswith('.yaml') or f.endswith('.md'):
            fp = os.path.join(root, f)
            try:
                with open(fp, 'r', encoding='utf-8', errors='ignore') as sfp:
                    txt = sfp.read()
                    if '56' in txt and ('pet' in txt.lower() or '128' in txt):
                        rel = os.path.relpath(fp, ROOT)
                        print(f"  [Found 56 mention] {rel}")
            except Exception:
                pass

