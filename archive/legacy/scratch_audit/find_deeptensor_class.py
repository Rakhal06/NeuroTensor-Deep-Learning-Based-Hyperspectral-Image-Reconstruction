import os

ROOT = 'e:/Rakhal_Tensor'

for root, dirs, files in os.walk(ROOT):
    if '.venv' in root or '.git' in root or 'src' in root: continue
    for f in files:
        if f.endswith('.py'):
            fp = os.path.join(root, f)
            with open(fp, 'r', errors='ignore') as sfp:
                txt = sfp.read()
                if 'class DeepTensor' in txt:
                    print(f"Found 'class DeepTensor' in: {os.path.relpath(fp, ROOT)}")
                if 'DeepTensor(' in txt:
                    print(f"Found 'DeepTensor(' in: {os.path.relpath(fp, ROOT)}")

