import os

ROOT = 'e:/Rakhal_Tensor'

cons_dir = os.path.join(ROOT, 'paper_reproduction/consolidated_results')
for fn in os.listdir(cons_dir):
    if fn.endswith(('.md', '.json', '.py')):
        fp = os.path.join(cons_dir, fn)
        with open(fp, 'r', errors='ignore') as f:
            lines = f.readlines()
            for l in lines:
                if 'figure 11' in l.lower() or 'video' in l.lower():
                    if any(c in l for c in ['21.', '25.', '23.', '27.']):
                        print(f"{fn}: {l.strip()}")

