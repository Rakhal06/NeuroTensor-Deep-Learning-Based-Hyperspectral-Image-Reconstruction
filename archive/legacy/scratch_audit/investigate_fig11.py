import os
import json

ROOT = 'e:/Rakhal_Tensor'
fig11_dir = os.path.join(ROOT, 'paper_reproduction/cnn_baseline/07_video_compressive_sensing')

print("=== CHECKING ALL FIGURE 11 METRICS ===")
for root, dirs, files in os.walk(fig11_dir):
    for f in files:
        if f.endswith(('.json', '.txt', '.csv', '.md')):
            p = os.path.join(root, f)
            print(f"\n--- {os.path.relpath(p, ROOT)} ---")
            with open(p, 'r', errors='ignore') as sfp:
                lines = sfp.readlines()
                for l in lines[:30]:
                    print(l.rstrip())

