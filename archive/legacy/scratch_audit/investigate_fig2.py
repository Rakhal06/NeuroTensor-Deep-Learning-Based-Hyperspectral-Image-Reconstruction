import os

ROOT = 'e:/Rakhal_Tensor'

print("=== CHECKING FIGURE 2 ACCURACY ===")
fig2_dir = os.path.join(ROOT, 'RESULTS-BASELINE/figure2')
if os.path.exists(fig2_dir):
    for fn in os.listdir(fig2_dir):
        fp = os.path.join(fig2_dir, fn)
        if os.path.isfile(fp):
            print(f"\n--- RESULTS-BASELINE/figure2/{fn} ---")
            with open(fp, 'r', errors='ignore') as f:
                print(f.read())

# Check final_report.txt
fr = os.path.join(ROOT, 'RESULTS-BASELINE/final_report.txt')
if os.path.exists(fr):
    with open(fr, 'r', errors='ignore') as f:
        for l in f:
            if 'figure 2' in l.lower() or 'eigenface' in l.lower() or 'accuracy' in l.lower():
                print(f"final_report: {l.strip()}")

