import os

ROOT = 'e:/Rakhal_Tensor'

print("Searching codebase for '88.67'...")
for root, dirs, files in os.walk(ROOT):
    if '.venv' in root or '.git' in root: continue
    for f in files:
        if f.endswith(('.md', '.txt', '.json', '.py')):
            fp = os.path.join(root, f)
            try:
                with open(fp, 'r', encoding='utf-8', errors='ignore') as sfp:
                    txt = sfp.read()
                    if '88.67' in txt or '88.6' in txt:
                        print(f"  Match in {os.path.relpath(fp, ROOT)}:")
                        for l in txt.splitlines():
                            if '88.67' in l or '88.6' in l:
                                print(f"    {l.strip()}")
            except Exception:
                pass

