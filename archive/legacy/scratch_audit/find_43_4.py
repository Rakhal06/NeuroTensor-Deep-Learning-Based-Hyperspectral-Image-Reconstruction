import os

ROOT = 'e:/Rakhal_Tensor'

print("Searching codebase for '43.4'...")
for root, dirs, files in os.walk(ROOT):
    if '.venv' in root or '.git' in root: continue
    for f in files:
        if f.endswith(('.md', '.txt', '.json', '.py')):
            fp = os.path.join(root, f)
            try:
                with open(fp, 'r', encoding='utf-8', errors='ignore') as sfp:
                    txt = sfp.read()
                    if '43.4' in txt:
                        rel = os.path.relpath(fp, ROOT)
                        print(f"  Match in {rel}:")
                        for l in txt.splitlines():
                            if '43.4' in l or '43.40' in l or '43.43' in l:
                                print(f"    {l.strip()}")
            except Exception:
                pass

