import os

ROOT = 'e:/Rakhal_Tensor'

def get_dir_size(path):
    total = 0
    for root, dirs, files in os.walk(path):
        if '.venv' in root or '.git' in root:
            continue
        for f in files:
            fp = os.path.join(root, f)
            try:
                total += os.path.getsize(fp)
            except Exception:
                pass
    return total

subdirs = [d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)) and not d.startswith('.') and d != '.venv']

print(f"{'Folder':<40s} | Size (MB)")
print("-" * 55)
total_repo = 0
for sd in sorted(subdirs):
    p = os.path.join(ROOT, sd)
    sz = get_dir_size(p) / (1024 * 1024)
    total_repo += sz
    print(f"{sd:<40s} | {sz:8.2f} MB")

root_files_size = sum(os.path.getsize(os.path.join(ROOT, f)) for f in os.listdir(ROOT) if os.path.isfile(os.path.join(ROOT, f))) / (1024 * 1024)
print(f"{'[Root Directory Files]':<40s} | {root_files_size:8.2f} MB")
total_repo += root_files_size
print("-" * 55)
print(f"{'Total Repository (excl .git & .venv)':<40s} | {total_repo:8.2f} MB ({total_repo/1024:.2f} GB)")

