import os
import hashlib
import csv
import datetime

ROOT = os.path.abspath('e:/Rakhal_Tensor')
MIGRATION_DIR = os.path.join(ROOT, 'migration')
os.makedirs(MIGRATION_DIR, exist_ok=True)

print("Computing SHA-256 for all files in repository...")
sha256_before_records = []
manifest_records = []

# Exclude directories
EXCLUDE_DIRS = {'.git', '.venv', '__pycache__', '.pytest_cache'}

def get_sha256(filepath):
    h = hashlib.sha256()
    try:
        with open(filepath, 'rb') as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        return f"ERROR: {str(e)}"

total_files = 0
for root, dirs, files in os.walk(ROOT):
    # filter in-place
    dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
    for f in files:
        full_p = os.path.join(root, f)
        rel_p = os.path.relpath(full_p, ROOT).replace('\\', '/')
        if rel_p.startswith('migration/'):
            continue
        total_files += 1
        h = get_sha256(full_p)
        sz = os.path.getsize(full_p)
        sha256_before_records.append({
            'path': rel_p,
            'sha256': h,
            'size_bytes': sz
        })

print(f"Total files hashed before migration: {len(sha256_before_records)}")

# Write migration/SHA256_BEFORE.csv
sha_before_csv = os.path.join(MIGRATION_DIR, 'SHA256_BEFORE.csv')
with open(sha_before_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['path', 'sha256', 'size_bytes'])
    writer.writeheader()
    writer.writerows(sha256_before_records)
print(f"Written: {sha_before_csv}")

# Initialize migration/MIGRATION_LOG.md
log_path = os.path.join(MIGRATION_DIR, 'MIGRATION_LOG.md')
timestamp = datetime.datetime.now().isoformat()
with open(log_path, 'w', encoding='utf-8') as f:
    f.write(f"# DeepTensor++ Migration Execution Log\n\n")
    f.write(f"- **Execution Start Time:** {timestamp}\n")
    f.write(f"- **Workspace Root:** `{ROOT}`\n")
    f.write(f"- **Pre-Migration Git Commit:** `b37e7d8`\n")
    f.write(f"- **Pre-Migration File Count:** {len(sha256_before_records)} files cataloged\n")
    f.write(f"- **Preservation Rules Enforced:** Strictly No deletions, No loss of checkpoints, No overwrite of results\n\n")
    f.write(f"## Chronological Step Log\n\n")
    f.write(f"### Phase 0: Integrity Checkpointing & Pre-Migration Hashing\n")
    f.write(f"- Staged and committed working tree bugfixes to `.gitignore` and `modules/lin_inverse.py` (Commit `b37e7d8`).\n")
    f.write(f"- Generated `{os.path.relpath(sha_before_csv, ROOT)}` cataloging {len(sha256_before_records)} physical files.\n\n")

print(f"Written: {log_path}")

