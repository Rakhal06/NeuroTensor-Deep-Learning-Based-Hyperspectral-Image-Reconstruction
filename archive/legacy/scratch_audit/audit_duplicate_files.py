import os
import hashlib
import json
import subprocess

ROOT = 'e:/Rakhal_Tensor'

def get_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def get_git_info(rel_path):
    try:
        res = subprocess.run(['git', 'log', '-n', '1', '--pretty=format:%h - %an, %ar : %s', '--', rel_path],
                             cwd=ROOT, capture_output=True, text=True, check=False)
        return res.stdout.strip()
    except Exception as e:
        return str(e)

with open(os.path.join(ROOT, 'scratch/comprehensive_audit_data.json'), encoding='utf-8') as f:
    data = json.load(f)

py_files = data['python_files']
duplicates = data['duplicates']

print("=== AUDIT OF SHA-256 IDENTICAL DUPLICATE GROUPS ===")
for h, paths in duplicates.items():
    print(f"\nHash: {h[:16]}... ({len(paths)} files)")
    for p in paths:
        git_info = get_git_info(p)
        print(f"  {p:<70s} | Git: {git_info}")

# Also check candidate modified copies or backups
candidates = [
    ('demo.py', 'demo_backup.py'),
    ('demo_backend.py', 'demo_backend_backup.py'),
    ('run_figure11.py', 'paper_reproduction/cnn_baseline/07_video_compressive_sensing/run_figure11.py'),
    ('run_figure12.py', 'paper_reproduction/cnn_baseline/08_ct_reconstruction/run_figure12_improved.py'),
    ('run_figure10_yale_tensor.py', 'paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/run_figure10_yale_tensor.py'),
    ('run_figure9_bm3d.py', 'paper_reproduction/cnn_baseline/05_hyperspectral_denoising/run_figure9.py'),
    ('generate_pdf.py', 'paper_reproduction/consolidated_results/generate_consolidated_report.py')
]

print("\n=== AUDIT OF CANDIDATE MODIFIED COPIES & BACKUPS ===")
for p1, p2 in candidates:
    fp1 = os.path.join(ROOT, p1)
    fp2 = os.path.join(ROOT, p2)
    e1 = os.path.exists(fp1)
    e2 = os.path.exists(fp2)
    print(f"\nComparing '{p1}' vs '{p2}':")
    if e1 and e2:
        h1 = get_sha256(fp1)
        h2 = get_sha256(fp2)
        sz1 = os.path.getsize(fp1)
        sz2 = os.path.getsize(fp2)
        if h1 == h2:
            print(f"  EXACT DUPLICATE (Hash match: {h1[:16]})")
        else:
            print(f"  MODIFIED COPY: {sz1} bytes vs {sz2} bytes")
            # Count line diffs
            with open(fp1, 'r', errors='ignore') as f1, open(fp2, 'r', errors='ignore') as f2:
                lines1 = f1.readlines()
                lines2 = f2.readlines()
            print(f"  Line counts: {len(lines1)} lines vs {len(lines2)} lines")
    else:
        print(f"  Exists: p1={e1}, p2={e2}")

