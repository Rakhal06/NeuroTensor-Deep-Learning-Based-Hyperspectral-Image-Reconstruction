import os
import json
import hashlib
import ast

ROOT = os.path.abspath("e:/Rakhal_Tensor")

def get_hash(path):
    h = hashlib.sha256()
    try:
        with open(path, 'rb') as fp:
            while chunk := fp.read(8192):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None

def analyze_python_file(rel_path):
    abs_path = os.path.join(ROOT, rel_path)
    size = os.path.getsize(abs_path)
    with open(abs_path, 'r', encoding='utf-8', errors='ignore') as fp:
        content = fp.read()
    
    lines = content.splitlines()
    num_lines = len(lines)
    
    classes = []
    functions = []
    imports = []
    doc = ""
    
    try:
        tree = ast.parse(content)
        doc = ast.get_docstring(tree) or ""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
            elif isinstance(node, ast.FunctionDef):
                functions.append(node.name)
            elif isinstance(node, ast.Import):
                for n in node.names:
                    imports.append(n.name)
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                for n in node.names:
                    imports.append(f"{mod}.{n.name}" if mod else n.name)
    except Exception as e:
        doc = f"Parse Error: {str(e)}"
    
    return {
        "path": rel_path.replace("\\", "/"),
        "size_bytes": size,
        "lines": num_lines,
        "classes": sorted(list(set(classes))),
        "functions": sorted(list(set(functions))),
        "imports": sorted(list(set(imports))),
        "doc": doc.strip(),
        "sha256": get_hash(abs_path)
    }

# 1. Discover all files by extension
all_files = {}
for root, dirs, files in os.walk(ROOT):
    # Skip virtual environment and git
    if '.venv' in root or '.git' in root or '__pycache__' in root or '.gemini' in root:
        continue
    for f in files:
        full_path = os.path.join(root, f)
        rel_path = os.path.relpath(full_path, ROOT).replace("\\", "/")
        ext = os.path.splitext(f)[1].lower()
        if ext not in all_files:
            all_files[ext] = []
        all_files[ext].append({
            "path": rel_path,
            "size": os.path.getsize(full_path),
            "filename": f
        })

py_details = [analyze_python_file(f["path"]) for f in all_files.get('.py', [])]

# Sort by path
py_details.sort(key=lambda x: x["path"])

# Identify duplicates
hash_map = {}
for p in py_details:
    h = p["sha256"]
    if h not in hash_map:
        hash_map[h] = []
    hash_map[h].append(p["path"])

duplicates = {h: paths for h, paths in hash_map.items() if len(paths) > 1}

output_summary = {
    "total_py_files": len(py_details),
    "file_counts_by_ext": {k: len(v) for k, v in all_files.items()},
    "duplicates": duplicates,
    "python_files": py_details,
    "checkpoints": [f for f in all_files.get('.pt', []) + all_files.get('.pth', [])],
    "metrics_json": [f for f in all_files.get('.json', []) if 'metric' in f['path'].lower() or 'result' in f['path'].lower() or 'config' in f['path'].lower()],
    "metrics_csv": [f for f in all_files.get('.csv', [])],
    "plots_png": [f for f in all_files.get('.png', [])],
    "datasets_mat": [f for f in all_files.get('.mat', [])],
    "reports_md": [f for f in all_files.get('.md', [])]
}

with open(os.path.join(ROOT, "scratch", "comprehensive_audit_data.json"), "w", encoding="utf-8") as fp:
    json.dump(output_summary, fp, indent=2)

print(f"Audit data extracted successfully:")
print(f"Total Python files: {len(py_details)}")
print(f"Total duplicate groups: {len(duplicates)}")
print(f"Total .pt checkpoints: {len(output_summary['checkpoints'])}")
print(f"Total .png plots: {len(output_summary['plots_png'])}")
print(f"Total .md reports: {len(output_summary['reports_md'])}")
print(f"Total .mat datasets: {len(output_summary['datasets_mat'])}")

