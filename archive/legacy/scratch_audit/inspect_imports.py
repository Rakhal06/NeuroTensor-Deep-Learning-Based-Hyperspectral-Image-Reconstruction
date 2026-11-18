import json
from collections import defaultdict

with open('scratch/comprehensive_audit_data.json', encoding='utf-8') as f:
    data = json.load(f)

internal_imports = defaultdict(list)
external_imports = defaultdict(list)

standard_libs = {'os', 'sys', 'time', 'json', 'math', 'glob', 'shutil', 'argparse', 'hashlib', 'ast', 'datetime', 'collections', 'copy', 're', 'warnings', 'random', 'io', 'gc'}

for pf in data['python_files']:
    p = pf['path']
    for imp in pf['imports']:
        root_pkg = imp.split('.')[0]
        if root_pkg in standard_libs:
            continue
        if root_pkg in ['modules', 'models', 'utils', 'lin_inverse', 'losses', 'deep_decoder', 'deep_prior', 'attention_spectral', 'spectral_transformer', 'cosine_annealing_with_warmup', 'deep_models']:
            internal_imports[p].append(imp)
        else:
            external_imports[root_pkg].append(p)

print("Unique external libraries imported across the project:")
for pkg, files in sorted(external_imports.items()):
    print(f"  {pkg:<20s} imported by {len(set(files)):2d} files")

print("\nInternal module usages:")
module_usage = defaultdict(list)
for file_path, imps in internal_imports.items():
    for imp in imps:
        module_usage[imp].append(file_path)

for imp, files in sorted(module_usage.items()):
    print(f"  {imp:<35s} imported by {len(files):2d} files")

