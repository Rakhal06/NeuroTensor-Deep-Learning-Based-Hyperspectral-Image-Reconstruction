import json

with open('scratch/comprehensive_audit_data.json', encoding='utf-8') as f:
    data = json.load(f)

modules_files = [p for p in data['python_files'] if p['path'].startswith('modules/')]
print(f"Modules files ({len(modules_files)}):\n")
for mf in modules_files:
    print(f"{mf['path']:<35s} | {mf['lines']:4d} lines | classes: {mf['classes']} | funcs: {mf['functions'][:4]}")

paper_repro_files = [p for p in data['python_files'] if p['path'].startswith('paper_reproduction/')]
print(f"\nPaper Reproduction files ({len(paper_repro_files)}):\n")
for pr in paper_repro_files:
    print(f"{pr['path']:<60s} | {pr['lines']:4d} lines | classes: {pr['classes']}")

prev_exp_files = [p for p in data['python_files'] if p['path'].startswith('previous_neurotensor_experiments/')]
print(f"\nPrevious NeuroTensor files ({len(prev_exp_files)}):\n")
for pe in prev_exp_files:
    print(f"{pe['path']:<65s} | {pe['lines']:4d} lines")

