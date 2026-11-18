import json

with open('scratch/comprehensive_audit_data.json', encoding='utf-8') as f:
    data = json.load(f)

root_files = [p for p in data['python_files'] if '/' not in p['path']]
print(f"Total root Python files: {len(root_files)}\n")
for rf in root_files:
    print(f"{rf['path']:<35s} | {rf['lines']:4d} lines | classes: {rf['classes']} | doc: {rf['doc'][:40]}")

