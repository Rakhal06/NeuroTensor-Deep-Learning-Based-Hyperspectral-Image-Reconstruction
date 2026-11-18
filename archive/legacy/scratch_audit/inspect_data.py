import json

with open('scratch/comprehensive_audit_data.json', encoding='utf-8') as f:
    data = json.load(f)

print('Duplicate groups:')
for h, paths in data['duplicates'].items():
    print('---')
    for p in paths:
        print('  ', p)

print('\nTotal python files:', len(data['python_files']))
print('\nDirectories containing Python files:')
dirs = {}
for p in data['python_files']:
    d = '/'.join(p['path'].split('/')[:-1]) or '.'
    dirs[d] = dirs.get(d, 0) + 1

for d, count in sorted(dirs.items()):
    print(f"  {d}: {count} files")

