import json
import os

with open('scratch_py_inventory.json') as fp:
    py_files = json.load(fp)

print(f"Total files: {len(py_files)}\n")

for f in py_files:
    d = os.path.dirname(f['path']) or '.'
    p = f['path']
    lines = f['lines']
    classes = len(f['classes'])
    funcs = len(f['functions'])
    doc = f.get('doc', '')
    print(f"{p:<65s} | {lines:4d} lines | {classes:2d} classes | {funcs:2d} funcs | {doc[:50]}")

