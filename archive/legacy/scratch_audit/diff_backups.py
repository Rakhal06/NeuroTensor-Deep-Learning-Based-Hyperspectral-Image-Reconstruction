import difflib

with open('demo.py', 'r', encoding='utf-8') as f1, open('demo_backup.py', 'r', encoding='utf-8') as f2:
    lines1 = f1.readlines()
    lines2 = f2.readlines()

diff = list(difflib.unified_diff(lines2, lines1, fromfile='demo_backup.py', tofile='demo.py', n=1))
print("Diff between demo_backup.py and demo.py (first 30 lines):")
for d in diff[:30]:
    print(d.rstrip())

with open('demo_backend.py', 'r', encoding='utf-8') as f1, open('demo_backend_backup.py', 'r', encoding='utf-8') as f2:
    lines1 = f1.readlines()
    lines2 = f2.readlines()

diff2 = list(difflib.unified_diff(lines2, lines1, fromfile='demo_backend_backup.py', tofile='demo_backend.py', n=1))
print("\nDiff between demo_backend_backup.py and demo_backend.py (first 30 lines):")
for d in diff2[:30]:
    print(d.rstrip())

