with open('run_figure11.py', 'r') as f:
    lines = f.readlines()
print(f"Root run_figure11.py: {len(lines)} lines")
print("First 25 lines:")
for l in lines[:25]:
    print(l.rstrip())

