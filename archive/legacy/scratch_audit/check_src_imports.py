import os

SRC = 'src/deeptensor'

for root, dirs, files in os.walk(SRC):
    for f in files:
        if f.endswith('.py'):
            fp = os.path.join(root, f)
            with open(fp, 'r', errors='ignore') as sfp:
                txt = sfp.read()
                if 'from deep_models' in txt or 'import deep_models' in txt:
                    print(f"Bare deep_models in: {fp}")
                if 'from modules' in txt or 'import modules' in txt:
                    print(f"Bare modules in: {fp}")

