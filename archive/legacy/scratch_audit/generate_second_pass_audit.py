import os
import json
import hashlib

ROOT = os.path.abspath('e:/Rakhal_Tensor')
AUDIT_DIR = os.path.join(ROOT, 'codebase_audit')
os.makedirs(AUDIT_DIR, exist_ok=True)

print("Starting generation of Second-Pass Consistency Audit Documents...")

