import os
import sys

# Add apps/streamlit_demo to sys.path
ROOT = os.path.abspath('e:/Rakhal_Tensor')
APP_DIR = os.path.join(ROOT, 'apps', 'streamlit_demo')
sys.path.insert(0, APP_DIR)
sys.path.insert(0, ROOT)

print("Testing Streamlit demo backend initialization...")
try:
    from demo_backend import DemoBackend
    backend = DemoBackend(expname='icvl')
    print(f"DemoBackend successfully initialized! Dataset: {backend.expname}, Shape: ({backend.H}, {backend.W}, {backend.nwvl}), Device: {backend.device}")
    print("Streamlit demo startup test PASSED!")
except Exception as e:
    print(f"FAILED Streamlit demo test: {e}")
    import traceback
    traceback.print_exc()

