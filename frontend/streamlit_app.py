"""Compatibility bridge for Streamlit Cloud legacy settings."""
import os, sys

# ROOT is one level up from this file
FILE_PATH = os.path.abspath(__file__)
FRONTEND_DIR = os.path.dirname(FILE_PATH)
ROOT = os.path.dirname(FRONTEND_DIR)

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Execute the real entry point
ENTRY_POINT = os.path.join(ROOT, "streamlit_app.py")
with open(ENTRY_POINT, encoding="utf-8") as f:
    code = compile(f.read(), ENTRY_POINT, 'exec')
    exec(code, {"__name__": "__main__", "__file__": ENTRY_POINT})

