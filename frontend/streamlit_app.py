"""Compatibility bridge for Streamlit Cloud legacy settings."""
import os, sys
# Add the actual project root to the path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
# Execute the real entry point
exec(open(os.path.join(ROOT, "streamlit_app.py"), encoding="utf-8").read())
