"""Root entry point for Streamlit Cloud — redirects to app/app.py."""
import os, sys, runpy
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
# Streamlit runs this file; we execute app/app.py in the same process
exec(open(os.path.join(ROOT, "app", "app.py"), encoding="utf-8").read())
