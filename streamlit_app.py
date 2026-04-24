"""Root entry point for Streamlit Cloud — redirects to app/app.py."""
import os, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from app.app import main
main()
