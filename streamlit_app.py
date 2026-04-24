"""Root entry point for Streamlit Cloud — redirects to app/app.py."""
import os, sys, runpy
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
# Streamlit runs this file; we import and run the app
from app import app

