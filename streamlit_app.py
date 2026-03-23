"""
PaperMind - Streamlit Cloud Entry Point
"""
import sys
import os
import subprocess
import streamlit as st
from pathlib import Path

# Check embeddings exist
embeddings_path = Path("data/processed/embeddings_512.npy")
image_embeddings_path = Path("data/processed/image_embeddings.npy")

if not embeddings_path.exists():
    st.error("⚠️ Text embeddings missing! Please contact developer.")
    st.stop()

if not image_embeddings_path.exists():
    st.warning("⚠️ Image embeddings missing - image search disabled. Text search only.")

# Add src directory to Python path
project_root = Path(__file__).parent.resolve()
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Change to project root so relative paths work
os.chdir(str(project_root))

# Execute web_ui.py with UTF-8 encoding
with open(src_path / "web_ui.py", "r", encoding="utf-8") as f:
    exec(f.read())