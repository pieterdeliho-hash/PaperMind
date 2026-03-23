"""
PaperMind - Streamlit Cloud Entry Point
"""
import sys
import os
import subprocess
from pathlib import Path

# Auto-regenerate embeddings if missing (first deployment)
embeddings_path = Path("data/processed/embeddings_512.npy")
image_embeddings_path = Path("data/processed/image_embeddings.npy")

if not embeddings_path.exists() or not image_embeddings_path.exists():
    print("=" * 70)
    print("FIRST-TIME SETUP: Generating embeddings...")
    print("This takes ~5-10 minutes on Streamlit Cloud")
    print("=" * 70)
    
    if not embeddings_path.exists():
        print("\n[1/2] Generating text embeddings...")
        subprocess.run(["python", "src/generate_embeddings.py"], check=True)
    
    if not image_embeddings_path.exists():
        print("\n[2/2] Generating image embeddings...")
        subprocess.run(["python", "src/generate_image_embeddings.py"], check=True)
    
    print("\n✅ Embeddings generated! App will now start.\n")

# Check if embeddings exist, if not generate them
embeddings_path = Path("data/processed/embeddings_512.npy")
if not embeddings_path.exists():
    import subprocess
    import streamlit as st

    st.info("First-time setup: Generating embeddings (this takes ~5 minutes)...")

    with st.spinner("Generating text embeddings..."):
        subprocess.run(["python", "src/generate_embeddings.py"], check=True)

    with st.spinner("Generating image embeddings..."):
        subprocess.run(["python", "src/generate_image_embeddings.py"], check=True)

    st.success("Embeddings generated! Reloading app...")
    st.rerun()

# Add src directory to Python path
project_root = Path(__file__).parent.resolve()
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Change to project root so relative paths work
os.chdir(str(project_root))

# Execute web_ui.py with UTF-8 encoding
with open(src_path / "web_ui.py", "r", encoding="utf-8") as f:
    exec(f.read())