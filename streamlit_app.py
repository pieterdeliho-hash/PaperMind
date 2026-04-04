import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

embeddings_path = Path("data/processed/embeddings_512.npy")
image_embeddings_path = Path("data/processed/image_embeddings.npy")

if not embeddings_path.exists():
    st.error("Text embeddings missing. Please contact developer.")
    st.stop()

if not image_embeddings_path.exists():
    st.warning("Image embeddings missing - using text search only.")

from web_ui import run_app

if __name__ == "__main__":
    run_app()