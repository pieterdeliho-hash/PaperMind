"""
PaperMind - Streamlit Cloud Entry Point
"""
import sys
import os
import subprocess
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Check if embeddings exist, generate if needed
embeddings_path = Path("data/processed/embeddings_512.npy")
image_embeddings_path = Path("data/processed/image_embeddings.npy")

if not embeddings_path.exists() or not image_embeddings_path.exists():
    import streamlit as st
    
    st.info("🔄 First-time setup: Generating embeddings (this takes ~5-10 minutes)")
    
    try:
        if not embeddings_path.exists():
            with st.spinner("Generating text embeddings..."):
                from generate_embeddings import EmbeddingGenerator
                
                generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
                generator.generate_embeddings(
                    chunks_file="data/processed/chunks_recursive_512.json",
                    output_file="data/processed/embeddings_512.json",
                    batch_size=32
                )
                st.success("✅ Text embeddings generated!")
        
        if not image_embeddings_path.exists():
            with st.spinner("Generating image embeddings..."):
                from generate_image_embeddings import ImageEmbeddingGenerator
                
                generator = ImageEmbeddingGenerator(model_name="openai/clip-vit-base-patch32")
                generator.generate_embeddings(
                    images_metadata_file="data/processed/faiss_image_index/images_metadata.pkl",
                    output_file="data/processed/image_embeddings.json",
                    batch_size=32
                )
                st.success("✅ Image embeddings generated!")
        
        st.success("✅ All embeddings generated! Reloading app...")
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error generating embeddings: {e}")
        st.stop()

# Add src directory to Python path
project_root = Path(__file__).parent.resolve()
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Change to project root so relative paths work
os.chdir(str(project_root))

# Execute web_ui.py with UTF-8 encoding
with open(src_path / "web_ui.py", "r", encoding="utf-8") as f:
    exec(f.read())