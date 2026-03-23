"""
PaperMind Web UI
Interactive multi-modal RAG interface with Streamlit
"""

import streamlit as st
from pathlib import Path
import sys
import os
from PIL import Image

# Project root is already set by streamlit_app.py
# Just use current working directory
project_root = Path.cwd()
sys.path.insert(0, str(project_root))

# Import RAG pipeline
from multimodal_rag_pipeline import MultiModalRAG

def get_retrieval_params(query: str) -> tuple:
    """
    Automatically determine optimal k_text and k_images based on query complexity

    Args:
        query: User's question

    Returns:
        (k_text, k_images) tuple
    """
    query_lower = query.lower()
    word_count = len(query.split())

    # Visual queries need more images
    visual_keywords = ['show', 'diagram', 'figure', 'visualiz', 'image', 'picture', 'graph', 'chart', 'plot']
    if any(keyword in query_lower for keyword in visual_keywords):
        return (4, 5)  # Fewer text, more images

    # Broad/complex queries need more text sources
    complex_keywords = ['compare', 'difference', 'vs', 'versus', 'explain', 'comprehensive', 'detail', 'analyze']
    if word_count > 15 or any(keyword in query_lower for keyword in complex_keywords):
        return (7, 3)  # More text sources

    # Default: balanced retrieval
    return (5, 3)

# Initialize
rag = MultiModalRAG()

# Page config
st.set_page_config(
    page_title="PaperMind - AI Research Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag' not in st.session_state:
    with st.spinner("Loading PaperMind... (this takes ~10 seconds)"):
        st.session_state.rag = MultiModalRAG()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "💬 Chat"

# Header
st.markdown('<div class="main-header">🧠 PaperMind</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI Research Assistant with Multi-Modal RAG</div>', unsafe_allow_html=True)

# Sidebar UI
# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    # Advanced mode toggle (hidden by default)
    advanced_mode = st.checkbox("🔧 Advanced Mode", value=False, help="Show advanced retrieval settings")

    if advanced_mode:
        st.subheader("Retrieval Configuration")
        st.caption("⚠️ Only adjust if you know what you're doing")
        text_k = st.slider("Text chunks to retrieve", 1, 10, 5, help="More chunks = more context but slower")
        image_k = st.slider("Images to retrieve", 0, 5, 3, help="0 = text-only mode")
        st.info("💡 Tip: System auto-adjusts these based on your query when Advanced Mode is off")
    else:
        # Auto-detection happens in ask_question function
        text_k = None  # Will be determined automatically
        image_k = None

    st.subheader("System Information")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Text Chunks", f"{st.session_state.rag.text_index.ntotal:,}")
    with col2:
        st.metric("Images", f"{st.session_state.rag.image_index.ntotal:,}")

    st.subheader("About")
    st.markdown("""
    **PaperMind** uses Retrieval-Augmented Generation to answer questions about transformer research papers.

    **Features:**
    - 200+ research papers indexed
    - Multi-modal search (text + images)
    - GPT-3.5 powered comprehensive answers
    - Automatic source citation

    **Tech Stack:**
    - FAISS vector search
    - CLIP image embeddings
    - OpenAI API
    - Streamlit UI
    """)

    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


def ask_question(question):
    """Process a question and add to history"""

    # Determine retrieval parameters
    if text_k is None or image_k is None:
        # Auto-detect based on query
        auto_text_k, auto_image_k = get_retrieval_params(question)
    else:
        # Use manual settings from Advanced Mode
        auto_text_k, auto_image_k = text_k, image_k

    with st.spinner("Searching papers and generating answer..."):
        result = st.session_state.rag.query(
            question,
            text_k=auto_text_k,
            image_k=auto_image_k,
            verbose=False
        )

    # Store with new metadata structure
    st.session_state.chat_history.append({
        'question': question,
        'answer': result['answer'],
        'text_sources': result['text_sources'],
        'image_sources': result['image_sources'],
        'metadata': result['metadata']
    })

    # Switch to chat tab
    st.session_state.active_tab = "💬 Chat"

# Main tabs
tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Example Queries", "ℹ️ How It Works"])

with tab1:
    st.subheader("Ask questions about transformer research papers")

    # Display chat history
    for entry in st.session_state.chat_history:
        # User message
        with st.chat_message("user"):
            st.write(entry['question'])

        # Assistant message
        with st.chat_message("assistant"):
            st.write(entry['answer'])

            # Sources expander
            with st.expander("📄 View Sources", expanded=False):
                st.markdown("### Text Sources:")
                for i, chunk in enumerate(entry['text_sources'], 1):
                    paper = chunk['paper']
                    chunk_id = chunk['chunk_id']
                    score = chunk['score']

                    st.markdown(f"**[{i}] {paper}**")
                    st.caption(f"Chunk {chunk_id} | Relevance: {score:.3f}")

                    # Show full text in a scrollable container
                    with st.container():
                        st.markdown(f"```\n{chunk['text']}\n```")
                    st.divider()

                if entry['image_sources']:
                    st.markdown("### Figure Sources:")
                    cols = st.columns(len(entry['image_sources']))
                    for j, (col, img) in enumerate(zip(cols, entry['image_sources']), 1):
                        with col:
                            st.markdown(f"**[Figure {j}]**")
                            st.markdown(f"{img['paper'][:40]}...")
                            st.markdown(f"Page {img['page']} | Score: {img['score']:.3f}")

                            try:
                                # Convert path for cross-platform compatibility
                                image_path_str = img['path'].replace('\\', '/')
                                image_path = Path(image_path_str)

                                # Make absolute if relative
                                if not image_path.is_absolute():
                                    project_root = Path.cwd()
                                    image_path = project_root / image_path

                                # Resolve and check existence
                                image_path = image_path.resolve()

                                if image_path.exists():
                                    # Open and display image
                                    with Image.open(str(image_path)) as img_obj:
                                        st.image(img_obj)
                                else:
                                    st.warning(f"Image not found: {image_path}")
                            except Exception as e:
                                st.warning(f"Error loading image: {str(e)}")

            # Handles old and new structure
            if 'metadata' in entry:
                # New structure
                meta = entry['metadata']
                st.caption(
                    f"⏱️ {meta['latency']}s | "
                    f"💬 {meta['total_tokens']} tokens | "
                    f"💰 ${meta['estimated_cost_usd']} | "
                    f"🤖 {meta['model']}"
                )
            else:
                # Old structure (backward compatibility)
                st.caption(
                    f"⏱️ {entry.get('latency', 'N/A')}s | "
                    f"💬 {entry.get('tokens_used', 'N/A')} tokens | "
                    f"🤖 {entry.get('model', 'gpt-3.5-turbo')}"
                )

    # Chat input
    question = st.chat_input("Ask a question about transformers...")
    if question:
        ask_question(question)
        st.rerun()

with tab2:
    st.subheader("🎯 Example Questions to Try")
    st.info("💡 Click any button to ask that question")

    st.markdown("### Architecture & Concepts")
    col1, col2 = st.columns(2)

    examples_arch = [
        "What is the transformer architecture?",
        "How does self-attention work?",
        "Explain multi-head attention",
        "What are the key components of BERT?",
    ]

    for i, example in enumerate(examples_arch):
        with col1 if i % 2 == 0 else col2:
            if st.button(example, key=f"arch_{i}", use_container_width=True):
                ask_question(example)
                st.rerun()

    st.markdown("### Visual Questions")
    col1, col2 = st.columns(2)

    examples_visual = [
        "Show me transformer architecture diagrams",
        "What do attention visualizations look like?",
        "Are there any training loss curves?",
        "Show me vision transformer patches",
    ]

    for i, example in enumerate(examples_visual):
        with col1 if i % 2 == 0 else col2:
            if st.button(example, key=f"visual_{i}", use_container_width=True):
                ask_question(example)
                st.rerun()

    st.markdown("### Comparisons & Analysis")
    col1, col2 = st.columns(2)

    examples_compare = [
        "Compare vision transformer architectures",
        "Transformers vs RNNs advantages?",
        "How do efficient transformers work?",
        "Common transformer training datasets?",
    ]

    for i, example in enumerate(examples_compare):
        with col1 if i % 2 == 0 else col2:
            if st.button(example, key=f"compare_{i}", use_container_width=True):
                ask_question(example)
                st.rerun()

with tab3:
    st.subheader("🔬 How PaperMind Works")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 1️⃣ Document Processing")
        st.markdown("""
        - **57 research papers** from ArXiv
        - **3,431 text chunks** (512 tokens each)
        - **2,655 images** extracted from papers
        - Recursive character splitting for optimal chunking
        """)

        st.markdown("### 2️⃣ Embedding Generation")
        st.markdown("""
        - **Text**: all-MiniLM-L6-v2 (384-dim)
        - **Images**: CLIP ViT-B/32 (512-dim)
        - Semantic embeddings capture meaning
        - Vector representations enable similarity search
        """)

    with col2:
        st.markdown("### 3️⃣ Retrieval")
        st.markdown("""
        - **FAISS** vector search (sub-millisecond)
        - Dual indexes: text + images
        - Top-k retrieval based on cosine similarity
        - Multi-modal: retrieve both text and figures
        """)

        st.markdown("### 4️⃣ Answer Generation")
        st.markdown("""
        - **GPT-3.5-turbo** for natural language answers
        - Context-aware responses
        - Citation tracking ([Source N], [Figure N])
        - Combines text evidence with visual content
        """)

    st.markdown("### 🎯 System Architecture")
    st.code("""
User Query
    ↓
[Text Embedding]     [Image Embedding (CLIP)]
    ↓                        ↓
[FAISS Text Search]  [FAISS Image Search]
    ↓                        ↓
[Top 3 chunks]       [Top 2 images]
    ↓________________________↓
              ↓
    [Context Assembly] 
              ↓
    [GPT-3.5 Generation]
              ↓
    [Answer + Citations]
    """, language="text")

    st.markdown("### 📊 Performance")
    perf_col1, perf_col2, perf_col3 = st.columns(3)
    with perf_col1:
        st.metric("Retrieval Speed", "<100ms", "FAISS optimization")
    with perf_col2:
        st.metric("Total Latency", "~2-3s", "LLM generation time")
    with perf_col3:
        st.metric("Cost per Query", "$0.003", "GPT-3.5 pricing")