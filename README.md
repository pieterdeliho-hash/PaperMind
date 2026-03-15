# PaperMind: AI Research Assistant with RAG

**Production-ready Retrieval-Augmented Generation system for academic research papers**

Transform how you interact with research literature. PaperMind uses semantic search and large language models to answer questions from 50+ academic papers with precise citations.

---

## Current Status

**Timeline**: March 4 - April 5, 2025 (1-month sprint)  
**Progress**: Week 1 Complete (Days 1-6 done in 1 day!)

**Completed**:
- PDF ingestion pipeline (57 papers indexed)
- Production-grade text chunking (3,431 chunks, 512 tokens, recursive splitting)
- Vector embeddings (all-MiniLM-L6-v2, 384-dim)
- FAISS vector search (sub-millisecond retrieval)
- LLM integration (GPT-3.5-turbo with citations)
- End-to-end RAG pipeline working

**Next Up**:
- Multi-modal retrieval (text + images with CLIP)
- Web interface (Streamlit)
- Evaluation framework

---

## ✨ Features

### Core RAG Pipeline
- [x] ArXiv paper downloader with API integration
- [x] PDF text extraction (PyPDF2 + pdfplumber fallback)
- [x] Token-aware recursive chunking (LangChain standard)
- [x] Semantic embeddings (sentence-transformers)
- [ ] FAISS vector search (k-NN retrieval)
- [ ] LLM answer generation (GPT-3.5/4 via OpenAI API)
- [ ] Citation tracking with source attribution

### Advanced Features (Week 2-3)
- [ ] Multi-modal retrieval (text + images with CLIP)
- [ ] Interactive web interface (Streamlit)
- [ ] Evaluation metrics (retrieval accuracy, answer quality)
- [ ] Query decomposition for complex questions
- [ ] Conversational memory

---

## 🛠️ Tech Stack

**Core**:
- Python 3.11
- sentence-transformers (embeddings)
- FAISS (vector search)
- LangChain (text splitting, orchestration)
- OpenAI API (LLM)

**Data Processing**:
- PyPDF2, pdfplumber (PDF extraction)
- tiktoken (tokenization)
- arxiv (paper download)

**Production**:
- Streamlit (UI) or FastAPI + React
- NumPy (vector operations)
- tqdm (progress tracking)

---

## 📦 Installation
```bash
# Clone repository
cd PaperMind

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (one-time)
python -c "import nltk; nltk.download('punkt')"
```

---

## 🚦 Quick Start

**1. Download papers**:
```bash
python src/download_papers.py
```

**2. Extract and chunk**:
```bash
python src/pdf_extractor.py
python src/text_chunker.py
```

**3. Generate embeddings**:
```bash
python src/generate_embeddings.py
```

**4. Query the system** *(coming Day 6)*:
```bash
python src/query_rag.py "What is the attention mechanism?"
```

---

## Performance Metrics

**Current Status (57 Papers)**:
- Total papers: 57
- Total chunks: 3,431
- Avg chunk size: 481.6 tokens
- Embedding dim: 384
- Retrieval quality: 0.51-0.59 similarity scores
- Processing speed: 55 chunks/sec
- Query latency: <5ms (FAISS) + 1.7-3.6s (LLM)

**Coverage**:
- Core architectures: Transformer, BERT, GPT, ViT
- Attention mechanisms: Self-attention, multi-head, cross-attention
- Applications: NLP, computer vision, time series
- Recent advances: Efficient transformers, sparse attention, linear attention
- Vision models: Swin, ResT, hierarchical transformers

---

## 📁 Project Structure
```
PaperMind/
├── data/
│   ├── papers/              # Raw PDF files
│   ├── processed/           # Extracted text, chunks, embeddings
├── src/
│   ├── download_papers.py   # ArXiv downloader
│   ├── pdf_extractor.py     # PDF → text
│   ├── text_chunker.py      # Text → chunks
│   ├── generate_embeddings.py  # Chunks → vectors
│   ├── build_faiss_index.py    # (Day 5)
│   └── query_rag.py            # (Day 6)
├── notebooks/               # Experiments, analysis
├── tests/                   # Unit tests
├── README.md
├── DEVLOG.md               # Development journal
├── ARCHITECTURE.md         # Technical design
└── requirements.txt
```

---

## 🎯 Development Roadmap

### Week 1: Core RAG ✅
- [x] PDF ingestion & chunking
- [x] Embeddings generation
- [ ] FAISS indexing
- [ ] LLM integration
- [ ] First working query

### Week 2: Multi-Modal
- [ ] Image extraction (CLIP)
- [ ] Multi-modal retrieval
- [ ] Evaluation framework

### Week 3: Production
- [ ] Web UI
- [ ] Citation tracking
- [ ] Error handling

### Week 4: Polish
- [ ] Documentation
- [ ] Demo video
- [ ] Code cleanup

---

## 📖 How It Works

1. **Ingestion**: Download papers from ArXiv, extract text
2. **Chunking**: Split into 512-token chunks with 20% overlap
3. **Embedding**: Convert chunks to 384-dim vectors (semantic meaning)
4. **Indexing**: Store in FAISS for fast similarity search
5. **Retrieval**: Find top-k most relevant chunks for query
6. **Generation**: LLM generates answer from retrieved context
7. **Citation**: Display sources with paper names and locations

---

## 🔬 Technical Highlights

**Why This Approach?**
- **Token-based chunking**: Matches LLM input limits exactly
- **Recursive splitting**: Respects document structure (paragraphs → sentences)
- **20% overlap**: Prevents context loss at boundaries
- **FAISS**: 100x faster than naive similarity search
- **MiniLM embeddings**: 384-dim balances quality and speed

**Design Decisions**:
- 512 tokens per chunk (industry standard)
- Cosine similarity (standard for embeddings)
- NPY format for embeddings (8x smaller than JSON)
- Local-first (no API costs for embeddings)

---

## 📝 Citation

If you use this project, please cite:
```
PaperMind: Production RAG System for Academic Research
Author: [Your Name]
Year: 2025
GitHub: [Your Repo URL]
```

---

## 📄 License

MIT License - see LICENSE file

---

## 🙏 Acknowledgments

- LangChain for text splitting utilities
- Sentence-Transformers for embedding models
- ArXiv for paper access
- FAISS for vector search

---

**Status Updates**:

**Week 1 (March 5)**:
- Day 1: Environment setup, project structure
- Day 2: Downloaded 14 papers, perfect extraction
- Day 3: Recursive chunking, 931 chunks generated
- Day 4: Embeddings working, semantic similarity verified
- Day 5: FAISS indexing, sub-millisecond search
- Day 6: LLM integration, end-to-end RAG working
- Corpus expansion: 57 papers, 3,431 chunks, improved retrieval

**Week 1 Status**: COMPLETE (all done in 1 day)

---

*Last updated: March 5, 2025*