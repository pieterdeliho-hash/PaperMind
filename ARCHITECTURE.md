# PaperMind System Architecture

## Overview
PaperMind is a multi-modal Retrieval-Augmented Generation (RAG) system that combines text and image search to answer questions about AI research papers. Built for transformer and deep learning papers, it retrieves relevant text chunks and figures, then synthesizes comprehensive answers using GPT-3.5-turbo.

---

## High-Level Flow
```
User Query
    ↓
[Query Processing] → Tokenization & Understanding
    ↓
[Parallel Embedding Generation]
    ├─→ Text Query Embedding (all-MiniLM-L6-v2, 384-dim)
    └─→ Image Query Embedding (CLIP ViT-B/32, 512-dim)
    ↓
[Parallel Vector Search]
    ├─→ Text FAISS Index → Top-5 chunks (512 tokens each)
    └─→ Image FAISS Index → Top-3 images (with metadata)
    ↓
[Context Assembly]
    ├─→ Format text chunks with [Source N] citations
    ├─→ Include page numbers and relevance scores
    └─→ Attach image metadata (source, page, similarity)
    ↓
[LLM Synthesis - GPT-3.5-turbo]
    ├─→ Enhanced prompt (2-4 paragraph structure)
    ├─→ Temperature: 0.3 (balanced accuracy/creativity)
    └─→ Max tokens: 800 (comprehensive responses)
    ↓
[Response Rendering]
    ├─→ Main answer with inline citations
    ├─→ Expandable text source viewer (full chunks)
    ├─→ Image gallery (3 figures with scores)
    └─→ Metadata (latency, tokens, cost)
```

---

## Components

### 1. Document Ingestion Pipeline

**Purpose**: Extract and process research papers into searchable chunks

**Input**: 
- PDF files from arXiv (transformers, AI, NLP papers)
- Target corpus: 200+ papers

**Process Flow**:
```
PDFs → Text Extraction → Chunking → Image Extraction → Storage
```

**Sub-Components**:

#### 1.1 Text Extraction (`pdf_extractor.py`)
- **Library**: PyMuPDF (fitz) - faster and more accurate than PyPDF2
- **Process**: 
  - Iterate through all pages
  - Extract raw text with metadata
  - Handle multi-column layouts
  - Preserve section structure
- **Output**: `data/processed/extracted_text.json`
```json
  [
    {
      "source": "Attention Is All You Need_1706.03762v7.pdf",
      "page": 3,
      "text": "The Transformer model architecture..."
    }
  ]
```

**Why PyMuPDF over PyPDF2:**
- 3-5x faster extraction
- Better handling of complex layouts
- More accurate text positioning
- Built-in image extraction support

#### 1.2 Text Chunking (`text_chunker.py`)
- **Strategy**: Recursive Character Splitting (LangChain)
- **Configuration**:
  - Chunk size: 512 tokens (~384 words)
  - Overlap: 102 tokens (20%)
  - Separators: `\n\n`, `\n`, `. `, ` `
- **Process**:
  - Split on paragraph boundaries first
  - Fall back to sentence/word boundaries
  - Preserve semantic coherence
  - Add metadata (source, page, chunk_id)
- **Output**: `data/processed/chunks_recursive_512.json`
```json
  [
    {
      "text": "The Transformer architecture relies entirely on...",
      "metadata": {
        "source": "Attention Is All You Need_1706.03762v7.pdf",
        "page": 3,
        "chunk_id": 42,
        "start_char": 1523,
        "end_char": 3847
      }
    }
  ]
```

**Why these choices:**
- **512 tokens**: Optimal for embedding models (max context 512)
- **20% overlap**: Prevents splitting key concepts across chunks
- **Recursive splitting**: Maintains semantic boundaries (paragraphs → sentences → words)

#### 1.3 Image Extraction (`extract_images.py`)
- **Library**: PyMuPDF (fitz)
- **Process**:
  - Scan all PDF pages for embedded images
  - Extract images as PNG with 300 DPI
  - Filter small/irrelevant images (< 100x100 px)
  - Save with metadata (source PDF, page number)
- **Output**: 
  - Images: `data/processed/images/*.png`
  - Metadata: `data/processed/images_metadata.json`
```json
  [
    {
      "path": "data/processed/images/Attention_Is_All_You_Need_page3_img1.png",
      "source": "Attention Is All You Need_1706.03762v7.pdf",
      "page": 3,
      "width": 800,
      "height": 600
    }
  ]
```

**Trade-offs:**
- Captures diagrams, architectures, plots
- Doesn't extract tables (future: table detection)
- No OCR for text-in-images (future: Tesseract integration)

---

### 2. Embedding Generation

**Purpose**: Convert text chunks and images into dense vector representations

#### 2.1 Text Embeddings (`generate_embeddings.py`)
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimensions**: 384
- **Performance**: ~2,800 chunks/second (GPU), ~400 chunks/second (CPU)
- **Process**:
  - Load model from HuggingFace
  - Batch encode chunks (batch_size=32)
  - Normalize embeddings (L2 norm)
  - Save as numpy array
- **Output**: `data/processed/embeddings_512.npy` (shape: [N, 384])

**Why all-MiniLM-L6-v2:**
- Fast inference (6-layer transformer)
- Good semantic understanding for technical text
- Balanced precision/recall on retrieval tasks
- Small model size (~80MB)

**Alternative considered:**
- `text-embedding-ada-002` (OpenAI): Better quality, but costs $0.0001/1K tokens
- `all-mpnet-base-v2`: Higher quality, but 2x slower

#### 2.2 Image Embeddings (`generate_image_embeddings.py`)
- **Model**: `openai/clip-vit-base-patch32`
- **Dimensions**: 512
- **Performance**: ~50 images/second (GPU), ~10 images/second (CPU)
- **Process**:
  - Load CLIP vision encoder
  - Preprocess images (resize, normalize)
  - Extract visual features
  - Save as numpy array
- **Output**: `data/processed/image_embeddings.npy` (shape: [M, 512])

**Why CLIP:**
- Trained on text-image pairs (understands semantic similarity)
- Can match text queries to visual content
- Robust to different diagram styles
- Open-source and lightweight

---

### 3. Vector Database (FAISS)

**Purpose**: Fast approximate nearest neighbor search

#### 3.1 Text Index (`build_faiss_index.py`)
- **Index Type**: IndexFlatL2 (exact L2 distance)
- **Size**: ~4 MB per 10,000 chunks
- **Build Time**: <1 second
- **Search Time**: ~50ms for top-5 from 10,000 chunks
- **Output**: `data/processed/faiss_index/faiss_index.bin`

**Why IndexFlatL2:**
- Exact search (no approximation errors)
- Fast enough for <100K vectors
- No training required
- Deterministic results

**Alternative for scaling:**
- `IndexIVFFlat`: For 100K-1M vectors
- `IndexHNSW`: For >1M vectors with faster search

#### 3.2 Image Index (`build_image_faiss_index.py`)
- **Index Type**: IndexFlatL2
- **Size**: ~2 MB per 5,000 images
- **Build Time**: <1 second
- **Search Time**: ~30ms for top-3 from 5,000 images
- **Output**: `data/processed/faiss_image_index/faiss_image_index.bin`

---

### 4. RAG Pipeline

#### 4.1 Text-Only RAG (`rag_pipeline.py`)
- **Input**: User query (string)
- **Process**:
  1. Embed query with all-MiniLM-L6-v2
  2. Search FAISS index for top-k chunks
  3. Assemble context from retrieved chunks
  4. Call GPT-3.5-turbo with context + query
  5. Return answer with sources
- **Output**: Answer + text sources

#### 4.2 Multi-Modal RAG (`multimodal_rag_pipeline.py`)
- **Input**: User query (string)
- **Configuration**:
  - k_text: 5 (retrieve 5 text chunks)
  - k_images: 3 (retrieve 3 images)
- **Process**:
```python
  # Parallel retrieval
  text_results = search_text(query, k=5)
  image_results = search_images(query, k=3)
  
  # Context assembly with citations
  context = format_with_citations(text_results)
  
  # LLM synthesis
  prompt = build_enhanced_prompt(query, context)
  answer = gpt35_turbo(prompt, max_tokens=800)
  
  # Return comprehensive result
  return {
      "answer": answer,
      "text_sources": text_results,
      "image_sources": image_results,
      "metadata": {tokens, cost, latency}
  }
```
- **Output**: Answer + text sources + image sources + metadata

**Enhanced Prompt Structure:**
```
System: "You are an expert AI research assistant..."

User: """
Based on the following 5 sources, provide a comprehensive answer...

Guidelines:
- Synthesize ALL sources into 2-4 paragraphs
- Include technical details
- Cite sources by number [Source N]
- Acknowledge conflicts if any

Question: {query}

Sources:
[Source 1] Paper_Name.pdf (Page 3) | Relevance: 0.521
{chunk_text_1}

[Source 2] ...

Provide detailed response:
"""
```

**Response Generation Parameters:**
- Model: `gpt-3.5-turbo`
- Temperature: 0.3 (balanced creativity/accuracy)
- Max tokens: 800 (allows 2-4 paragraph responses)
- Top-p: 1.0 (no nucleus sampling)

---

### 5. Web Interface (`web_ui.py` + `streamlit_app.py`)

**Framework**: Streamlit 1.31.0

**Layout**:
```
┌─────────────────────────────────────────┐
│  PaperMind: AI Research Assistant    │
├─────────────────────────────────────────┤
│  [Text Input: "What is..."]      [Ask]  │
│                                         │
│  Example Questions: [Button] [Button]   │
├─────────────────────────────────────────┤
│  Answer (2-4 paragraphs)             │
│  "The transformer architecture uses..." │
│  [Source 1] [Source 2] ...              │
├─────────────────────────────────────────┤
│  View Sources (Expandable)           │
│  ├─ [Source 1] Paper.pdf (Page 3)       │
│  │   Chunk text preview...              │
│  ├─ [Source 2] ...                      │
├─────────────────────────────────────────┤
│  Figure Sources (Image Gallery)      │
│  [Image 1] [Image 2] [Image 3]          │
│  Score: 0.45  Score: 0.42  Score: 0.39  │
├─────────────────────────────────────────┤
│  Metadata                             │
│  Latency: 3.2s | Tokens: 3,019          │
│  Cost: $0.006 | Model: gpt-3.5-turbo    │
└─────────────────────────────────────────┘
```

**Key Features**:
- Responsive design (works on mobile)
- Real-time streaming responses (future)
- Expandable source viewer
- Image gallery with similarity scores
- Performance metrics display

**Path Handling (Cross-Platform)**:
```python
# Convert Windows paths to POSIX for Streamlit Cloud
image_path_str = img['path'].replace('\\', '/')
image_path = Path(image_path_str)

# Make absolute if relative
if not image_path.is_absolute():
    project_root = Path.cwd()
    image_path = project_root / image_path
```

---

## Evaluation Framework

### RAGAS Metrics (`evaluate_with_ragas.py`)

**Framework**: RAGAS v0.1.9 (Retrieval-Augmented Generation Assessment)

**Metrics Tracked**:

#### 1. Answer Relevancy (0.0 - 1.0)
- **Measures**: How well the answer addresses the question
- **Method**: Cosine similarity between question and answer embeddings
- **Interpretation**:
  - > 0.80: Excellent alignment
  - 0.60-0.80: Good, minor tangents
  - < 0.60: Answer doesn't fully address question
- **Target**: > 0.80

#### 2. Faithfulness (0.0 - 1.0)
- **Measures**: Answer grounding in retrieved context (hallucination detection)
- **Method**: NLI (Natural Language Inference) model checks if answer claims are entailed by context
- **Interpretation**:
  - > 0.90: Highly faithful, no hallucinations
  - 0.70-0.90: Mostly faithful, minor extrapolations
  - < 0.70: Contains unsupported claims
- **Target**: > 0.90

#### 3. Context Precision (0.0 - 1.0)
- **Measures**: Relevance of retrieved chunks to the question
- **Method**: Checks if ground truth answer is derivable from top-k chunks
- **Interpretation**:
  - > 0.70: Most chunks are relevant
  - 0.50-0.70: Mixed relevance
  - < 0.50: Many irrelevant chunks retrieved
- **Target**: > 0.70

#### 4. Context Recall (0.0 - 1.0)
- **Measures**: Completeness - did retrieval capture all necessary information?
- **Method**: Checks if all ground truth facts appear in retrieved context
- **Interpretation**:
  - > 0.75: Complete information retrieved
  - 0.50-0.75: Some gaps in coverage
  - < 0.50: Missing key information
- **Target**: > 0.75

**Test Suite**:
- **Size**: 10 diverse queries
- **Categories**:
  - Architecture questions (3)
  - Component questions (3)
  - Comparison questions (2)
  - Limitation questions (2)
- **Ground Truth**: Manually curated expected answers
- **Execution**: `python src/evaluate_with_ragas.py`
- **Results**: Saved to `data/evaluation/ragas_results.json`

---

## Performance Characteristics

### Corpus Statistics (Current)
- **Papers**: 200+ research papers
- **Text Chunks**: ~10,000 chunks (512 tokens each)
- **Images**: ~5,000 figures and diagrams
- **Total Embeddings**: ~50 MB (text + image)
- **Index Size**: ~150 MB (FAISS indexes + metadata)

### Retrieval Performance
- **Text Search Latency**: ~50ms (P50), ~120ms (P95)
- **Image Search Latency**: ~30ms (P50), ~80ms (P95)
- **Text Retrieval Precision@5**: 35%
- **Image Retrieval Precision@3**: 28%
- **Top-1 Text Relevance**: 0.539 (cosine similarity)
- **Top-1 Image Relevance**: 0.451 (cosine similarity)

### Response Generation
- **End-to-End Latency**: 3.5s (P50), 5.2s (P95)
  - Text search: 0.2s
  - Image search: 0.3s
  - LLM generation: 3.0s
- **Cost Per Query**: $0.006 USD (GPT-3.5-turbo)
  - Input tokens: ~2,500 (context)
  - Output tokens: ~500 (answer)
- **Throughput**: ~17 queries/minute (with rate limiting)

### System Resources
- **RAM Usage**: ~2 GB (models loaded)
- **Disk Space**: ~500 MB (full system)
- **GPU**: Optional (10x faster embedding generation)

---

## Trade-offs & Limitations

### Current Limitations

#### 1. Document Coverage
- **Not handling**: Scanned PDFs without text layer
  - **Future**: Add Tesseract OCR for scanned documents
- **Not handling**: Tables and equations
  - **Future**: Table extraction with camelot/tabula
  - **Future**: LaTeX equation parsing
- **Language**: English-only
  - **Future**: Multilingual models (mBERT, XLM-R)

#### 2. Retrieval Quality
- **No reranking**: Top-k results may include some irrelevant chunks
  - **Future**: Add cross-encoder reranking
- **No diversity**: Results can be redundant (multiple chunks from same paper)
  - **Future**: Implement MMR (Maximal Marginal Relevance)
- **Fixed k**: Always retrieves k=5 text, k=3 images
  - **Future**: Adaptive retrieval based on query complexity

#### 3. Response Generation
- **No conversation memory**: Each query is independent
  - **Future**: Add session state for multi-turn conversations
- **No streaming**: User waits 3-5 seconds for full response
  - **Future**: Stream GPT tokens in real-time
- **Single model**: Only GPT-3.5-turbo
  - **Future**: Support GPT-4, Claude, or local models

#### 4. Scalability
- **Corpus size**: Limited to ~200 papers (~10K chunks) for fast exact search
  - **Future**: Switch to IndexIVFFlat or IndexHNSW for 100K+ chunks
- **Memory**: All embeddings loaded in RAM
  - **Future**: Memory-mapped indexes for large-scale deployment
- **Single-node**: No distributed search
  - **Future**: Distributed FAISS or Pinecone/Weaviate

### Design Trade-offs

#### Trade-off 1: Exact vs. Approximate Search
**Choice**: IndexFlatL2 (exact search)
- **Pros**: Perfect recall, deterministic, no parameter tuning
- **Cons**: Doesn't scale beyond 100K vectors
- **Rationale**: 10K chunks is small enough for exact search; prioritize quality over speed

#### Trade-off 2: Chunk Size
**Choice**: 512 tokens with 20% overlap
- **Pros**: Fits embedding model context, preserves semantic units
- **Cons**: Longer chunks may dilute relevance signal
- **Rationale**: Academic papers have dense technical content; larger chunks provide better context

#### Trade-off 3: LLM Selection
**Choice**: GPT-3.5-turbo over GPT-4
- **Pros**: 10x cheaper, 2x faster, good quality for synthesis tasks
- **Cons**: Lower reasoning capability for complex questions
- **Rationale**: Most queries are factual retrieval, not complex reasoning; cost/latency matter

#### Trade-off 4: Deployment Platform
**Choice**: Streamlit Cloud (free tier)
- **Pros**: Zero-cost hosting, auto-deployment from GitHub, easy setup
- **Cons**: 1GB RAM limit, shared CPU, cold starts
- **Rationale**: Student project; free tier sufficient for demo/portfolio

---

## Future Architecture Improvements

### Phase 1: Quality (Weeks 5-8)
1. **Hybrid Search**: Combine dense (FAISS) + sparse (BM25) retrieval
2. **Reranking**: Add cross-encoder for top-k refinement
3. **Query Expansion**: Generate multiple query variants for better coverage
4. **Citation Extraction**: Parse and link to specific paper sections

### Phase 2: Scalability (Weeks 9-12)
1. **Index Optimization**: Switch to IndexIVFFlat for 100K+ chunks
2. **Caching**: Cache frequent queries (Redis)
3. **Async Processing**: Parallel embedding generation
4. **Batch Endpoints**: Process multiple queries efficiently

### Phase 3: Features (Weeks 13-16)
1. **Conversation Memory**: Multi-turn dialogue with context
2. **Streaming Responses**: Real-time token streaming
3. **Paper Upload**: User-uploaded PDFs
4. **Advanced Filters**: Filter by author, year, paper type

### Phase 4: Production (Optional)
1. **Monitoring**: Prometheus + Grafana dashboards
2. **A/B Testing**: Test different retrieval strategies
3. **User Feedback Loop**: Thumbs up/down for answer quality
4. **API**: REST API for programmatic access

---

## 🛠️ Tech Stack Summary

| Component | Technology | Version | Rationale |
|-----------|-----------|---------|-----------|
| **Embedding (Text)** | all-MiniLM-L6-v2 | - | Fast, accurate, 384-dim |
| **Embedding (Image)** | CLIP ViT-B/32 | - | Text-image alignment |
| **Vector DB** | FAISS | 1.7.4 | Fast exact search |
| **LLM** | GPT-3.5-turbo | - | Cost-effective synthesis |
| **Web Framework** | Streamlit | 1.31.0 | Rapid prototyping |
| **PDF Processing** | PyMuPDF | 1.23.26 | Fast, accurate |
| **Chunking** | LangChain | - | Semantic-aware splitting |
| **Evaluation** | RAGAS | 0.1.9 | RAG-specific metrics |
| **Deployment** | Streamlit Cloud | - | Free, auto-deploy |

---

## File Structure
```
PaperMind/
├── data/
│   ├── papers/              # PDF files (gitignored if > 50MB)
│   ├── processed/
│   │   ├── extracted_text.json
│   │   ├── chunks_recursive_512.json
│   │   ├── embeddings_512.npy
│   │   ├── image_embeddings.npy
│   │   ├── images_metadata.json
│   │   ├── papers_metadata.json
│   │   ├── images/          # ~5,000 PNG files
│   │   ├── faiss_index/
│   │   │   ├── faiss_index.bin
│   │   │   └── chunks_metadata.pkl
│   │   └── faiss_image_index/
│   │       ├── faiss_image_index.bin
│   │       └── images_metadata.pkl
│   └── evaluation/
│       ├── test_queries.json
│       ├── results.json
│       └── ragas_results.json
├── src/
│   ├── download_papers.py
│   ├── download_more_papers.py
│   ├── pdf_extractor.py
│   ├── text_chunker.py
│   ├── extract_images.py
│   ├── generate_embeddings.py
│   ├── generate_image_embeddings.py
│   ├── build_faiss_index.py
│   ├── build_image_faiss_index.py
│   ├── rag_pipeline.py
│   ├── multimodal_rag_pipeline.py
│   ├── web_ui.py
│   ├── evaluate_system.py
│   └── evaluate_with_ragas.py
├── .streamlit/
│   ├── config.toml           # UI settings (safe to commit)
│   └── secrets.toml          # API keys (gitignored)
├── streamlit_app.py          # Entry point
├── requirements.txt
├── .gitignore
├── .env                      # Local API keys (gitignored)
├── README.md
├── DEVLOG.md
├── ARCHITECTURE.md           # This file
└── DEPLOYMENT.md
```

---

## Security Architecture

### Secrets Management
- **Local Development**: `.env` file (gitignored)
- **Production**: Streamlit Cloud Secrets (encrypted)
- **Never committed**: API keys, credentials, tokens

### API Key Rotation
- **Frequency**: Quarterly or after any leak
- **Process**: 
  1. Generate new key in OpenAI dashboard
  2. Update `.env` locally
  3. Update Streamlit Cloud secrets
  4. Revoke old key
  5. Test deployment

### Rate Limiting
- **OpenAI API**: Tier-based (default: 3,500 RPM)
- **Internal**: None (future: add rate limiting for public deployment)

---

## Monitoring & Observability

### Metrics Collected
- Query latency (P50, P95, P99)
- Token usage (prompt + completion)
- Cost per query
- Error rates
- User satisfaction (future)
- Retrieval quality drift (future)

### Logging
- **Level**: INFO (shows all queries + latency)
- **Format**: Timestamp + query + metrics
- **Storage**: Streamlit Cloud logs (7-day retention)

### Alerting (Future)
- High error rate (> 5%)
- High latency (P95 > 10s)
- High cost (> $1/hour)
- API quota exceeded

---

## Success Criteria

### Technical Metrics
- Retrieval Precision@5: > 35% (current: 35%)
- End-to-end Latency: < 5s (current: 3.5s)
- Answer Relevancy (RAGAS): > 0.80 (TBD)
- Faithfulness (RAGAS): > 0.90 (TBD)
- Uptime: > 95% (Streamlit Cloud handles this)

### User Experience
- Works on mobile + desktop
- Images render correctly
- Sources are verifiable (page numbers shown)
- Fast enough for interactive use (< 5s)

### Portfolio Value
- Live demo URL
- Clean GitHub repo with documentation
- Quantitative evaluation results
- Technical depth (multi-modal RAG, RAGAS)
- Production deployment experience

---

*Last Updated: March 15, 2026 (Day 16)*
*Next Review: March 22, 2026 (after corpus expansion)*