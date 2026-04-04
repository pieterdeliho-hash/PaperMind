# PaperMind System Architecture

## Table of Contents

1. [System Overview](#system-overview)
2. [Data Pipeline](#data-pipeline)
3. [Retrieval System](#retrieval-system)
4. [Generation Pipeline](#generation-pipeline)
5. [Deployment Architecture](#deployment-architecture)
6. [Key Design Decisions](#key-design-decisions)
7. [Performance Characteristics](#performance-characteristics)
8. [Future Architecture Improvements](#future-architecture-improvements)

\---

## System Overview

PaperMind is a production-deployed RAG system that performs semantic search over 258 AI research papers using multimodal retrieval across both text chunks and figure embeddings. The LLM synthesizes retrieved evidence into a coherent answer with automatic source citations.

The core distinction from a standard text-only RAG pipeline is the dual-index architecture: text chunks and paper figures are embedded separately using domain-appropriate models, searched in parallel, and combined into a single context block before generation.

```
User Query
    |
    +---------------------------+
    |                           |
Text Embedding             Image Embedding
(all-MiniLM-L6-v2)        (CLIP ViT-B/32)
    |                           |
FAISS Text Search          FAISS Image Search
11,780 chunks              6,591 figures
    |                           |
Top-5 Text Results         Top-3 Figure References
    |                           |
    +---------------------------+
                |
        Context Assembly
                |
        GPT-3.5-turbo Generation
                |
        Answer + Citations
```

\---

## Data Pipeline

### Stage 1: Document Acquisition

258 papers were downloaded from ArXiv covering the transformer and deep learning research space, spanning publications from 2017 through 2024. Selection criteria prioritised foundational and high-citation works including BERT, GPT variants, Vision Transformers, and efficient attention mechanisms.

```python
python src/download\_papers.py
```

### Stage 2: PDF Processing

Raw PDFs are processed with PyMuPDF to extract both text content and embedded figures.

Text extraction handles UTF-8 encoding with error recovery and pulls paper-level metadata including title, authors, and publication year. Image extraction filters figures by minimum dimensions (100x100 pixels) and saves them as PNG files with positional metadata recording the source paper and page number.

Output:

* `extracted\_text.json` — 17 MB, full text for all 258 papers
* `images/` folder — 6,591 PNG files (not committed to GitHub, over 400 MB)

### Stage 3: Text Chunking

```
Raw Text -> LangChain RecursiveCharacterTextSplitter -> 512-token chunks
```

The recursive splitter respects semantic boundaries by splitting on paragraphs first, then sentences, then words, then characters as a last resort. This produces more coherent chunks than fixed-size splitting.

Parameters:

* Target chunk size: 512 tokens (GPT-3.5 tokenizer)
* Overlap: 102 tokens (20%) for continuity across chunk boundaries
* Average output: 45 chunks per paper

Output: `chunks\_recursive\_512.json` — 23 MB, 11,787 chunks total

### Stage 4: Embedding Generation

**Text Embeddings**

```python
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(chunks)  # shape: (11787, 384)
```

all-MiniLM-L6-v2 was chosen for its balance of speed, model size, and retrieval quality. It processes approximately 70 chunks per second on CPU, has an 80 MB footprint, and produces 384-dimensional vectors that perform well on semantic similarity tasks.

**Image Embeddings**

```python
model = CLIPModel.from\_pretrained("openai/clip-vit-base-patch32")
embeddings = model.encode\_image(images)  # shape: (6591, 512)
```

CLIP ViT-B/32 is trained on aligned text-image pairs, which allows text queries to retrieve semantically relevant figures through cross-modal similarity. This is the key property that enables the system to find architecture diagrams, attention visualisations, and training curves in response to natural language queries.

**Storage Format**

Both embedding matrices are saved as NumPy binary files (.npy). This format is approximately 9 times smaller than JSON for floating-point arrays:

* `embeddings\_512.npy` — 5 MB (equivalent JSON would be \~45 MB)
* `image\_embeddings.npy` — 13 MB (equivalent JSON would be \~120 MB)

### Stage 5: Index Construction

```python
index = faiss.IndexFlatL2(embedding\_dim)
index.add(embeddings)
faiss.write\_index(index, "faiss\_index.bin")
```

IndexFlatL2 performs exact nearest-neighbour search using L2 (Euclidean) distance. It requires no training, guarantees 100% recall, and runs under 10ms for the corpus sizes used here. A more complex index type such as IndexIVFFlat would only be necessary beyond approximately 100,000 vectors.

Output:

* `faiss\_index/` — 20 MB, text search index
* `faiss\_image\_index/` — 15 MB, image search index

\---

## Retrieval System

### Text Retrieval

```python
def retrieve\_text(query: str, k: int = 5) -> List\[Dict]:
    query\_embedding = text\_model.encode(\[query])\[0]
    query\_vector = query\_embedding.astype('float32').reshape(1, -1)

    distances, indices = text\_index.search(query\_vector, k)

    results = \[]
    for dist, idx in zip(distances\[0], indices\[0]):
        chunk = text\_metadata\[idx]
        similarity = 1 / (1 + dist)  # Convert L2 distance to similarity score
        results.append({...})

    return results
```

Search time is under 10ms for 11,787 vectors. The L2 distance is converted to a similarity score in the range (0, 1] for display purposes.

### Image Retrieval

```python
def retrieve\_images(query: str, k: int = 3) -> List\[Dict]:
    inputs = clip\_processor(text=\[query], return\_tensors="pt", padding=True)
    text\_embeds = clip\_model.get\_text\_features(\*\*inputs)
    text\_embeds = F.normalize(text\_embeds, p=2, dim=1)

    distances, indices = image\_index.search(text\_embeds.numpy(), k)
    ...
```

The text query is encoded using CLIP's text encoder and normalised before searching the image index. This cross-modal search is what allows a query like "show me attention visualisations" to retrieve relevant figures even though no text describes those figures in the index.

### Auto-tuned Retrieval Parameters

Rather than using fixed k values for all queries, the system classifies each query and adjusts retrieval parameters accordingly:

```python
def get\_retrieval\_params(query: str) -> tuple:
    # Visual queries: fewer text chunks, more figure references
    visual\_keywords = \['show', 'diagram', 'figure', 'visualiz', 'image', 'graph', 'chart']
    if any(kw in query.lower() for kw in visual\_keywords):
        return (k\_text=4, k\_images=5)

    # Complex or comparative queries: more text sources
    complex\_keywords = \['compare', 'difference', 'vs', 'explain', 'comprehensive', 'analyze']
    if word\_count > 15 or any(kw in query.lower() for kw in complex\_keywords):
        return (k\_text=7, k\_images=3)

    # Default: balanced
    return (k\_text=5, k\_images=3)
```

Advanced Mode in the sidebar allows manual override of these parameters.

\---

## Generation Pipeline

### Context Assembly

Retrieved results are formatted into a structured context block with numbered citations:

```
\[Source 1] attention\_is\_all\_you\_need.pdf | Chunk 14 | Relevance: 0.821
The dominant sequence transduction models are based on complex recurrent...

\[Source 2] bert\_pretraining.pdf | Chunk 3 | Relevance: 0.774
...

\[Figure 1] attention\_is\_all\_you\_need.pdf, Page 3, Score: 0.643
\[Figure 2] vision\_transformer.pdf, Page 7, Score: 0.591
```

### Prompt Design

The system prompt establishes the LLM as a specialist research assistant with access to both text excerpts and figure metadata. The user prompt provides the question, the assembled context, and explicit instructions to synthesise across sources, use 2-4 paragraphs, and cite sources by number.

Temperature is set to 0.3 — low enough for factual accuracy while allowing natural phrasing. Max tokens is set to 800, sufficient for approximately 600 words of output.

### Cost Breakdown

Each query consumes approximately:

* Input: \~2,000 tokens ($0.001 at GPT-3.5-turbo rates)
* Output: \~400 tokens ($0.0006)
* Total: \~$0.006 per query

\---

## Deployment Architecture

### Local Development

```
Windows Machine
├── Python 3.11
├── Virtual environment (venv/)
├── All dependencies installed from requirements.txt
└── .streamlit/secrets.toml (API key, gitignored)
```

### Streamlit Cloud Production

```
Streamlit Cloud (Free Tier)
├── OS: Ubuntu 20.04
├── Python: 3.11.15
├── RAM: 1 GB
├── CPU: Shared
├── Secrets: OPENAI\_API\_KEY via dashboard
└── Auto-deploys on push to main branch
```

### Repository Size Management

The raw project data exceeds 600 MB. The GitHub repository must stay under 100 MB. This was achieved by:

* Not committing PDF files (\~500 MB) — regenerate locally if needed
* Not committing extracted image files (\~400 MB) — regenerate locally if needed
* Not committing JSON embedding files (\~215 MB) — replaced by .npy equivalents
* Committing only pre-generated .npy embeddings and FAISS indexes

Final repository size: 93 MB

Files committed to GitHub:

|File|Size|Purpose|
|-|-|-|
|embeddings\_512.npy|5 MB|Text embeddings|
|image\_embeddings.npy|13 MB|Image embeddings|
|chunks\_recursive\_512.json|23 MB|Text chunks and metadata|
|extracted\_text.json|17 MB|Raw paper text|
|faiss\_index/|20 MB|Text search index|
|faiss\_image\_index/|15 MB|Image search index|

### Cold Start Behaviour

First deployment after a push:

1. Repository clone: \~30 seconds
2. Dependency installation: \~2 minutes
3. FAISS index loading: \~5 seconds
4. Embedding model loading: \~10 seconds
5. OpenAI client initialisation: \~1 second

Total cold start: approximately 3 minutes. Subsequent warm starts complete in under 5 seconds as Streamlit Cloud keeps active apps in memory.

### API Key Loading

The pipeline uses a three-stage fallback for API key resolution, supporting both cloud and local environments without code changes:

```python
def \_load\_api\_key(self):
    # Stage 1: Streamlit secrets (Streamlit Cloud deployment)
    try:
        if "OPENAI\_API\_KEY" in st.secrets:
            self.api\_key = st.secrets\["OPENAI\_API\_KEY"].strip()
            return
    except Exception:
        pass

    # Stage 2: Environment variable (CI/CD or shell export)
    api\_key = os.getenv("OPENAI\_API\_KEY")
    if api\_key:
        self.api\_key = api\_key.strip()
        return

    # Stage 3: .env file (local development fallback)
    load\_dotenv()
    api\_key = os.getenv("OPENAI\_API\_KEY")
    if api\_key:
        self.api\_key = api\_key.strip()
        return

    raise ValueError("OPENAI\_API\_KEY not found in secrets, environment, or .env file")
```

\---

## Key Design Decisions

### FAISS over Cloud Vector Databases

Pinecone and Weaviate offer managed vector search with horizontal scaling, but introduce external API dependencies, monthly costs, and network latency on every query. For a corpus of 258 papers and approximately 18,000 total vectors, FAISS IndexFlatL2 delivers exact search in under 10ms with zero operational overhead. The break-even point where a cloud vector database becomes preferable is around 1 million vectors or when distributed search across multiple machines is required.

### GPT-3.5-turbo over GPT-4o

In a RAG system the retrieval step supplies the factual content. The LLM's role is synthesis and articulation, not knowledge recall. GPT-3.5-turbo handles this at one-tenth the cost ($0.006 vs approximately $0.06 per query) and roughly three times the speed of GPT-4o. The correct investment to improve answer quality is improving retrieval precision through reranking or hybrid search, not upgrading the LLM.

### Pre-generated Embeddings

Streamlit Cloud's free tier cannot sustain the CPU load required to generate 11,787 text embeddings and 6,591 image embeddings at startup. Attempting this caused the account to be blocked for exceeding fair-use CPU limits. The solution is to generate all embeddings locally once, commit the .npy files, and load them at startup. This also reduces startup time from over 10 minutes to under 15 seconds and produces consistent, reproducible embeddings.

### Recursive over Fixed-Size Chunking

Fixed-size chunking splits text at arbitrary character boundaries, frequently cutting sentences or paragraphs mid-way. LangChain's RecursiveCharacterTextSplitter tries paragraph boundaries first, falling back to sentence, word, and character boundaries only when necessary. This produces chunks with more complete semantic units, which improves retrieval coherence. The 20% overlap between chunks ensures that information near chunk boundaries is not lost.

### Text-only Display with Multimodal Retrieval

The system retrieves and scores both text chunks and figures in every query. The LLM references figures in its answers using \[Figure N] citations. However, the extracted figure PNG files are not committed to GitHub due to size constraints, so images cannot be rendered in the UI. This is a deployment constraint rather than an architectural one. The multimodal retrieval capability is fully functional and would support image display if the figures were hosted on external storage such as AWS S3 or Cloudinary.

\---

## Performance Characteristics

### Latency Breakdown

```
Total query time: approximately 3.0 seconds

  Text retrieval (FAISS):     0.01s
  Image retrieval (FAISS):    0.02s
  Context assembly:           0.01s
  LLM generation (OpenAI):    2.96s
```

The LLM API call accounts for 98% of total latency. FAISS search is effectively instantaneous at this corpus size. Reducing latency further would require either caching frequent queries or switching to a faster model.

### Scalability Limits

|Dimension|Current|Estimated Maximum (IndexFlatL2)|
|-|-|-|
|Papers|258|\~1,500|
|Text chunks|11,787|\~100,000|
|Image embeddings|6,591|\~100,000|

Beyond approximately 100,000 vectors, IndexFlatL2 search time grows linearly and IndexIVFFlat should be used instead. IndexIVFFlat uses approximate nearest-neighbour search with clustering, maintaining fast query times at the cost of a small reduction in recall.

\---

## Future Architecture Improvements

### Reranking Layer

Initial retrieval with a bi-encoder (the current approach) is fast but imprecise. A cross-encoder reranker takes the query and each candidate chunk as a pair and produces a more accurate relevance score. The typical pattern is to retrieve k=20 candidates with FAISS, then rerank to the top 5 with a cross-encoder.

Estimated precision improvement: +10-12%

### Hybrid Search

Dense retrieval with sentence embeddings captures semantic similarity but can miss exact keyword matches, particularly for model names, paper titles, and technical terms. BM25 is a classical sparse retrieval method that excels at keyword matching. Combining both with a weighted merge (Reciprocal Rank Fusion is a common approach) captures the strengths of both.

Estimated precision improvement: +8-12%

### Query Expansion

A single query may not cover all relevant phrasings of a topic. Query expansion generates two or three semantically distinct variants of the original query using the LLM, runs retrieval for each, and merges the result sets before reranking.

Estimated recall improvement: +5-10%

### Agentic Retrieval

Rather than using fixed retrieval parameters, an agent-based approach would allow the LLM to decide whether to retrieve more text, more figures, or to issue a follow-up retrieval query based on initial results. This is sometimes called iterative or self-reflective RAG.

\---

Last updated: April 4, 2026
Author: Pieter Deliho
Project: PaperMind Multi-Modal RAG System

