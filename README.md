# PaperMind: AI Research Assistant with Multi-Modal RAG

[!\[Live Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://papermind-ai-assistant.streamlit.app)
[!\[Python 3.11](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org)
[!\[License: MIT](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)

A Retrieval-Augmented Generation (RAG) system for querying 258 AI research papers using semantic search over text and image embeddings.

!\[PaperMind Homepage](docs/screenshots/homepage.png)

\---

## Key Features

### Multi-Modal Retrieval

* **Text Search**: 11,780 text chunks embedded with sentence-transformers (all-MiniLM-L6-v2, 384-dim)
* **Image Search**: 6,591 figure embeddings using CLIP (ViT-B/32, 512-dim)
* **Hybrid Intelligence**: LLM synthesizes both text evidence and figure references into a single answer

### Performance

* **Corpus**: 258 AI research papers sourced from ArXiv
* **Retrieval Speed**: Under 100ms via FAISS vector search
* **Answer Generation**: Approximately 3 seconds total latency
* **Cost Efficient**: $0.006 per query using GPT-3.5-turbo

### Smart Features

* **Auto-tuned Retrieval**: Adjusts k\_text and k\_images based on query type and complexity
* **Citation Tracking**: Automatic source attribution with \[Source N] and \[Figure N] references
* **Advanced Mode**: Manual control over retrieval parameters via sidebar
* **System Metrics**: Live display of token usage, latency, and cost per query

\---

## Live Demo

**Try it now:** [papermind-ai-assistant.streamlit.app](https://papermind-ai-assistant.streamlit.app)

Example queries to get started:

* "What are transformer architectures?"
* "Explain multi-head attention mechanisms"
* "Compare vision transformers to standard transformers"
* "How do efficient transformers handle long sequences?"

\---

## Screenshots

### Query and Answer

!\[Query and Answer](docs/screenshots/query\_answer.png)

### Text Sources

!\[Sources View](docs/screenshots/sources.png)

!\[Sources View](docs/screenshots/sources2.png)

### Sidebar and System Metrics

!\[Sidebar](docs/screenshots/sidebar.png)

\---

## Architecture

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

## Technical Details

|Component|Technology|Specification|
|-|-|-|
|Text Embeddings|sentence-transformers|all-MiniLM-L6-v2 (384-dim)|
|Image Embeddings|CLIP|ViT-B/32 (512-dim)|
|Vector Database|FAISS|IndexFlatL2, 11,787 + 6,591 vectors|
|LLM|OpenAI|GPT-3.5-turbo|
|Chunking|LangChain|Recursive, 512 tokens, 20% overlap|
|Framework|Streamlit|1.31.0|
|Deployment|Streamlit Cloud|Free tier|

\---

## Performance Metrics

|Metric|Value|
|-|-|
|Papers Indexed|258|
|Text Chunks|11,787|
|Image Embeddings|6,591|
|Average Query Latency|\~3.0s|
|Retrieval Precision|45-50% (estimated)|
|Cost per Query|$0.006|
|FAISS Search Time|Under 100ms|

\---

## Local Setup

**Prerequisites:**

* Python 3.11 or higher
* OpenAI API key

**Installation:**

```bash
# Clone the repository
git clone https://github.com/pieterdeliho-hash/PaperMind.git
cd PaperMind

# Create and activate virtual environment
python -m venv venv
venv\\Scripts\\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

**API Key Setup:**

Create `.streamlit/secrets.toml` with your OpenAI API key:

```toml
OPENAI\_API\_KEY = "key-here"
```

**Run the application:**

```bash
streamlit run streamlit\_app.py
```

\---

## Project Structure

```
PaperMind/
├── src/
│   ├── multimodal\_rag\_pipeline.py   # Core RAG system
│   ├── web\_ui.py                    # Streamlit interface
│   ├── pdf\_extractor.py             # PDF processing
│   ├── text\_chunker.py              # Text chunking
│   ├── generate\_embeddings.py       # Text embedding generation
│   ├── generate\_image\_embeddings.py # Image embedding generation
│   ├── build\_faiss\_index.py         # FAISS text index builder
│   └── build\_image\_faiss\_index.py   # FAISS image index builder
├── data/processed/
│   ├── embeddings\_512.npy           # Pre-generated text embeddings (5 MB)
│   ├── image\_embeddings.npy         # Pre-generated image embeddings (13 MB)
│   ├── chunks\_recursive\_512.json    # Text chunks with metadata (23 MB)
│   ├── extracted\_text.json          # Raw extracted paper text (17 MB)
│   ├── faiss\_index/                 # FAISS text search index (20 MB)
│   └── faiss\_image\_index/           # FAISS image search index (15 MB)
├── docs/
│   └── screenshots/                 # UI screenshots for documentation
├── streamlit\_app.py                 # Application entry point
├── requirements.txt                 # Pinned Python dependencies
├── ARCHITECTURE.md                  # System architecture deep-dive
├── DEVLOG.md                        # Development log and journey
└── .gitignore                       # Excludes PDFs, images, large files
```

\---

## Design Decisions

**FAISS over cloud vector databases (Pinecone, Weaviate)**
FAISS runs locally with no external dependencies, no API costs, and performs exact search in under 100ms for the corpus size used here. At 258 papers and roughly 18,000 total vectors, IndexFlatL2 is more than sufficient. A cloud vector database would add latency, cost, and operational complexity without meaningful benefit at this scale.

**GPT-3.5-turbo over GPT-4o**
In a RAG system, the retrieval step provides the facts — the LLM only needs to synthesize and articulate them. GPT-3.5-turbo handles this well at one-tenth the cost and roughly three times the speed of GPT-4o. Upgrading the retrieval quality (reranking, hybrid search) would yield more improvement than upgrading the LLM.

**Pre-generated embeddings over on-demand generation**
Streamlit Cloud's free tier does not support the CPU load required to generate embeddings at startup. All embeddings are generated locally and committed to GitHub as .npy files, reducing the repository from 606 MB to 93 MB and reducing startup time to under 5 seconds.

**Text-only UI with multimodal retrieval**
The system retrieves and scores both text chunks and figures, and the LLM references figures in its answers. Images are not rendered in the UI because the extracted figure files are not committed to GitHub (they account for over 400 MB). This keeps the deployment lean while preserving the multimodal retrieval capability.

\---

## Future Improvements

* **Reranking**: Cross-encoder reranking after initial retrieval for improved precision (+10-12% estimated)
* **Hybrid Search**: Combine BM25 keyword search with dense retrieval for better recall
* **Query Expansion**: Generate multiple query variants and merge results
* **GPT-4o Upgrade**: Higher answer quality for complex multi-paper synthesis
* **Corpus Expansion**: Scale to 1,000+ papers with IndexIVFFlat for faster search
* **Image Display**: Serve extracted figures from cloud storage and render them in the UI

\---

## Educational Value

This project demonstrates practical implementation of:

* Multi-modal machine learning with separate text and image embedding models
* Vector database design and semantic search
* Full RAG architecture from document ingestion to answer generation
* Production deployment with cloud hosting and API key management
* Performance optimization through embedding compression and pre-computation
* Cross-platform Python development (Windows local, Linux cloud)

\---

## Contact

**Pieter Deliho**
Year 2 CS/AI Student, Asia Pacific University

* Email: pieterdeliho@gmail.com
* GitHub: [pieterdeliho-hash](https://github.com/pieterdeliho-hash)
* LinkedIn: [pieter-deli-ho-843216332](https://www.linkedin.com/in/pieter-deli-ho-843216332)

\---

## License

MIT License. See LICENSE file for details.

\---

Built with Python, FAISS, OpenAI, Streamlit, sentence-transformers, and CLIP.

