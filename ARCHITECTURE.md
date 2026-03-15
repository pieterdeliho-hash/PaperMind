ARCHITECTURE file (technical designs, start week 2, monthly)


# System Architecture

## High-Level Flow
[Diagram - even hand-drawn photo is fine initially]

User Query → Embedding → Vector Search → Context Retrieval → LLM → Answer

## Components

### 1. Document Ingestion Pipeline
- **Input**: PDF files from arxiv
- **Process**: 
  - Text extraction (PyPDF2)
  - Chunking (500 words, 50 word overlap)
  - Metadata extraction (title, authors, date)
- **Output**: JSON with chunks + metadata

**Why these choices:**
- PyPDF2: Lightweight, handles most academic PDFs
- 500 word chunks: Balance between context and precision
- 50 word overlap: Prevents splitting concepts mid-sentence

### 2. Embedding Generation
[Fill in Week 2]

### 3. Vector Database
[Fill in Week 3]

## Trade-offs & Limitations
- Not handling scanned PDFs (would need Tesseract OCR - future work)
- English-only (multilingual would need different embedding model)
- Max 100 papers (memory constraints on laptop)
```

---

## Documentation Timeline

**Week 1-4**: Just update DEVLOG daily + README checklist
**Week 5-8**: Add to ARCHITECTURE as you build components  
**Week 9-12**: Start writing explanatory sections in README
**Week 13-16**: Polish everything, add diagrams, write blog post

**Daily habit** (5 min after coding):
```
1. Update DEVLOG with what you did
2. Note any decisions you made
3. Write down questions/confusions
4. Check off README feature list if applicable