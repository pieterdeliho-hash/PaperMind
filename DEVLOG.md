(Do after every coding session)

\# Development Log - PaperMind



\*\*Project Timeline\*\*: March 4 - April 5, 2025 (1-month sprint)  

\*\*Goal\*\*: Production-ready RAG system for CV/internship applications  

\*\*Target Completion\*\*: April 1 (5 days buffer for polish)



---



\## Sprint Plan



\*\*Week 1 (March 4-10)\*\*: Core RAG Pipeline 

\*\*Week 2 (March 11-17)\*\*: Multi-Modal + Evaluation  

\*\*Week 3 (March 18-24)\*\*: Production UI + Features  

\*\*Week 4 (March 25-31)\*\*: Documentation + Demo  

\*\*April 1-5\*\*: CV Integration + Interview Prep



---



\## Progress Tracker



\- \[x] Day 1: Setup (1 hr)

\- \[x] Day 2: Papers (45 min)

\- \[x] Day 3: Chunking (1.5 hr)

\- \[x] Day 4: Embeddings (1 hr)

\- \[ ] Day 5: FAISS (45 min)

\- \[ ] Day 6: LLM (1.5 hr)

\- \[ ] Day 7: Pipeline (1 hr)



\*\*Total so far\*\*: 4.75 hours  

\*\*Pace\*\*: Ahead of schedule



\# Development Log - PaperMind



\## Week 1: Foundation Setup



\### Day 1 - March 4, 2025

\*\*Time spent\*\*: 1 hour



\*\*What I did:\*\*

\- Set up Python 3.11 environment

\- Created project structure (data/, src/, notebooks/)

\- Initialized git repository

\- Installed initial libraries (PyPDF2, pdfplumber)

\- Wrote basic PDF extraction module



\*\*Decisions made:\*\*

\- Using PyPDF2 as primary extraction library (lightweight, good for text PDFs)

\- Added pdfplumber as backup for complex layouts

\- Storing extracted text as JSON for easy inspection

\- Separate folders for raw papers vs processed data



\*\*Challenges:\*\*

\- None yet! Setup went smoothly



\*\*Questions/TODO:\*\*

\- Need to download sample papers tomorrow

\- Should I use arxiv API or manual download?

\- How many papers to start with? (thinking 10-20)



\*\*Next session:\*\*

\- Download 10 research papers from arxiv

\- Test extraction on real PDFs

\- Handle any encoding/format issues



\*\*Mood\*\*: Excited! Good start.



\### Day 1 Continued - Planning for Dataset



\*\*Paper selection criteria:\*\*

\- Focus: Transformer architectures and NLP (want to learn more about this)

\- Source: ArXiv (free, abundant, easy API access)

\- Quantity: Starting with 10-20 papers, can scale to 100+

\- Storage: Allocated 50GB (more than enough - likely use <1GB)



\*\*Why transformers:\*\*

\- Core to modern LLM architecture

\- Want deeper understanding beyond surface level

\- Highly relevant for RAG system I'm building

\- Papers will actually be useful for my studies



\*\*Dataset decisions:\*\*

\- Will use arxiv API for automated download

\- Filter for papers with "transformer" or "attention" keywords

\- Prioritize highly-cited papers (quality signal)

\- Mix of foundational papers (2017-2020) + recent advances (2023-2024)



\*\*Next:\*\* Implement arxiv paper downloader



\### Day 2 - March 5, 2025

\*\*Time spent\*\*: 45 minutes



\*\*What I did:\*\*

\- Installed arxiv library (arxiv==2.1.0)

\- Built automated paper downloader with ArXiv API

\- Downloaded 14 papers (10 from search + 4 landmark papers)

\- Ran text extraction on all papers

\- Created quality check script to verify extraction



\*\*Papers downloaded:\*\*

Search-based (10):

\- "transformer attention mechanism" query

\- Top relevance-ranked papers



Specific landmark papers (4):

\- "Attention Is All You Need" (1706.03762) - Original Transformer

\- "BERT" (1810.04805) - Bidirectional transformers

\- "GPT-3" (2005.14165) - Language model scaling

\- "Vision Transformer (ViT)" (2010.11929) - Transformers for images



\*\*Extraction results:\*\*

\- Total papers: 14

\- Successfully extracted: 14/14 ✓

\- Good quality (>5000 chars): 14/14 ✓

\- Issues: NONE - all papers extracted perfectly!



\*\*Sample extraction quality:\*\*

Vision Transformer paper extracted with:

\- Clean text formatting

\- Proper mathematical symbols (×, •)

\- Intact citations and references

\- Abstract, introduction, sections all readable

\- ~50,000+ characters of quality content per paper



\*\*Decisions made:\*\*

\- Using ArXiv API with automated search (efficient, reproducible)

\- Added both search-based AND specific paper downloads (flexibility)

\- Included 4 foundational papers manually (Attention, BERT, GPT-3, ViT)

  - Reasoning: These are must-know papers for understanding transformers

\- 3-second delay between downloads (respectful to ArXiv servers)

\- Quality threshold confirmed: >5000 chars works well

\- Keeping extracted text as JSON (easy to inspect and debug)



\*\*Challenges:\*\*

\- None! Download and extraction went smoother than expected

\- PyPDF2 handled all academic papers without issues

\- No need for pdfplumber fallback (yet)



\*\*Learnings:\*\*

\- ArXiv API is incredibly simple and powerful

\- Academic PDFs (text-based) extract much better than I anticipated

\- Metadata from ArXiv is rich (authors, dates, summaries, IDs)

\- Having both automated search + manual curation is the right approach

\- 14 papers is a good starting corpus for Week 1 testing



\*\*Technical observations:\*\*

\- Average paper: ~30,000-60,000 characters of text

\- Vision Transformer paper: excellent test case (equations, figures references, complex layout)

\- PDF filenames cleaned automatically (removed special chars, limited length)

\- Metadata saved separately - will be useful for citation tracking later



\*\*Questions/TODO:\*\*

\- What's the optimal chunk size for these papers?

  - Initial thought: 500-1000 words with 50-100 word overlap

  - Need to test different strategies

\- Should I download more papers now or wait?

  - Decision: 14 is sufficient for Week 1-2 development

  - Will expand to 50+ when retrieval is working

\- How to handle mathematical equations in text?

  - Noted: Some equations extracted as Unicode symbols

  - Decision: Good enough for now, evaluate impact later



\*\*Next session:\*\*

\- Implement text chunking strategies

\- Test different chunk sizes (sentences, paragraphs, fixed-length)

\- Research semantic chunking vs fixed-size chunking

\- Create chunking quality inspector



\*\*Mood\*\*: Fired up! Zero issues, perfect extractions, ahead of schedule!



\#### Day 3 - March 5, 2025 (Revised - Production Implementation)

\*\*Time spent\*\*: 1.5 hours



\*\*What I did:\*\*

\- Researched industry-standard chunking approaches

\- Found recursive character splitting is production standard

\- Installed tiktoken (OpenAI tokenizer) and langchain-text-splitters

\- Implemented token-aware recursive character splitting

\- Tested two configurations (512 and 1024 tokens)

\- Created production-ready chunking module



\*\*Research findings:\*\*

\- Industry standard: Recursive character splitting

\- Optimal chunk size: 512 tokens (not words!)

\- Optimal overlap: 10-20% (I chose 20% = 102 tokens)

\- Token-based measurement > word-based (more accurate)

\- Source: LangChain documentation, RAG best practices papers



\*\*Chunking results:\*\*

Config 1 - Standard (512 tokens, 20% overlap):

\- Total chunks: 931

\- Avg per paper: 66.5

\- Avg tokens: 481.6

\- Avg characters: 1519



Config 2 - Large (1024 tokens, 20% overlap):

\- Total chunks: 470

\- Avg per paper: 33.6

\- Avg tokens: 965.8

\- Avg characters: 3039



\*\*Critical decision: Why Recursive Character Splitting?\*\*



Initial approach: Simple sentence-based splitting (500 words)

Research showed: This is NOT industry standard



Switched to: Recursive character splitting with token measurement



\*\*Reasoning:\*\*

1\. \*\*Token-based > Word-based\*\*

   - LLMs work in tokens, not words

   - "Token" varies by word (encoding matters)

   - Accurate measurement prevents exceeding LLM limits

   - 1 token ≈ 0.75 words ≈ 4 characters (approximate)

 

2\. \*\*Recursive = Respects Structure\*\*

   - Tries to split at paragraph breaks first (\\n\\n)

   - Falls back to sentences (.), then words, then characters

   - Preserves semantic coherence better than fixed splitting

   - Academic papers benefit from structure-aware chunking

 

3\. \*\*20% Overlap is Proven Optimal\*\*

   - Research shows 10-20% prevents context loss at chunk boundaries

   - I chose 20% (conservative, more context preserved)

   - 102 tokens overlap for 512 token chunks

   - Ensures concepts spanning chunks aren't split

 

4\. \*\*Production-Ready\*\*

   - Same approach used in real RAG systems (LangChain, LlamaIndex)

   - Will integrate easily with LangChain later

   - Better for CV: "industry-standard methods" vs "custom approach"

   - Demonstrates understanding of production best practices

 

5\. \*\*Hardware Not a Constraint\*\*

   - Legion laptop easily handles token encoding

   - Tiktoken is extremely fast (~1M tokens/sec)

   - No reason to use suboptimal approach for performance



\*\*Trade-offs considered:\*\*

\- Simple sentence splitting: Easier to understand, faster to implement

  - But: Not optimal for retrieval quality, word-based inaccurate

\- Recursive splitting: Slightly more complex, requires tokenizer

  - But: Better results, industry standard, worth the complexity



\*\*Why 512 tokens specifically?\*\*

\- Balances context vs. precision

\- Small enough for focused retrieval (specific facts)

\- Large enough to contain complete thoughts (paragraphs)

\- Standard in RAG literature and production systems

\- Leaves room for query tokens in LLM context window

\- Can test 1024 later if retrieval lacks context



\*\*Implementation details:\*\*

\- Using cl100k\_base tokenizer (GPT-3.5/4 standard)

\- Separator hierarchy: \\n\\n → \\n → . → ! → ? → ; → , → space

\- Exact token counting (not estimation)

\- Metadata tracking: tokens, chars, position in paper

\- LangChain's RecursiveCharacterTextSplitter



\*\*Results analysis:\*\*

\- 931 chunks from 14 papers = excellent corpus size

\- Avg 481.6 tokens = good targeting (under 512 limit)

\- Avg 66.5 chunks/paper = granular retrieval possible

\- Token variance is minimal (good consistency)



\*\*Challenges:\*\*

\- Initial confusion about tokens vs words vs characters

  - Learned: Must use proper tokenizer for accuracy

  - Word count is approximate, tokens are exact

\- Deciding between 512 and 1024 tokens

  - Testing both, will evaluate during retrieval phase

  - Starting with 512 (industry standard)



\*\*Learnings:\*\*

\- Critical thinking: Don't accept first solution, research best practices

\- Token-based measurement is crucial for LLM applications

\- Overlap percentage more important than absolute number

\- Production systems use recursive splitting for good reason

\- Academic papers benefit from structure-aware splitting

\- Research > implementation speed for quality



\*\*Chunk quality verification:\*\*

\- Inspected random samples using inspect\_chunks\_v2.py

\- Results: Excellent coherence across content types

  - References: Clean citation boundaries

  - Tables: Data structures preserved

  - Algorithms: Pseudocode intact

\- No mid-sentence cuts observed

\- Token variance minimal (494-498 range)

\- Ready for embedding generation



\*\*Next session:\*\*

\- Generate embeddings with sentence-transformers

\- Research which embedding model to use

\- Set up FAISS vector database

\- Implement similarity search

\- Test retrieval quality on sample queries



\*\*Mood\*\*: Leveled up! Learned the difference between "works" and "works well"



\### Day 4 - March 5, 2025 (Same day as Day 3!)

\*\*Time spent\*\*: 1 hour



\*\*What I did:\*\*

\- Installed sentence-transformers and faiss-cpu

\- Chose all-MiniLM-L6-v2 embedding model (384 dimensions)

\- Generated embeddings for all 931 chunks

\- Created embedding quality test with semantic similarity

\- Verified retrieval quality across multiple query types



\*\*Embedding generation results:\*\*

\- Model: all-MiniLM-L6-v2 (80MB, 384 dimensions)

\- Total chunks embedded: 931

\- Processing time: 16.89 seconds

\- Speed: 55.1 chunks/second

\- Output files:

  - JSON: 11.60 MB (includes metadata)

  - NPY: 1.36 MB (just vectors, more efficient)



\*\*Model selection reasoning:\*\*

\- Chose all-MiniLM-L6-v2 over alternatives:

  - all-mpnet-base-v2: Better quality but 768 dim (overkill for 931 chunks)

  - OpenAI ada-002: Best quality but costs money (API calls)

  - MiniLM: Free, local, fast, 384 dim = smaller index

 

\- Why 384 dimensions is sufficient:

  - Research shows 384 dim captures semantic meaning well

  - Smaller dimensions = faster search, less memory

  - Can always upgrade to mpnet later if quality insufficient

  - Good balance for academic papers (technical but structured)



\*\*Semantic similarity test results:\*\*



Query 1: "What is the attention mechanism?"

\- Top score: 0.5632 (EXCELLENT)

\- Retrieved: "Attention Is All You Need", attention mechanism papers

\- Analysis: Strong semantic match, correct papers found



Query 2: "How do transformers work?"

\- Top score: 0.4473 (GOOD)

\- Retrieved: Vision Transformer, transformer architecture papers

\- Analysis: Broader query, still highly relevant results



Query 3: "What datasets were used?"

\- Top score: 0.3694 (ACCEPTABLE)

\- Retrieved: Dataset sections from GPT-3, XING, UserBehavior papers

\- Analysis: Specific query, found correct content despite lower score



Query 4: "Image classification accuracy"

\- Top score: 0.5192 (VERY GOOD)

\- Retrieved: Accuracy tables, ImageNet results, metrics

\- Analysis: Perfect retrieval for quantitative queries



\*\*Quality verdict: EXCELLENT\*\* 

\- All queries retrieved semantically relevant chunks

\- Score distribution appropriate (0.35-0.56 range)

\- Different queries → different results (good discrimination)

\- High scores (>0.5) for direct matches

\- Lower scores (0.3-0.4) for broader queries (expected)



\*\*Technical observations:\*\*

\- Embedding shape: (931, 384) - each chunk → 384-dim vector

\- Cosine similarity working correctly

\- Model loaded once, cached for subsequent runs

\- NPY format 8x smaller than JSON (1.36 MB vs 11.60 MB)

\- HuggingFace cache warning (symlinks) - not critical, just storage inefficiency



\*\*Decisions made:\*\*

\- Using cosine similarity for vector comparison (standard for embeddings)

\- Saving both JSON (human-readable) and NPY (efficient) formats

\- Batch size 32 (good balance for Legion laptop)

\- Not enabling performance mode - current speed sufficient



\*\*Challenges:\*\*

\- Initial model download took extra time (6 seconds)

\- First run slower than expected (55 vs 200+ chunks/sec)

  - Reason: Not in performance mode, model initialization overhead

  - Decision: Acceptable, one-time cost, queries will be instant

\- Understanding score interpretation

  - Learned: 0.3-0.7 is normal range for semantic similarity

  - Perfect 1.0 scores only for identical text



\*\*Learnings:\*\*

\- Embeddings convert semantic meaning into math

\- Similar meanings → similar vectors in 384D space

\- Cosine similarity measures angle between vectors (0-1 scale)

\- Lower dimensions don't mean worse quality (MiniLM well-optimized)

\- Batch processing significantly faster than one-at-a-time

\- NPY format crucial for production (8x space savings)



\*\*What embeddings enable:\*\*

\- Mathematical similarity search (not keyword matching)

\- "Attention mechanism" matches "self-attention", "multi-head attention"

\- Retrieves semantically related content, not just exact words

\- Foundation of RAG: find relevant context for LLM



\*\*Next session:\*\*

\- Set up FAISS vector database for fast similarity search

\- Index all 931 embeddings

\- Implement k-nearest neighbors retrieval

\- Compare FAISS speed vs naive cosine similarity

\- Build query interface



\*\*Mood\*\*: RAG is coming alive! Embeddings working beautifully.



\### Day 5 - March 5, 2025 (Same day as Day 4!)

\*\*Time spent\*\*: 45 minutes



\*\*What I did:\*\*

\- Installed FAISS for production vector search

\- Built FAISS IndexFlatL2 index for 931 embeddings

\- Tested search speed (FAISS vs naive cosine similarity)

\- Created search engine interface

\- Implemented interactive query system



\*\*FAISS indexing results:\*\*

\- Index type: IndexFlatL2 (exact search, L2 distance)

\- Build time: 0.003 seconds (3 milliseconds!)

\- Total vectors: 931

\- Files created:

&nbsp; - faiss\_index.bin (FAISS index)

&nbsp; - chunks\_metadata.pkl (chunk information)

&nbsp; - index\_info.json (index metadata)



\*\*Speed comparison results:\*\*

\- Naive cosine similarity: ~2-5 ms per query

\- FAISS search: ~0.2-0.5 ms per query

\- Speedup: 5-20x faster (even with just 931 chunks)

\- Projected speedup for 10k chunks: 100-500x faster



\*\*Decisions made:\*\*

\- Using IndexFlatL2 (exact search) over IVF (approximate)

&nbsp; - Reasoning: 931 chunks is small, exact search has negligible overhead

&nbsp; - Quality: Perfect recall (no approximation errors)

&nbsp; - Speed: Still sub-millisecond queries

&nbsp; - Future: Can switch to IVF when scaling to 10k+ chunks

&nbsp; 

\- L2 distance vs cosine similarity:

&nbsp; - FAISS uses L2 (Euclidean) distance by default

&nbsp; - For normalized embeddings, L2 ≈ cosine similarity

&nbsp; - Converting distance to similarity: score = 1 / (1 + distance)

&nbsp; - Verification: Results match naive cosine similarity rankings

&nbsp; 

\- Saving metadata separately (pickle):

&nbsp; - FAISS only stores vectors, not chunk text/metadata

&nbsp; - Metadata (chunk text, paper names) saved as pickle file

&nbsp; - Fast loading (~10ms), efficient storage



\*\*Search engine features:\*\*

\- Query embedding in real-time

\- Top-k retrieval (default k=5)

\- Score display (similarity + L2 distance)

\- Preview of retrieved chunks

\- Interactive search mode



\*\*Test query results:\*\*



Query: "What is the attention mechanism?"

\- Top score: 0.8523 (converted from L2 distance)

\- Retrieved: Attention papers, transformer architecture

\- Latency: <1ms



Query: "How do vision transformers work?"

\- Top score: 0.7891

\- Retrieved: ViT paper chunks, image processing sections

\- Latency: <1ms



Query: "What datasets were used?"

\- Top score: 0.6234

\- Retrieved: Dataset sections from multiple papers

\- Latency: <1ms



\*\*Quality observations:\*\*

\- FAISS returns identical rankings to naive search ✓

\- Sub-millisecond query latency ✓

\- Retrieval quality maintained (exact search) ✓

\- Interactive search is responsive and smooth ✓



\*\*Challenges:\*\*

\- Understanding L2 vs cosine distance

&nbsp; - Learned: For unit-normalized vectors, they're equivalent

&nbsp; - L2 distance² = 2(1 - cosine\_similarity)

\- Score interpretation

&nbsp; - L2 distance: lower = more similar

&nbsp; - Converted to 0-1 similarity for consistency with Day 4 tests

\- Metadata management

&nbsp; - FAISS doesn't store metadata, need separate file

&nbsp; - Pickle chosen for speed (vs JSON)



\*\*Learnings:\*\*

\- FAISS is production-standard for vector search

\- Even small indexes benefit from FAISS optimization

\- IndexFlatL2 perfect for <10k vectors

\- Can scale to millions with IVF/HNSW indexes later

\- Search speed critical for user experience (<100ms target)



\*\*Technical details:\*\*

\- Index built in RAM, saved to disk

\- Binary index file (~1.4 MB for 931 vectors)

\- Loading index: ~5ms from disk

\- Memory usage: Minimal (~3 MB loaded)



\*\*What FAISS enables:\*\*

\- Production-ready search speed

\- Scalability to 10k+ chunks without code changes

\- Multiple concurrent queries (future API)

\- Foundation for advanced features (filtering, re-ranking)



\*\*Interactive search testing:\*\*

\- Tested with queries: attention mechanism, vision transformers, datasets

\- Response time: Instant (<10ms perceived latency)

\- Retrieval quality: Excellent semantic matching

\- Key papers (Attention Is All You Need, ViT) retrieved correctly

\- System ready for LLM integration



\*\*Next session:\*\*

\- Integrate OpenAI API for answer generation

\- Build end-to-end RAG pipeline (retrieval + generation)

\- Test complete question-answering flow

\- Add citation tracking



\*\*Mood\*\*: Lightning fast! Search is instant, retrieval working perfectly.



\### Day 6 - March 5, 2025 (Same day as Day 5!)

\*\*Time spent\*\*: 1 hour



\*\*What I did:\*\*

\- Set up OpenAI API integration

\- Built complete RAG pipeline (retrieval + generation)

\- Implemented citation tracking

\- Tested end-to-end question answering

\- Created interactive query interface



\*\*OpenAI setup:\*\*

\- Created API key with usage limits

\- Added $5 credit (more than enough for project)

\- Secured key in .env file (gitignored)

\- Using GPT-3.5-turbo (fast, cheap, good quality)



\*\*RAG pipeline results:\*\*



Query 1: "What is the attention mechanism in transformers?"

\- Retrieved 3 chunks in <10ms

\- Generated answer in 2.43s

\- Tokens used: 1732 (~$0.003)

\- Quality: Excellent - mentioned relative positional attention, sparse attention, Neighborhood Attention

\- Citations: \[Source 1], \[Source 2], \[Source 3] working correctly



Query 2: "How do vision transformers process images?"

\- Retrieved ViT paper chunks

\- Answer explains: patch projection, position embeddings, self-attention

\- Latency: 2.34s

\- Tokens: 1721

\- Quality: Very good - technical details accurate



Query 3: "What are the key advantages of transformers over RNNs?"

\- Cross-paper synthesis (combined multiple papers)

\- Mentioned: linear complexity, compute efficiency

\- Latency: 2.26s

\- Quality: Good - comparative analysis



Interactive testing:

\- Query: "What is the attention mechanism?"

&nbsp; - Retrieved "Attention Is All You Need" paper ✓

&nbsp; - Coherent explanation with focus weights concept

&nbsp; - 1.48s response time

&nbsp; 

\- Query: "What is transformers?"

&nbsp; - Explained architecture, fixed dimensionality, vision applications

&nbsp; - 1.33s response time (fastest!)

&nbsp; - Good beginner-friendly explanation



\*\*Technical implementation:\*\*



System prompt design:

```

You are an AI research assistant. Answer questions based on provided research paper excerpts.

\- Answer directly and concisely

\- Cite sources using \[Source N] notation

\- Say "I don't have enough information" if uncertain

\- Academic tone

```



Context formatting:

\- Top-k chunks (k=3 for testing, k=5 for production)

\- Separator: "---" between chunks

\- Format: \[Source N: paper\_name] chunk\_text

\- Max context: ~2000-2500 tokens (fits in GPT-3.5 context)



LLM parameters:

\- Model: gpt-3.5-turbo

\- Temperature: 0.3 (low = deterministic, factual)

\- Max tokens: 500 (answer length limit)

\- Total context window: ~4000 tokens (safe for GPT-3.5)



\*\*Decisions made:\*\*



\- GPT-3.5-turbo over GPT-4:

&nbsp; - Reasoning: 10x cheaper, faster, quality sufficient for demo

&nbsp; - Cost: $0.002 per 1K tokens vs $0.02 for GPT-4

&nbsp; - Can upgrade later if needed

&nbsp; - Trade-off: Slightly less nuanced answers, acceptable



\- Temperature 0.3:

&nbsp; - Low enough for factual consistency

&nbsp; - High enough to avoid robotic responses

&nbsp; - Sweet spot for technical Q\&A



\- k=3 retrieval:

&nbsp; - 3 chunks = ~1500 tokens context

&nbsp; - Leaves room for query + answer in 4K window

&nbsp; - Reduces noise from less-relevant chunks

&nbsp; - Can increase to k=5 for complex queries



\- Citation format \[Source N]:

&nbsp; - Simple, clear reference system

&nbsp; - Maps directly to source list

&nbsp; - Easy for users to verify

&nbsp; - Professional presentation



\*\*Quality observations:\*\*



Strengths:

✓ Retrieval finds highly relevant chunks (scores 0.47-0.53)

✓ Answers are coherent and well-structured

✓ Citations used appropriately

✓ Cross-paper synthesis works (combines multiple sources)

✓ Technical accuracy verified against papers

✓ Handles different query types (what, how, advantages)



Areas for improvement:

\- Latency: 2-2.5s is acceptable but could be faster

&nbsp; - LLM call is bottleneck (expected)

&nbsp; - Could use gpt-3.5-turbo-16k for faster model

\- Sometimes retrieves similar chunks from same paper

&nbsp; - Could add diversity penalty in retrieval

\- Citations could include page numbers/sections

&nbsp; - Future: Extract metadata from chunks



\*\*Cost analysis:\*\*

\- Demo queries (3): ~5100 tokens = $0.01

\- Interactive (2): ~3000 tokens = $0.01

\- Total so far: ~$0.02

\- Projected project cost: $0.50-$2.00 (very affordable!)



\*\*Challenges:\*\*



\- Initial API key setup

&nbsp; - Learned: Need payment method even for small usage

&nbsp; - Added $5 credit (overkill, but safe)

&nbsp; 

\- Prompt engineering

&nbsp; - First attempt: Answers too verbose

&nbsp; - Solution: Added "concisely" to system prompt

&nbsp; - Result: Much better length control



\- Understanding token usage

&nbsp; - Learned: 1 token ≈ 0.75 words

&nbsp; - Context uses ~1500-2000 tokens

&nbsp; - Answer uses ~200-500 tokens

&nbsp; - Total: 1700-2500 tokens per query



\*\*Learnings:\*\*



\- RAG transforms retrieval into natural language answers

\- Citation tracking essential for trust and verification

\- Temperature tuning impacts answer consistency

\- Context window management is critical

\- GPT-3.5 is sufficient for technical Q\&A (don't need GPT-4)

\- Prompt engineering makes huge quality difference

\- Token counting important for cost control



\*\*What RAG enables:\*\*

\- Natural language interaction with research papers

\- Multi-paper synthesis (combines knowledge)

\- Cited answers (verifiable information)

\- Scalable to any domain (just swap papers)

\- Cost-effective ($0.002 per query)



\*\*End-to-end pipeline performance:\*\*

1\. Query embedding: ~10ms

2\. FAISS search: ~1ms

3\. Context assembly: <1ms

4\. LLM generation: ~2000ms

5\. \*\*Total latency: ~2010ms\*\* (acceptable for demo)



\*\*Next session:\*\*

\- Download more papers (expand to 30-50)

\- Re-index with larger corpus

\- Test retrieval quality at scale

\- Build web interface (Streamlit)

\- Add response streaming for better UX



\*\*Milestone achieved:\*\* 

\*\*WEEK 1 COMPLETE!\*\* Full RAG pipeline working end-to-end.

\- Days 1-6 done in 1 day

\- Ahead of schedule by 6 days

\- All core functionality implemented

\- Ready for Week 2 (multi-modal + polish)



\*\*Mood\*\*: INCREDIBLE! Built a production RAG system in one day!



\### Corpus Expansion - March 5, 2026

\*\*Time spent\*\*: 30 minutes



\*\*What I did:\*\*

\- Downloaded 43 additional papers across diverse topics

\- Re-ran extraction, chunking, embedding, and indexing pipeline

\- Tested retrieval quality with expanded corpus

\- Verified improvement in answer quality and diversity



\*\*Download strategy:\*\*

\- 14 search queries covering:

&nbsp; - Core architectures (Transformer, BERT, GPT)

&nbsp; - Vision transformers (ViT variants)

&nbsp; - Attention mechanisms (self, multi-head, cross)

&nbsp; - Applications (NLP, CV, time series)

&nbsp; - Recent advances (efficient, sparse, linear)

&nbsp; 

\- Papers per query: 3-5 (strategic selection)

\- Skipped duplicates automatically

\- Total new papers: 43

\- Download errors: 0



\*\*Results after expansion:\*\*

\- Total papers: 57

\- Total chunks: 3,431

\- Embedding generation time: 62 seconds

\- FAISS build time: 0.004 seconds

\- Index size: 5.2 MB



\*\*Quality improvements observed:\*\*



Before (14 papers):

\- Total chunks: 931

\- Source diversity: 1-2 papers per query

\- Coverage gaps: Time series, cross-attention, efficient transformers

\- Relevance scores: 0.35-0.53



After (57 papers):

\- Total chunks: 3,431 (3.7x increase)

\- Source diversity: 3.6/5 papers per query (much better)

\- Coverage: All major transformer topics

\- Relevance scores: 0.51-0.59 (significantly higher)

\- More specific answers



\*\*Test results:\*\*



Query: "What is BERT and how does it work?"

\- Relevance: 0.5103

\- Diversity: 3/5 unique papers

\- Answer: Specific BERT explanation (bidirectional encoder representations)



Query: "Explain efficient transformer architectures"

\- Relevance: 0.5394

\- Diversity: 5/5 unique papers

\- Answer: ResT, Swin Transformer specifics



Query: "How are transformers used in time series?"

\- Relevance: 0.5919 (highest score)

\- Diversity: 1/5 (specialized topic)

\- Answer: Forecasting, anomaly detection applications



Query: "What is cross-attention?"

\- Relevance: 0.5528

\- Diversity: 4/5 unique papers

\- Answer: Technical explanation with asymmetric mapping



Query: "Compare different vision transformer architectures"

\- Relevance: 0.5501

\- Diversity: 5/5 unique papers

\- Answer: Comparative analysis of Swin, Swin-V2, WSA, SWSA



\*\*Decisions made:\*\*

\- 57 papers is excellent for demo

&nbsp; - Sufficient coverage across topics

&nbsp; - Good source diversity (3-5 papers per query)

&nbsp; - Retrieval quality strong (0.51-0.59 relevance)

&nbsp; - Can expand to 100+ later if needed

&nbsp; 

\- Diverse topics over deep coverage in one area

&nbsp; - Better demonstrates RAG versatility

&nbsp; - Shows system handles different transformer applications

&nbsp; - More impressive for interviews



\*\*Learnings:\*\*

\- More papers significantly improves retrieval quality

\- 3.7x more chunks led to higher relevance scores

\- Diversity matters: 5/5 unique sources for broad queries

\- Strategic selection better than random download

\- 3,431 chunks is ideal corpus size for demo

\- FAISS scales effortlessly (0.004s build time, no degradation)



\*\*Performance observations:\*\*

\- Query latency: 1.7-3.6 seconds (acceptable)

\- FAISS retrieval: <5ms (instant)

\- LLM generation: 1.5-3.5s (bottleneck, expected)

\- Token usage: ~1500-2000 per query

\- Cost per query: ~$0.003



\*\*Next steps:\*\*

\- Week 2: Multi-modal (images + CLIP)

\- Build web UI (Streamlit)

\- Add evaluation metrics



\*\*Mood\*\*: System feels much more robust now with 3.7x more data


### Day 7 - Multi-Modal RAG - March 6, 2025
**Time spent**: 2 hours

**What I did:**
- Extracted 2,655 images from 57 research papers
- Generated CLIP embeddings for all images (512-dim)
- Built separate FAISS index for image search
- Tested text-to-image retrieval with 6 queries
- Verified multi-modal search quality

**Image extraction results:**
- Total papers processed: 57
- Papers with images: ~51 (90%)
- Total images extracted: 2,655
- Avg images per paper: 52
- Image formats: PNG, JPEG
- Size filtering: 100x100 to 3000x3000 pixels

**Image embedding generation:**
- Model: CLIP ViT-B/32 (openai/clip-vit-base-patch32)
- Embedding dimension: 512
- Device: GPU (CUDA)
- Processing time: ~150 seconds
- Speed: ~17.7 images/second
- Success rate: 100%
- Output size: 6.5 MB NPY, ~50 MB JSON

**FAISS image index:**
- Index type: IndexFlatL2 (exact search)
- Build time: <0.1 seconds
- Total vectors: 2,655
- Index size: ~6 MB

**Multi-modal search quality:**

Query: "transformer architecture diagram"
- Top similarity: 0.4183
- Retrieved: TransMorph architecture, ViT diagrams
- Quality: Excellent match

Query: "attention mechanism visualization"  
- Top similarity: 0.4152
- Retrieved: GPT-NeoX attention plots, mechanism diagrams
- Quality: Perfect match

Query: "neural network architecture"
- Top similarity: 0.4264 (highest overall)
- Retrieved: ResT architecture, network diagrams
- Quality: Excellent

Query: "training loss graph"
- Top similarity: 0.4293 (highest!)
- Retrieved: Actual loss curves from papers
- Quality: Perfect - exactly what was queried

Query: "comparison table results"
- Top similarity: 0.4117
- Retrieved: Results tables from multiple papers
- Quality: Good match

Query: "vision transformer image patches"
- Top similarity: 0.4146
- Retrieved: ViT patch visualizations
- Quality: Relevant

**Decisions made:**

- CLIP ViT-B/32 over larger models:
  - Reasoning: 512-dim sufficient for 2,655 images
  - Fast inference (GPU: 17.7 imgs/sec)
  - Good text-image alignment
  - Can upgrade to ViT-L/14 later if needed

- Separate FAISS index for images:
  - Reasoning: Different embedding dimensions (512 vs 384)
  - Allows independent scaling
  - Can weight text vs image results separately
  - Clean separation of concerns

- Image size filtering (100-3000 pixels):
  - Removed tiny icons/logos (<100px)
  - Removed full-page scans (>3000px)
  - Kept relevant figures, diagrams, charts
  - Result: 2,655 high-quality images

**Technical observations:**
- CLIP text-image embeddings in same space
- Text queries retrieve semantically similar images
- Similarity scores 0.41-0.43 (typical for cross-modal)
- Lower than text-text (0.51-0.59) but still strong
- GPU acceleration essential for 2,655 images

**Challenges:**

- Initial CLIP API confusion:
  - Error: 'BaseModelOutputWithPooling' has no attribute 'norm'
  - Root cause: Using get_text_features() incorrectly
  - Solution: Access model components directly (text_model, visual_projection)
  - Applied same fix to both embedding generation and search

- Large image corpus (2,655 images):
  - Initial concern about processing time
  - GPU handled it well (~150 seconds)
  - FAISS index build still instant (<0.1s)
  - No performance degradation

**Learnings:**
- CLIP enables powerful text-to-image search
- Same embedding space for text and images
- Cross-modal retrieval harder than single-modal (lower scores expected)
- Image extraction from PDFs yields rich visual content
- 52 images per paper is excellent coverage
- PyMuPDF better than pdf2image for extraction

**Quality observations:**
- Retrieved images semantically match text queries
- "Training loss graph" finds actual loss curves
- "Architecture diagram" finds network diagrams
- Cross-paper retrieval working (diversity)
- Page numbers preserved for citation

**What multi-modal enables:**
- Answer questions about figures and diagrams
- "Show me transformer architecture" → retrieve actual diagrams
- Visual evidence for claims
- Richer context for LLM generation
- Better user experience (show don't just tell)

**Performance metrics:**
- Image search latency: <10ms (FAISS)
- Text encoding: ~50ms (CLIP text model)
- Total query time: ~60ms
- Scales to 10k+ images easily

**Next session:**
- Integrate multi-modal search into RAG pipeline
- Combined text + image retrieval
- LLM references both text chunks and figures
- Test end-to-end multi-modal Q&A

**Mood**: Multi-modal RAG working! Text-to-image search is impressive.

### Day 7 Complete - Multi-Modal RAG Integration - March 6, 2025
**Time spent**: 45 minutes

**What I did:**
- Integrated text and image retrieval into unified pipeline
- Modified RAG to retrieve both text chunks and images
- Updated LLM prompt to reference figures
- Tested end-to-end multi-modal Q&A
- Created interactive multi-modal interface

**Multi-modal pipeline architecture:**
- Dual retrieval: Text (3 chunks) + Images (2 figures)
- Two FAISS indexes: Text (384-dim) + Images (512-dim)
- Two embedding models: MiniLM (text) + CLIP (images)
- Combined context sent to LLM
- LLM references both text and figures in answer

**Test results:**

Query: "Explain the transformer architecture. Are there any diagrams?"
- Retrieved: 3 text chunks + 2 architecture diagrams
- Text scores: 0.51-0.53
- Image scores: 0.41-0.42
- Answer: Referenced [Figure 1] and [Figure 2]
- Quality: Excellent - LLM mentioned specific diagrams

Query: "How does the attention mechanism work visually?"
- Retrieved: Attention mechanism explanations + visualizations
- LLM output: Explained concept AND referenced visual diagrams
- Figure relevance: Perfect match

Query: "Show me examples of training loss curves"
- Retrieved: Loss curve descriptions + actual loss plots
- Images: Training loss graphs from GPT-3, transformer papers
- Quality: Retrieved exactly what was asked

**System capabilities:**
- Answers questions using text context
- References relevant figures in explanations
- Provides paths to actual images
- User can view referenced diagrams
- Combines semantic understanding (text) with visual evidence (images)

**Performance:**
- Text retrieval: ~5ms
- Image retrieval: ~50ms (CLIP encoding)
- Total retrieval: ~55ms
- LLM generation: ~2s
- Total latency: ~2.1s

**Decisions made:**
- 3 text chunks + 2 images default:
  - Reasoning: Text provides detailed explanation
  - Images provide visual evidence
  - 2 images enough without overwhelming context
  - Balances text detail with visual support

- Separate retrieval for text and images:
  - Different embedding models (MiniLM vs CLIP)
  - Different scoring (can't directly compare)
  - Allows independent k values
  - Clean separation of concerns

- LLM instructed to reference figures:
  - System prompt explicitly mentions [Figure N] notation
  - Encourages visual thinking
  - User knows which images to view
  - Better user experience

**Quality observations:**
- LLM successfully references figures when relevant
- Retrieved images semantically match queries
- Combined text+image context richer than text alone
- Users can verify claims by viewing figures
- System handles "show me" queries well

**Challenges:**
- Context window management:
  - 3 text chunks ~1500 tokens
  - Image descriptions ~200 tokens
  - Total context ~1700 tokens (safe)
  - Could add more images if needed

- Image reference formatting:
  - Tested different formats ([Fig 1] vs [Figure 1])
  - Settled on [Figure N] (clearer, more academic)
  - LLM naturally uses this notation

**Learnings:**
- Multi-modal RAG significantly richer than text-only
- CLIP enables powerful text-to-image search
- LLMs can effectively reference visual content
- Figure paths allow users to view actual diagrams
- Visual evidence builds trust in answers

**What multi-modal RAG enables:**
- "Show me X" queries → retrieve actual images
- Visual explanations for complex concepts
- Evidence-based answers (text + visual proof)
- Better understanding through diagrams
- Richer user experience

**Next steps:**
- Build web UI for better image display
- Add image thumbnails in response
- Implement image caching
- Week 3: Production polish and deployment

**Mood**: Multi-modal RAG complete! System can now reason over text and images.

### Day 7 Complete - Multi-Modal RAG Verified - March 6, 2025
**Time spent**: 1 hour total

**Testing results:**

Demo Question 1: "Explain the transformer architecture. Are there any diagrams?"
- LLM response: Successfully mentioned "Figure 1 in the provided figures illustrates the framework"
- Retrieved: TransMorph architecture, ViT architecture diagrams
- Text scores: 0.477-0.469
- Image scores: 0.414-0.413
- Tokens: 2013
- Latency: 3.70s

Demo Question 2: "How does the attention mechanism work visually?"
- LLM response: Referenced multiple figures (Figure 12, Figures 6/7/8, Figure 2)
- Retrieved: Attention heatmaps, visualization papers
- Text scores: 0.564-0.545 (very high)
- Image scores: 0.415
- Quality: LLM naturally discussed visual explanations

Demo Question 3: "Show me examples of training loss curves"
- LLM response: "Figure 2 from the paper Benign Overfitting..."
- Retrieved: Actual loss curve images (GPT-3, transformer papers)
- Image scores: 0.426-0.425
- Perfect match: Retrieved exactly what was requested

Interactive testing:
- Query: "Show me transformer architecture diagrams"
  - Retrieved architecture diagrams successfully
  - LLM provided figure references
  
- Query: "What do training loss curves look like?"
  - Retrieved loss visualizations
  - Referenced [Figure 1] and [Figure 2]
  
- Query: "Explain attention with a visualization"
  - Text score: 0.5887 (excellent)
  - Retrieved attention heatmaps and visualizations
  - LLM explained concept with visual references

**System capabilities verified:**
- Dual retrieval working (text + images)
- LLM successfully references figures
- Image paths accurate and accessible
- Cross-modal semantic matching strong
- "Show me" queries handled correctly
- Visual evidence supports text explanations

**Performance confirmed:**
- Text retrieval: ~5ms
- Image retrieval: ~50ms
- Total retrieval: ~55ms
- LLM generation: 2-5s
- Total latency: 2-5s (acceptable)
- Token usage: 1144-2013 per query
- Cost: ~$0.003-0.004 per query

**Week 2 Day 7: COMPLETE**
Multi-modal RAG system fully operational with text and image retrieval.

**Project status:**
- Week 1: Core RAG (Days 1-6) - COMPLETE
- Week 2 Day 7: Multi-modal - COMPLETE
- Total progress: 7/31 days (23% timeline, ~60% functionality)
- Ahead of schedule: 20+ days buffer remaining

### Day 8 - Streamlit Web UI - March 6, 2025
**Time spent**: 3 hours

**What I did:**
- Built professional web interface with Streamlit
- Created multi-tab layout (Chat, Examples, How It Works)
- Integrated multi-modal RAG pipeline
- Added inline image display
- Implemented chat history
- Created example query buttons
- Added system information dashboard

**UI Features:**

Main Interface:
- Clean chat-based interaction
- Real-time answer generation
- Source citation display
- Inline image viewing
- Chat history persistence
- Professional styling

Sidebar:
- Retrieval configuration (text_k, image_k sliders)
- System metrics (chunk count, image count)
- About section
- Clear history button

Tabs:
1. Chat: Main Q&A interface
2. Example Queries: Pre-built questions (architecture, visual, comparison)
3. How It Works: System explanation and architecture

**Technical implementation:**

Session state management:
- RAG pipeline loaded once, cached
- Chat history persisted across queries
- Settings adjustable per query

Image display:
- PIL integration for image loading
- Responsive column layout for multiple images
- Error handling for missing images
- Thumbnail display with metadata

Source presentation:
- Expandable source sections
- Text sources with relevance scores
- Image sources with inline display
- Clean formatting and typography

**UI/UX decisions:**

- Chat interface over form-based:
  - More natural conversation flow
  - Easier to see history
  - Modern UI pattern
  
- Inline image display:
  - Users see figures immediately
  - No need to open separate files
  - Better understanding of visual evidence
  
- Three-tab layout:
  - Chat for interaction
  - Examples for new users
  - How It Works for understanding
  
- Sidebar for settings:
  - Doesn't clutter main area
  - Quick access to configuration
  - System info always visible

**Performance:**
- Initial load: ~10 seconds (model loading)
- Per query: 2-3 seconds (same as CLI)
- Image display: Instant (local files)
- UI responsiveness: Excellent

**Testing results:**

Query: "What is the transformer architecture?"
- Answer displayed in chat
- Sources expandable
- 3 text sources shown
- 2 images displayed inline
- Latency: 3.2s
- Clean presentation

Query: "Show me architecture diagrams"
- Retrieved relevant diagrams
- Images displayed immediately
- Captions with paper names and pages
- User can zoom images
- Professional appearance

Example queries:
- All 12 examples tested
- Click triggers immediate query
- Results display correctly
- Good variety of question types

**Quality observations:**
- UI is clean and professional
- Images display correctly
- Sources are well-formatted
- Chat history works smoothly
- Settings changes take effect
- No UI bugs encountered

**User experience improvements:**
- Inline images much better than file paths
- Chat history helps see conversation context
- Example queries help new users
- System info builds confidence
- How It Works tab explains system

**Challenges:**
- Image loading error handling:
  - Some images might not exist
  - Added try-catch with friendly message
  
- State management:
  - Streamlit reruns on interaction
  - Used session_state to persist data
  
- Layout optimization:
  - Multiple images in columns
  - Responsive design for different screens

**Next steps:**
- Add more example queries
- Implement query history download
- Add feedback buttons (thumbs up/down)
- Optional: Deploy to Streamlit Cloud

**Mood**: Web UI looks professional! Much better than CLI for demos.

**UI Testing Results:**

Main interface:
- Clean dark theme with brain emoji branding
- Professional typography and spacing
- Sidebar metrics displaying correctly (3,431 chunks, 2,655 images)
- Sliders working for retrieval configuration
- System responsive and fast

Chat functionality:
- Questions display in chat bubbles
- Answers well-formatted with proper paragraphs
- View Sources expander working perfectly
- Text sources showing with relevance scores
- Figure references in answers ([Figure N])
- Metadata displayed (latency, tokens, model)
- Chat history persists across queries

Example queries:
- All 12 examples categorized properly
- Buttons display correctly
- Initial implementation had navigation issue
- Fixed: Examples now generate answers directly
- Success message guides user to Chat tab

How It Works tab:
- Four-section breakdown clear and informative
- System architecture diagram in ASCII art
- Performance metrics displayed professionally
- Educational value for users and interviewers

Overall quality:
- UI is production-ready
- Professional appearance
- Fast and responsive
- No visual bugs
- Ready for screenshots and demo video

Screenshots captured:
- ui_MainPage.png: Clean landing page
- ui_ExampleQs1.png: Chat with answer
- ui_ExampleQs2.png: Answer with sources
- ui_HowItWorks1.png: System explanation
- ui_HowItWorks2.png: Architecture and performance
- ui_ExampleQueries.png: Example buttons

UI feedback:
- Design exceeds expectations
- Professional enough for job applications
- Inline images work perfectly
- Citation tracking clear and useful
- Would impress in interviews

### Day 9 - Evaluation Framework - March 7, 2025
**Time spent**: 2 hours

**What I did:**
- Created test query dataset (10 queries across difficulty levels)
- Built comprehensive evaluation framework
- Measured retrieval accuracy, latency, and cost
- Generated quantitative performance metrics
- Saved detailed results to JSON

**Test dataset:**
- 10 queries total
- Categories: Architecture (3), Mechanism (3), Vision (1), Comparison (2), Data (1)
- Difficulty: Easy (3), Medium (5), Hard (2)
- Expected papers defined for precision measurement

**Evaluation metrics collected:**

Retrieval Performance:
- Mean Precision: 35.00% (finding expected papers in top-5)
- Top Relevance Score: 0.539 (strong semantic matching)
- Average Relevance Score: 0.524 (consistent quality)
- Retrieval Time P50: 20.1ms (sub-100ms target achieved)
- Retrieval Time P95: ~30ms (fast even at tail)

End-to-End Performance:
- Total Latency P50: 3.02s (acceptable for interactive use)
- Total Latency P95: ~3.5s (consistent performance)
- LLM Generation Time: 2.93s (97% of total latency)
- Retrieval Time: 0.09s (3% of total latency)
- Average Tokens per Query: 1896 (efficient context usage)
- Average Answer Length: 1090 characters (detailed responses)
- Text Citations per Answer: 2.3 (good source attribution)
- Figure Citations per Answer: 0.1 (low, most queries text-focused)

Cost Analysis:
- Cost per Query: $0.0038 (less than half a cent)
- Total Evaluation Cost: $0.0379 (under 4 cents for 10 queries)
- Projected Monthly Cost (1000 queries): $3.80
- Breakdown: Input tokens ~70%, Output tokens ~30%

**Performance interpretation:**

Retrieval Quality:
- 35% precision is good for 57-paper corpus
- Industry baseline: 30-50% for RAG systems
- High relevance scores (0.52-0.54) indicate strong semantic matching
- FAISS optimization working (20ms retrieval)

Latency Breakdown:
- LLM generation is bottleneck (97% of time)
- Retrieval is optimized (3% of time)
- Cannot reduce LLM time without switching models
- Overall 3s latency acceptable for demo/research tool

Citation Quality:
- 2.3 text citations per answer (good)
- Most answers reference 2-3 sources
- Low figure citations (0.1) because test queries are text-heavy
- Visual queries would increase this metric

Cost Efficiency:
- Very affordable at $0.0038 per query
- GPT-3.5-turbo pricing: $0.002/1K tokens
- Could serve 1000s of queries for <$10
- Production-viable pricing

**By difficulty level:**

Easy queries (3):
- Precision: ~50% (higher, as expected)
- Latency: 2.8s average
- Good performance on straightforward questions

Medium queries (5):
- Precision: ~30% (reasonable)
- Latency: 3.1s average
- Handles nuanced questions well

Hard queries (2):
- Precision: ~20% (lower, expected)
- Latency: 3.3s average
- Complex comparisons more challenging

**Decisions made:**

- 10 query test set sufficient for demo evaluation
  - Covers diverse categories and difficulty levels
  - Larger test sets (50-100) for production systems
  - Current set validates system works correctly

- Precision at 35% acceptable for prototype
  - Could improve with:
    - More papers (100+ would help)
    - Query expansion techniques
    - Re-ranking algorithms
  - Current quality sufficient for CV/demo

- P50 latency metric chosen over mean
  - More representative of typical user experience
  - Less affected by outliers
  - Industry standard for latency reporting

**Technical observations:**
- Evaluation framework automated and reproducible
- Results saved to JSON for documentation
- Can re-run evaluation after improvements
- Metrics align with RAG system benchmarks

**Quality insights:**
- System reliably retrieves relevant context
- LLM generates well-cited answers
- Performance consistent across queries
- Cost-effective for extended use

**Limitations identified:**
- Precision could be higher (35% vs 50%+ ideal)
- Figure citations low (text-heavy test set)
- No human evaluation of answer quality yet
- Small test set (10 queries)

**Next steps:**
- Add evaluation metrics to README
- Create performance comparison table
- Consider adding re-ranking for precision boost
- Expand test set to 20-30 queries (if time)

**Files created:**
- data/evaluation/test_queries.json (test dataset)
- src/evaluate_system.py (evaluation framework)
- data/evaluation/results.json (detailed results)

**Mood**: System performance validated with hard numbers. Ready for documentation!


## Day 16 - PRODUCTION DEPLOYMENT & QUALITY IMPROVEMENTS

### Major Milestone: Live Deployment
- **URL**: https://papermind-ai-research-assistant.streamlit.app
- Successfully deployed to Streamlit Cloud after fixing:
  - Git history cleanup (removed leaked API keys)
  - OpenAI version compatibility (1.12.0 → 1.52.0)
  - httpx version pinning (0.27.2)
  - Cross-platform path handling (Windows → Linux)
  - Pillow/Streamlit image display compatibility

### Performance Enhancements
**Richer Responses:**
- Increased retrieval: 3 → 5 text chunks, 2 → 3 images
- Enhanced GPT prompt for comprehensive synthesis
- Longer responses: 1 paragraph → 2-4 paragraphs
- Added source citations with page numbers
- Increased max_tokens: 500 → 800

**UI Improvements:**
- Full text chunks visible in source viewer
- Page number display for all citations
- Better error messages and loading states

### Corpus Expansion (In Progress)
- Target: 200+ papers (from 57)
- Expected: 10,000+ chunks, 5,000+ images
- Script ready: `download_more_papers.py

### Evaluation Framework
**RAGAS Integration:**
- Added 10 test queries with ground truth
- Metrics: Answer Relevancy, Faithfulness, Context Precision, Context Recall
- Script: `src/evaluate_with_ragas.py
- Results stored in: `data/evaluation/ragas_results.json

### Technical Debt Addressed
- Fixed Windows/Linux path incompatibilities
- Proper .gitignore for secrets management
- Version pinning for reproducibility
- Better error handling in image display

### Next Steps (Days 17-21)
1. Run corpus expansion to 200 papers
2. Execute RAGAS evaluation
3. Analyze results and identify weaknesses
4. Implement hybrid search (dense + sparse)
5. Add conversation memory for multi-turn queries

---

## Day 17 - Corpus Expansion to 258 Papers - March 20, 2026

**Time spent**: 3 hours

**What I did:**
- Expanded corpus from 57 to 258 papers using download_papers.py
- Re-ran full pipeline: extraction, chunking, embedding, indexing
- Discovered and fixed a variable name bug in text_chunker.py
- Discovered and fixed a tokenization error in generate_embeddings.py
- Verified all 258 papers processed correctly

**Corpus expansion results:**
- Papers downloaded: 258 (expanded from 57)
- Text chunks generated: 11,787 (expanded from 3,431)
- Images extracted: 6,591 (expanded from 2,655)
- Embedding generation time: approximately 5 minutes on CPU
- FAISS index build time: under 1 second

**Bug 1: text_chunker.py variable name mismatch**
- Error: NameError: name 'results' is not defined at line 146
- Cause: Variable was named 'output' throughout the function but json.dump referenced 'results'
- Fix: Changed json.dump(results, f) to json.dump(output, f)
- Resolution time: 5 minutes

**Bug 2: generate_embeddings.py tokenization error**
- Error: TypeError: TextEncodeInput must be Union[TextInputSequence, ...]
- Cause: Some chunks contained null bytes, Unicode surrogates, and other invalid characters that the sentence-transformers tokenizer could not handle
- Fix: Added a robust text cleaning pipeline that strips null bytes and Unicode surrogates, then tests each chunk individually before batch encoding, skipping any that still fail with clear log output
- Resolution time: 2 hours
- Failure rate: 3 chunks skipped out of 11,787 (0.03%)

**Decisions made:**
- Kept CPU-only embedding generation rather than switching to GPU
  - Reasoning: The cloud deployment target is CPU-only anyway, so keeping local generation consistent with the deployment environment reduces surprises
  - Trade-off: Slower locally, but a one-time cost since embeddings are pre-generated

**Learnings:**
- Robust text cleaning is essential before any tokenizer call -- academic PDFs contain more encoding edge cases than expected
- Testing chunks individually before batch encoding is the correct pattern for fault-tolerant embedding pipelines
- A 0.03% failure rate on a text cleaning pipeline is acceptable

**Mood**: Frustrating bugs but satisfying to fix. System now 4.5x larger.

---

## Day 18 - Image Embeddings and GitHub Size Crisis - March 23, 2026

**Time spent**: 4 hours

**What I did:**
- Generated CLIP image embeddings for 6,591 figures
- Attempted first push to GitHub -- failed due to file size limits
- Analysed repository size and identified oversized files
- Created fresh repository with correct file selection
- Successfully pushed 93 MB repository to GitHub

**Image embedding generation:**
- Model: CLIP ViT-B/32
- Images processed: 6,591
- Processing time: approximately 8 minutes on CPU
- Success rate: 100%
- Output: image_embeddings.npy (13 MB)

**GitHub size crisis:**

First push attempt failed with:
```
error: File data/processed/embeddings_512.json is 142.95 MB;
this exceeds GitHub's file size limit of 100.00 MB
```

Full repository breakdown:
- embeddings_512.json: 142 MB (JSON embedding file)
- image_embeddings.json: 73 MB (JSON embedding file)
- images/ folder: 419 MB (6,591 extracted PNG files)
- PDF files: 500 MB

Total: approximately 606 MB -- six times over the GitHub limit.

**Solution:**

Key insight: .npy (NumPy binary) format is approximately 9 times smaller than JSON for floating-point arrays.
- embeddings_512.json: 142 MB vs embeddings_512.npy: 5 MB
- image_embeddings.json: 73 MB vs image_embeddings.npy: 13 MB

Approach taken:
1. Renamed local folder to PaperMind_BACKUP
2. Cloned fresh repository from GitHub
3. Copied only the files that fit within limits: source code, metadata JSONs, FAISS indexes, .npy embedding files
4. Updated .gitignore to exclude PDFs, extracted images, and JSON embeddings
5. Final push: 93 MB -- within the 100 MB limit

Files committed:
- embeddings_512.npy (5 MB)
- image_embeddings.npy (13 MB)
- chunks_recursive_512.json (23 MB)
- extracted_text.json (17 MB)
- faiss_index/ (20 MB)
- faiss_image_index/ (15 MB)

Files excluded:
- PDF files (~500 MB) -- can be regenerated locally if needed
- images/ folder (~419 MB) -- can be regenerated locally if needed
- JSON embedding files (~215 MB) -- replaced by .npy equivalents

**Decisions made:**
- Fresh repository over git history rewriting
  - Reasoning: Trying to remove large files from git history with git filter-branch or BFG is error-prone and time-consuming. Starting fresh with the correct file selection is cleaner and faster.
  - Trade-off: Lost granular commit history from earlier development days. Acceptable given the deadline.

**Learnings:**
- Always plan the deployment file strategy before generating large intermediate files
- .npy format should be the default for any floating-point matrix that needs to be stored or committed
- git filter-branch exists but a fresh repository is often the faster practical choice

**Mood**: Stressful but resolved cleanly. The 9x compression insight was the key.

---

## Day 19 - Streamlit Cloud Block and Debugging - March 23-30, 2026

**Time spent**: 6 hours across multiple sessions

**What I did:**
- Attempted first cloud deployment to Streamlit Cloud
- Account was blocked due to excessive CPU usage on startup
- Emailed Streamlit support and waited for unblock
- Diagnosed and fixed four separate bugs in the codebase
- Verified local app running correctly after all fixes

**Cloud deployment attempt:**

Deployed to Streamlit Cloud. App started but then the account was blocked:
```
Error 403: Your account has exceeded the fair-use limits
and was blocked by the system.
```

Root cause: The original streamlit_app.py called subprocess.run() to trigger embedding generation on startup. On the cloud environment, attempting to generate embeddings for 11,787 chunks and 6,591 images on a shared CPU exceeded the fair-use limits and triggered an automatic block.

Timeline:
- Blocked: March 23, approximately 2pm
- Support email sent: March 23, approximately 3pm
- Support response: March 26, 10am
- Account unblocked: March 26, 11am

Correct solution: Remove all embedding generation from the cloud startup path entirely. Pre-generate locally, commit .npy files, and simply load them at startup. Startup time went from a potential 10+ minutes to under 15 seconds.

**Bug 1: streamlit_app.py entry point using exec()**

Problem: The original entry point used exec(f.read()) to load web_ui.py, which caused namespace pollution and made import errors nearly impossible to debug.

Fix:
```python
from web_ui import run_app
if __name__ == "__main__":
    run_app()
```

**Bug 2: API key loading failure**

Problem: The API key loading logic did not handle the case where st.secrets exists but the key is absent, and did not strip whitespace from loaded keys. This caused silent failures where an apparently valid key was actually invalid due to leading or trailing whitespace.

Fix: Added .strip() to all three key loading paths (Streamlit secrets, environment variable, .env file). Added explicit fallback chain with clear error messages at each stage.

**Bug 3: Duplicate RAG initialisation in web_ui.py**

Problem: The RAG pipeline was being initialised twice -- once at module level (line 49) and once inside session_state (line 77). This doubled model loading time and memory usage on every cold start.

Fix: Removed the module-level initialisation at line 49, kept only the session_state version which correctly initialises once and persists across reruns.

**Bug 4: secrets.toml format error**

Problem: The secrets.toml file had the API key without quotes:
```
OPENAI_API_KEY = sk-proj-xxx
```
Streamlit's TOML parser raised TomlDecodeError: Invalid date or number because it tried to interpret the key value as a non-string type.

Fix: Added quotes:
```
OPENAI_API_KEY = "sk-proj-xxx"
```

**Learnings:**
- Never run heavy compute on cloud startup -- pre-process everything locally
- exec() is an anti-pattern in any production Python code, including Streamlit apps
- Whitespace in API keys is a common silent failure mode -- always strip
- Session state initialisation must be guarded with if 'key' not in st.session_state to prevent duplicate loading
- TOML string values require quotes, unlike some other config formats

**Mood**: The three-day wait for support was difficult under deadline pressure. The debugging session after was productive.

---

## Day 20 - UI Fixes, Documentation Sprint, and Final Deployment - April 2-4, 2026

**Time spent**: 8 hours across April 2 to April 4

**What I did:**
- Fixed UI duplication bug caused by unconditional st.rerun() calls
- Removed image display from the UI (text-only figure references)
- Confirmed local and cloud deployments both fully operational
- Updated the How It Works tab with accurate corpus numbers
- Completed README.md (full rewrite)
- Completed ARCHITECTURE.md (technical deep-dive)
- Updated DEVLOG with final entries

**Bug: UI duplication on query submission**

Problem: After submitting a question, the page would visually duplicate before showing the answer. The cause was st.rerun() being called unconditionally after ask_question(), even when the query raised an exception. This triggered a full page rerender mid-error state, producing a doubled UI.

Fix: Made all st.rerun() calls conditional on chat_history having content, meaning the rerun only fires if the query actually completed and appended a result:
```python
if question:
    ask_question(question)
    if st.session_state.chat_history:
        st.rerun()
```
Applied the same pattern to all four button handlers in the Example Queries tab.

**Image display removal:**

The extracted figure PNG files are not committed to GitHub due to size constraints. The UI was showing "Image not found" warnings in yellow boxes for every query because the image paths pointed to local Windows paths that do not exist in the cloud environment.

Decision: Remove the image rendering block entirely and replace with clean text-only figure references showing the paper name, filename, page number, and relevance score. The multimodal retrieval capability is preserved -- the RAG pipeline still searches both indexes and the LLM still references figures in answers -- but the UI no longer attempts to display images it cannot access.

This also allowed removing the PIL import from web_ui.py, which was the only remaining use of Pillow in the UI layer.

**Cloud deployment fix:**

The web deployment was failing at query time with a 401 authentication error. The cause was a stale or incorrect API key in the Streamlit Cloud secrets dashboard. After updating the secret with the correct key and pushing the updated web_ui.py, both local and cloud deployments are confirmed working correctly.

**Documentation completed:**

README.md: Full rewrite covering features, live demo link, architecture diagram, technical specifications table, performance metrics, local setup instructions, project structure, key design decisions, future improvements, and contact information.

ARCHITECTURE.md: Technical deep-dive covering all five pipeline stages (acquisition, processing, chunking, embedding generation, index construction), both retrieval systems with code examples, the generation pipeline including prompt design and cost breakdown, deployment architecture with repository size management strategy, five key design decisions with full reasoning, performance characteristics with latency breakdown and scalability limits, and four concrete future improvements.

**Project totals:**
- Development time: approximately 40 hours over 5 weeks
- Lines of code: approximately 2,500
- Papers indexed: 258
- Text chunks: 11,787
- Image embeddings: 6,591
- Total embeddings: 18,378
- Bugs fixed: tracked across devlog entries
- Deployment platform: Streamlit Cloud (live)

**Lessons learned across the full project:**

1. Plan deployment constraints before generating data -- the GitHub size limit and Streamlit CPU limits should have been researched on day one.
2. Pre-process everything that can be pre-processed -- embedding generation belongs in the development pipeline, not the startup path.
3. File format matters more than expected -- .npy vs .json was a 9x size difference that determined whether the project could be deployed at all.
4. Clean architecture pays off -- removing the exec() anti-pattern made the subsequent debugging significantly faster.
5. Documentation cannot wait until the end -- writing ARCHITECTURE.md after the fact required reconstructing decisions that would have taken seconds to note at the time.
6. Robust error handling in text pipelines is not optional -- academic PDFs contain enough encoding edge cases that any tokenizer call without sanitisation will eventually fail.

**Mood**: Satisfied. System is live, documented, and functional end-to-end.