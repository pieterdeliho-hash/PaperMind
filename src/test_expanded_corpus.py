"""
Test retrieval quality with expanded corpus
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import rag_pipeline
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Now change directory so relative paths work
import os
os.chdir(parent_dir)

from rag_pipeline import RAGPipeline

rag = RAGPipeline()

print("=" * 80)
print("TESTING EXPANDED CORPUS")
print("=" * 80)
print(f"Total chunks indexed: {rag.index.ntotal}")
print(f"Expected: ~3,300-3,500 chunks from ~50 papers\n")

# Test queries that should benefit from more papers
test_queries = [
    "What is BERT and how does it work?",
    "Explain efficient transformer architectures",
    "How are transformers used in time series?",
    "What is cross-attention?",
    "Compare different vision transformer architectures"
]

for query in test_queries:
    print(f"\nQuery: {query}")
    print("-" * 80)

    result = rag.query(query, k=5, verbose=False)

    # Check diversity of sources
    papers = [s['paper'][:40] for s in result['sources']]
    unique_papers = len(set(papers))

    print(f"Answer: {result['answer'][:200]}...")
    print(f"\nSource diversity: {unique_papers}/5 unique papers")
    print(f"Top relevance: {result['sources'][0]['score']:.4f}")
    print(f"Latency: {result['latency']:.2f}s")