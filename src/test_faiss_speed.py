"""
Compare FAISS vs naive search speed
Demonstrate why FAISS is necessary for production
"""

import numpy as np
import faiss
import pickle
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model
print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index
print("Loading FAISS index...")
index = faiss.read_index("data/processed/faiss_index/faiss_index.bin")

with open("data/processed/faiss_index/chunks_metadata.pkl", 'rb') as f:
    chunks_metadata = pickle.load(f)

# Load embeddings for naive comparison
embeddings = np.load("data/processed/embeddings_512.npy")

print(f"Loaded {index.ntotal} vectors\n")

# Test queries
test_queries = [
    "What is the attention mechanism?",
    "How do transformers work?",
    "What datasets were used?",
    "Image classification accuracy",
    "Self-attention mechanism explanation"
]

print("=" * 80)
print("SPEED COMPARISON: FAISS vs Naive Cosine Similarity")
print("=" * 80)

for query_text in test_queries:
    print(f"\nQuery: '{query_text}'")
    print("-" * 80)

    # Embed query
    query_embedding = model.encode([query_text])[0]

    # Method 1: Naive cosine similarity (current approach)
    start = time.time()
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    top_indices_naive = np.argsort(similarities)[::-1][:5]
    naive_time = (time.time() - start) * 1000  # Convert to ms

    # Method 2: FAISS search
    start = time.time()
    query_faiss = query_embedding.astype('float32').reshape(1, -1)
    distances, top_indices_faiss = index.search(query_faiss, 5)
    faiss_time = (time.time() - start) * 1000  # Convert to ms

    # Calculate speedup
    speedup = naive_time / faiss_time if faiss_time > 0 else float('inf')

    print(f"Naive search time: {naive_time:.3f} ms")
    print(f"FAISS search time: {faiss_time:.3f} ms")
    print(f"Speedup: {speedup:.1f}x faster")

    # Show top result from FAISS
    top_idx = top_indices_faiss[0]
    top_chunk = chunks_metadata[top_idx]
    preview = top_chunk['chunk_text'][:150].replace('\n', ' ')

    print(f"\nTop result:")
    print(f"  Paper: {top_chunk['paper_filename'][:50]}")
    print(f"  Preview: {preview}...")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)