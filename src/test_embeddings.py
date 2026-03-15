"""Test embedding quality with similarity search"""
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model and embeddings
print("Loading model and embeddings...")
model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = np.load("data/processed/embeddings_512.npy")

with open("data/processed/embeddings_512.json", 'r') as f:
    data = json.load(f)
chunks = data['chunks']

print(f"Loaded {len(embeddings)} embeddings\n")

# Test queries
test_queries = [
    "What is the attention mechanism?",
    "How do transformers work?",
    "What datasets were used?",
    "Image classification accuracy"
]

print("=" * 80)
print("TESTING SEMANTIC SIMILARITY")
print("=" * 80)

for query in test_queries:
    print(f"\nQuery: '{query}'")
    print("-" * 80)

    # Embed query
    query_embedding = model.encode([query])[0]

    # Calculate similarity with all chunks
    similarities = cosine_similarity([query_embedding], embeddings)[0]

    # Get top 3
    top_indices = np.argsort(similarities)[::-1][:3]

    for i, idx in enumerate(top_indices, 1):
        score = similarities[idx]
        chunk = chunks[idx]
        text_preview = chunk['chunk_text'][:200].replace('\n', ' ')

        print(f"\n{i}. Score: {score:.4f}")
        print(f"   Paper: {chunk['paper_filename'][:50]}")
        print(f"   Preview: {text_preview}...")

print("\n" + "=" * 80)