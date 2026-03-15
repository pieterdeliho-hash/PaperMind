"""
Search interface for querying papers
Demonstrates end-to-end retrieval (without LLM yet)
"""

import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from typing import List, Dict


class PaperSearchEngine:
    """Search engine for research papers using FAISS"""

    def __init__(
            self,
            index_path: str = "data/processed/faiss_index/faiss_index.bin",
            metadata_path: str = "data/processed/faiss_index/chunks_metadata.pkl",
            model_name: str = "all-MiniLM-L6-v2"
    ):
        print("Loading search engine...")

        # Load FAISS index
        self.index = faiss.read_index(index_path)

        # Load chunk metadata
        with open(metadata_path, 'rb') as f:
            self.chunks_metadata = pickle.load(f)

        # Load embedding model
        self.model = SentenceTransformer(model_name)

        print(f"✓ Loaded {self.index.ntotal} chunks")
        print(f"✓ Model: {model_name}")
        print("Search engine ready!\n")

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """
        Search for relevant chunks

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of result dictionaries with chunk info and scores
        """
        # Embed query
        query_embedding = self.model.encode([query])[0]
        query_vector = query_embedding.astype('float32').reshape(1, -1)

        # Search FAISS index
        distances, indices = self.index.search(query_vector, k)

        # Format results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            chunk = self.chunks_metadata[idx]

            # Convert L2 distance to similarity score (0-1)
            # Lower distance = higher similarity
            # Approximate conversion: similarity ≈ 1 / (1 + distance)
            similarity_score = 1 / (1 + dist)

            results.append({
                'rank': i + 1,
                'score': similarity_score,
                'distance': dist,
                'paper': chunk['paper_filename'],
                'chunk_id': chunk['chunk_id'],
                'chunk_text': chunk['chunk_text'],
                'tokens': chunk['chunk_tokens']
            })

        return results

    def display_results(self, query: str, results: List[Dict]):
        """Pretty print search results"""
        print("=" * 80)
        print(f"QUERY: {query}")
        print("=" * 80)

        for result in results:
            print(f"\n[{result['rank']}] Score: {result['score']:.4f}")
            print(f"Paper: {result['paper'][:60]}")
            print(f"Chunk {result['chunk_id'] + 1} ({result['tokens']} tokens)")
            print("-" * 80)

            # Show first 300 characters
            preview = result['chunk_text'][:300].replace('\n', ' ')
            print(preview + "...")
            print("-" * 80)


def interactive_search():
    """Interactive search loop"""
    engine = PaperSearchEngine()

    print("=" * 80)
    print("PAPERMIND SEARCH ENGINE")
    print("=" * 80)
    print("Enter your questions about transformer research papers")
    print("Type 'quit' to exit\n")

    while True:
        query = input("Query: ").strip()

        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not query:
            continue

        # Search
        results = engine.search(query, k=3)

        # Display
        engine.display_results(query, results)
        print("\n")


if __name__ == "__main__":
    # Demo queries
    engine = PaperSearchEngine()

    demo_queries = [
        "What is the attention mechanism?",
        "How do vision transformers process images?",
        "What are the key advantages of transformers over RNNs?"
    ]

    print("=" * 80)
    print("DEMO: Running example queries")
    print("=" * 80)

    for query in demo_queries:
        results = engine.search(query, k=2)
        engine.display_results(query, results)
        print("\n")

    # Start interactive mode
    print("\n" + "=" * 80)
    print("Starting interactive mode...")
    print("=" * 80)
    interactive_search()