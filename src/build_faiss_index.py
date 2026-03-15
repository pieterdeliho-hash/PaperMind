"""
Build FAISS vector index for fast similarity search
Converts embeddings into optimized search structure
"""

import numpy as np
import faiss
import json
import pickle
from pathlib import Path
import time


class FAISSIndexBuilder:
    """Build and manage FAISS index for vector search"""

    def __init__(self, embedding_dim: int = 384):
        """
        Args:
            embedding_dim: Dimension of embedding vectors
        """
        self.embedding_dim = embedding_dim
        self.index = None
        self.chunks_metadata = None

    def build_index(
            self,
            embeddings_file: str,
            index_type: str = "Flat"
    ):
        """
        Build FAISS index from embeddings

        Args:
            embeddings_file: Path to embeddings NPY file
            index_type: Type of FAISS index
                - "Flat": Exact search (best quality, slower for millions)
                - "IVF": Approximate search (faster, slight quality loss)
        """
        print("=" * 70)
        print("BUILDING FAISS INDEX")
        print("=" * 70)

        # Load embeddings
        print(f"\nLoading embeddings from: {embeddings_file}")
        embeddings = np.load(embeddings_file)

        # Load metadata
        json_file = Path(embeddings_file).with_suffix('.json')
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.chunks_metadata = data['chunks']

        print(f"Loaded {len(embeddings)} embeddings")
        print(f"Embedding dimension: {embeddings.shape[1]}")
        print(f"Index type: {index_type}")

        # Ensure embeddings are float32 (FAISS requirement)
        embeddings = embeddings.astype('float32')

        # Build index
        print(f"\nBuilding {index_type} index...")
        start_time = time.time()

        if index_type == "Flat":
            # L2 (Euclidean) distance - most common for embeddings
            # Note: For cosine similarity, we normalize vectors first
            self.index = faiss.IndexFlatL2(self.embedding_dim)

        elif index_type == "IVF":
            # IVF = Inverted File index (for larger datasets)
            # Clusters vectors for faster approximate search
            nlist = 100  # Number of clusters
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
            # Train the index
            print("Training IVF index...")
            self.index.train(embeddings)

        # Add vectors to index
        self.index.add(embeddings)

        elapsed = time.time() - start_time

        print(f"✓ Index built in {elapsed:.3f} seconds")
        print(f"  Total vectors indexed: {self.index.ntotal}")

        return self.index

    def save_index(self, output_dir: str):
        """
        Save FAISS index and metadata to disk

        Args:
            output_dir: Directory to save index files
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        # Save FAISS index
        index_file = output_path / "faiss_index.bin"
        faiss.write_index(self.index, str(index_file))
        print(f"✓ Saved FAISS index: {index_file}")

        # Save metadata (chunk info for retrieval)
        metadata_file = output_path / "chunks_metadata.pkl"
        with open(metadata_file, 'wb') as f:
            pickle.dump(self.chunks_metadata, f)
        print(f"✓ Saved metadata: {metadata_file}")

        # Save index info
        info = {
            "num_vectors": self.index.ntotal,
            "embedding_dim": self.embedding_dim,
            "index_type": type(self.index).__name__
        }
        info_file = output_path / "index_info.json"
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)
        print(f"✓ Saved index info: {info_file}")

        print(f"\n✓ Index saved to: {output_path}")

    def test_search(self, query_embedding: np.ndarray, k: int = 5):
        """
        Test search with a query

        Args:
            query_embedding: Query vector (384-dim)
            k: Number of results to return

        Returns:
            distances, indices of top-k results
        """
        query = query_embedding.astype('float32').reshape(1, -1)
        distances, indices = self.index.search(query, k)
        return distances[0], indices[0]


if __name__ == "__main__":
    print("=" * 70)
    print("FAISS INDEX BUILDER")
    print("Building production-ready vector search index")
    print("=" * 70)

    # Initialize builder
    builder = FAISSIndexBuilder(embedding_dim=384)

    # Build index
    builder.build_index(
        embeddings_file="data/processed/embeddings_512.npy",
        index_type="Flat"  # Exact search (best for <10k vectors)
    )

    # Save to disk
    builder.save_index(output_dir="data/processed/faiss_index")

    print("\n" + "=" * 70)
    print("INDEX BUILD COMPLETE!")
    print("=" * 70)