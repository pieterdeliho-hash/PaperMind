"""
Generate embeddings for text chunks
Converts text into vector representations for similarity search
"""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import time


class EmbeddingGenerator:
    """Generate embeddings using sentence-transformers"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Args:
            model_name: Sentence-transformer model to use
                - all-MiniLM-L6-v2: Fast, lightweight (384 dim)
                - all-mpnet-base-v2: Better quality (768 dim)
        """
        print(f"Loading embedding model: {model_name}")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"✓ Model loaded! Embedding dimension: {self.embedding_dim}")

    def generate_embeddings(
            self,
            chunks_file: str,
            output_file: str,
            batch_size: int = 32
    ):
        """
        Generate embeddings for all chunks

        Args:
            chunks_file: Path to chunks JSON
            output_file: Where to save embeddings
            batch_size: How many chunks to process at once
        """
        print(f"\nLoading chunks from: {chunks_file}")
        with open(chunks_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        chunks = data['chunks']
        metadata = data['metadata']

        print(f"Total chunks to embed: {len(chunks)}")
        print(f"Batch size: {batch_size}")
        print(f"Estimated time: ~{len(chunks) / (batch_size * 10):.1f} seconds\n")

        # Extract just the text
        texts = [chunk['chunk_text'] for chunk in chunks]

        # Generate embeddings with progress bar
        print("Generating embeddings...")
        start_time = time.time()

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        elapsed = time.time() - start_time

        print(f"\n✓ Embeddings generated!")
        print(f"  Time taken: {elapsed:.2f} seconds")
        print(f"  Speed: {len(chunks) / elapsed:.1f} chunks/second")
        print(f"  Embedding shape: {embeddings.shape}")

        # Save embeddings and metadata
        output_data = {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "num_chunks": len(chunks),
            "chunk_metadata": metadata,
            "embeddings": embeddings.tolist(),  # Convert to list for JSON
            "chunks": chunks  # Keep chunk info for reference
        }

        print(f"\nSaving to: {output_file}")
        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True, parents=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)

        # Also save embeddings as numpy array (more efficient)
        npy_path = output_path.with_suffix('.npy')
        np.save(npy_path, embeddings)

        print(f"✓ Saved embeddings JSON: {output_path}")
        print(f"✓ Saved embeddings NPY: {npy_path}")
        print(f"  JSON size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
        print(f"  NPY size: {npy_path.stat().st_size / 1024 / 1024:.2f} MB")

        return embeddings, chunks


if __name__ == "__main__":
    print("=" * 70)
    print("EMBEDDING GENERATION")
    print("Converting text chunks to vector representations")
    print("=" * 70)

    # Initialize generator
    generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")

    # Generate embeddings for 512-token chunks
    embeddings, chunks = generator.generate_embeddings(
        chunks_file="data/processed/chunks_recursive_512.json",
        output_file="data/processed/embeddings_512.json",
        batch_size=32
    )

    print("\n" + "=" * 70)
    print("EMBEDDING GENERATION COMPLETE!")
    print("=" * 70)