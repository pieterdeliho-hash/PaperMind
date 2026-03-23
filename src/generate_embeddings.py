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
        print("Device: CPU")
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
        Generate embeddings for all chunks with ultra-robust error handling

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

        texts = []
        valid_chunks = []
        skipped = 0

        print("\nCleaning and validating text chunks...")
        print("Testing each chunk individually (this may take a moment)...\n")

        for i, chunk in enumerate(tqdm(chunks, desc="Validating chunks")):
            try:
                # Get text
                text = chunk.get('chunk_text', '')

                # Force to string
                if not isinstance(text, str):
                    text = str(text) if text is not None else ''

                # Remove ALL problematic characters
                text = text.strip()
                text = text.replace('\x00', '')  # Null bytes
                text = text.replace('\ufffd', '')  # Replacement char
                text = text.replace('\u0000', '')  # Another null
                text = ''.join(char for char in text if ord(char) < 65536)  # Remove high unicode

                # Skip if too short or too long
                if len(text) < 10:
                    text = "This is a placeholder for an empty or very short chunk."

                if len(text) > 10000:
                    text = text[:10000]  # Truncate very long chunks

                # TEST if this text can be encoded (critical step!)
                try:
                    # Try encoding just this one text to see if it works
                    test_embed = self.model.encode([text], show_progress_bar=False, convert_to_numpy=True)

                    # If we got here, the text is valid!
                    texts.append(text)
                    valid_chunks.append(chunk)

                except Exception as encode_error:
                    # This specific text fails encoding - skip it
                    print(f"\n  Skipping chunk {i} - encoding failed: {str(encode_error)[:100]}")
                    skipped += 1
                    continue

            except Exception as e:
                print(f"\n  Skipping chunk {i} - processing failed: {str(e)[:100]}")
                skipped += 1
                continue

        print(f"\n✓ Validated {len(texts)} chunks (skipped {skipped})\n")

        # Update chunks to only include valid ones
        chunks = valid_chunks

        # NOW generate embeddings in batches (we know all texts are valid)
        print("Generating embeddings on CPU...")
        start_time = time.time()

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False
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
            "num_skipped": skipped,
            "chunk_metadata": metadata,
            "embeddings": embeddings.tolist(),
            "chunks": chunks
        }

        print(f"\nSaving to: {output_file}")
        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True, parents=True)

        # Save JSON
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
    print("EMBEDDING GENERATION (CPU)")
    print("Converting text chunks to vector representations")
    print("=" * 70)

    # Initialize generator
    generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")

    # Generate embeddings for 512-token chunks
    embeddings, chunks = generator.generate_embeddings(
        chunks_file="data/processed/chunks_recursive_512.json",
        output_file="data/processed/embeddings_512.json",
        batch_size=32  # CPU-optimized batch size
    )

    print("\n" + "=" * 70)
    print("EMBEDDING GENERATION COMPLETE!")
    print("=" * 70)