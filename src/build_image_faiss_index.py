"""
Build FAISS index for image embeddings
Enables fast image similarity search
"""

import numpy as np
import faiss
import json
import pickle
from pathlib import Path
import time


class ImageFAISSBuilder:
    """Build FAISS index for CLIP image embeddings"""

    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        self.index = None
        self.images_metadata = None

    def build_index(
            self,
            embeddings_file: str = "data/processed/image_embeddings.npy",
            metadata_file: str = "data/processed/image_embeddings.json"
    ):
        """
        Build FAISS index from image embeddings
        """
        print("=" * 70)
        print("BUILDING IMAGE FAISS INDEX")
        print("=" * 70)

        # Load embeddings
        print(f"\nLoading embeddings from: {embeddings_file}")
        embeddings = np.load(embeddings_file)

        # Load metadata
        with open(metadata_file, 'r') as f:
            data = json.load(f)

        self.images_metadata = data['images']

        print(f"Loaded {len(embeddings)} image embeddings")
        print(f"Embedding dimension: {embeddings.shape[1]}")

        # Ensure float32
        embeddings = embeddings.astype('float32')

        # Build index
        print(f"\nBuilding IndexFlatL2 index...")
        start_time = time.time()

        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(embeddings)

        elapsed = time.time() - start_time

        print(f"Index built in {elapsed:.3f} seconds")
        print(f"Total vectors indexed: {self.index.ntotal}")

        return self.index

    def save_index(self, output_dir: str = "data/processed/faiss_image_index"):
        """Save FAISS index and metadata"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        # Save FAISS index
        index_file = output_path / "faiss_image_index.bin"
        faiss.write_index(self.index, str(index_file))
        print(f"\nSaved FAISS index: {index_file}")

        # Save metadata
        metadata_file = output_path / "images_metadata.pkl"
        with open(metadata_file, 'wb') as f:
            pickle.dump(self.images_metadata, f)
        print(f"Saved metadata: {metadata_file}")

        # Save index info
        info = {
            "num_vectors": self.index.ntotal,
            "embedding_dim": self.embedding_dim,
            "index_type": type(self.index).__name__,
            "model": "openai/clip-vit-base-patch32"
        }
        info_file = output_path / "index_info.json"
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)
        print(f"Saved index info: {info_file}")

        print(f"\nIndex saved to: {output_path}")


if __name__ == "__main__":
    builder = ImageFAISSBuilder(embedding_dim=512)
    builder.build_index()
    builder.save_index()

    print("\n" + "=" * 70)
    print("IMAGE INDEX BUILD COMPLETE")
    print("=" * 70)