"""
Test multi-modal search: text queries retrieve relevant images
"""

import numpy as np
import faiss
import pickle
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from pathlib import Path

class MultiModalSearch:
    """Search images using text queries via CLIP"""

    def __init__(self):
        print("Loading multi-modal search engine...")

        # Load image FAISS index
        self.image_index = faiss.read_index("data/processed/faiss_image_index/faiss_image_index.bin")

        with open("data/processed/faiss_image_index/images_metadata.pkl", 'rb') as f:
            self.images_metadata = pickle.load(f)

        # Load CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        print(f"Loaded {self.image_index.ntotal} images")
        print(f"Device: {self.device}\n")

    def search_images(self, text_query: str, k: int = 5):
        """
        Search images using text query

        Args:
            text_query: Natural language query
            k: Number of images to return
        """
        # Encode text query - FIXED VERSION
        inputs = self.processor(text=[text_query], return_tensors="pt", padding=True)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        with torch.no_grad():
            # Get text model output
            text_outputs = self.model.text_model(**inputs)
            text_embeds = text_outputs.pooler_output

            # Project to joint embedding space
            text_embeds = self.model.text_projection(text_embeds)

            # Normalize
            text_embeds = F.normalize(text_embeds, p=2, dim=1)

        # Convert to numpy
        query_vector = text_embeds.cpu().numpy().astype('float32')

        # Search FAISS
        distances, indices = self.image_index.search(query_vector, k)

        # Format results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            img_meta = self.images_metadata[idx]
            similarity = 1 / (1 + dist)

            results.append({
                'similarity': similarity,
                'distance': dist,
                'paper': img_meta['paper'],
                'page': img_meta['page'],
                'filename': img_meta['filename'],
                'path': img_meta['path'],
                'size': f"{img_meta['width']}x{img_meta['height']}"
            })

        return results

    def display_results(self, query: str, results: list):
        """Display search results"""
        print("=" * 80)
        print(f"QUERY: {query}")
        print("=" * 80)

        for i, result in enumerate(results, 1):
            print(f"\n[{i}] Similarity: {result['similarity']:.4f}")
            print(f"    Paper: {result['paper'][:60]}")
            print(f"    Page: {result['page']}")
            print(f"    Size: {result['size']}")
            print(f"    File: {result['filename']}")
            print(f"    Path: {result['path']}")


def demo():
    """Demo multi-modal search"""
    search = MultiModalSearch()

    # Test queries
    test_queries = [
        "transformer architecture diagram",
        "attention mechanism visualization",
        "neural network architecture",
        "training loss graph",
        "comparison table results",
        "vision transformer image patches"
    ]

    print("=" * 80)
    print("MULTI-MODAL SEARCH DEMO")
    print("Testing text-to-image retrieval with CLIP")
    print("=" * 80)

    for query in test_queries:
        results = search.search_images(query, k=3)
        search.display_results(query, results)
        print("\n")


if __name__ == "__main__":
    demo()