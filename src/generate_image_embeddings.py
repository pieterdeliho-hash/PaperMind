"""
Generate CLIP embeddings for extracted images
Enables text-to-image search
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image
import pickle
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import time


class ImageEmbeddingGenerator:
    """Generate CLIP embeddings for images"""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = None
    ):
        print("Initializing CLIP model...")

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Using device: {self.device}")
        print("Downloading/loading CLIP model...")

        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.embedding_dim = self.model.config.projection_dim

        print(f"Model loaded: {model_name}")
        print(f"Embedding dimension: {self.embedding_dim}")

    def generate_image_embedding(self, image_path: str) -> np.ndarray:
        """Generate CLIP embedding for a single image"""
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")

            # Process image
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate embedding
            with torch.no_grad():
                # Get vision model output
                vision_outputs = self.model.vision_model(**inputs)
                image_embeds = vision_outputs.pooler_output

                # Project to joint embedding space
                image_embeds = self.model.visual_projection(image_embeds)

                # Normalize
                image_embeds = F.normalize(image_embeds, p=2, dim=1)

            # Convert to numpy
            embedding = image_embeds.cpu().numpy()[0]

            return embedding

        except Exception as e:
            print(f"\nError: {Path(image_path).name}: {str(e)[:50]}")
            return None

    def generate_all_embeddings(
        self,
        metadata_path: str = "data/processed/faiss_image_index/images_metadata.pkl",
        output_path: str = "data/processed/image_embeddings.json"
    ):
        """Generate embeddings for all extracted images"""

        print("\n" + "=" * 70)
        print("GENERATING IMAGE EMBEDDINGS")
        print("=" * 70)

        # Check metadata exists
        if not Path(metadata_path).exists():
            print(f"ERROR: Metadata not found: {metadata_path}")
            print(f"Tried: {metadata_path}")

            # Try alternative paths
            alt_paths = [
                "data/processed/images/images_metadata.json",
                "data/processed/faiss_image_index/images_metadata.pkl"
            ]

            for alt_path in alt_paths:
                if Path(alt_path).exists():
                    print(f"Found alternative: {alt_path}")
                    metadata_path = alt_path
                    break
            else:
                print("ERROR: No metadata file found!")
                return None

        # Load metadata (support both JSON and PKL)
        print(f"Loading metadata from: {metadata_path}")

        if metadata_path.endswith('.pkl'):
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
        else:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

        # Handle different metadata formats
        if isinstance(metadata, dict) and 'images' in metadata:
            images = metadata['images']
        elif isinstance(metadata, list):
            images = metadata
        else:
            print(f"ERROR: Unexpected metadata format: {type(metadata)}")
            return None

        if len(images) == 0:
            print("ERROR: No images in metadata!")
            return None

        print(f"Total images: {len(images)}")
        print(f"Device: {self.device}")
        print(f"Embedding dim: {self.embedding_dim}\n")

        embeddings = []
        valid_images = []
        failed = 0

        start_time = time.time()

        for img_meta in tqdm(images, desc="Generating embeddings"):
            # Handle different metadata structures
            if isinstance(img_meta, dict):
                img_path = Path(img_meta.get('path', img_meta.get('filepath', '')))
            else:
                img_path = Path(str(img_meta))

            if not img_path.exists():
                failed += 1
                continue

            embedding = self.generate_image_embedding(str(img_path))

            if embedding is not None:
                embeddings.append(embedding)
                valid_images.append(img_meta)
            else:
                failed += 1

        elapsed = time.time() - start_time

        if len(embeddings) == 0:
            print("\nERROR: No embeddings generated!")
            return None

        embeddings_array = np.array(embeddings)

        print(f"\n{'=' * 70}")
        print("COMPLETE")
        print("=" * 70)
        print(f"Total: {len(images)}")
        print(f"Success: {len(embeddings)}")
        print(f"Failed: {failed}")
        print(f"Rate: {len(embeddings) / len(images) * 100:.1f}%")
        print(f"Time: {elapsed:.1f}s")
        print(f"Speed: {len(embeddings) / elapsed:.1f} imgs/s")
        print(f"Shape: {embeddings_array.shape}")

        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)

        output_data = {
            "model_name": "openai/clip-vit-base-patch32",
            "embedding_dim": self.embedding_dim,
            "num_images": len(embeddings),
            "embeddings": embeddings_array.tolist(),
            "images": valid_images
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f)

        npy_path = output_path.with_suffix('.npy')
        np.save(npy_path, embeddings_array)

        print(f"\nSaved JSON: {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
        print(f"Saved NPY: {npy_path} ({npy_path.stat().st_size / 1024 / 1024:.1f} MB)")

        return embeddings_array, valid_images


if __name__ == "__main__":
    generator = ImageEmbeddingGenerator()
    result = generator.generate_all_embeddings()

    if result:
        print("\nSuccess!")
    else:
        print("\nFailed!")