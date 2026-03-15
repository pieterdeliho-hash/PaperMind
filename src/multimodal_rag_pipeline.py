"""
Multi-Modal RAG Pipeline
Retrieves both text chunks and images, generates answers with figure references
"""

import os
from dotenv import load_dotenv

# Streamlit Cloud compatibility - check for secrets first
try:
    import streamlit as st
    try:
        if "OPENAI_API_KEY" in st.secrets:
            os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
        else:
            load_dotenv()
    except:
        load_dotenv()
except ImportError:
    load_dotenv()

import faiss
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
from openai import OpenAI
from typing import List, Dict, Tuple
import time

class MultiModalRAG:
    """Complete multi-modal RAG system with text and image retrieval"""

    def __init__(self):
        """Initialize the multi-modal RAG pipeline"""

        print("Initializing Multi-Modal RAG Pipeline...")

        # Get absolute paths
        from pathlib import Path
        base_dir = Path(__file__).parent.parent  # Project root

        # Paths to indexes and metadata
        text_index_path = str(base_dir / "data/processed/faiss_index/faiss_index.bin")
        text_metadata_path = str(base_dir / "data/processed/faiss_index/chunks_metadata.pkl")
        image_index_path = str(base_dir / "data/processed/faiss_image_index/faiss_image_index.bin")
        image_metadata_path = str(base_dir / "data/processed/faiss_image_index/images_metadata.pkl")

        # Load text retrieval components
        print("  Loading text index...")
        self.text_index = faiss.read_index(text_index_path)
        with open(text_metadata_path, 'rb') as f:
            self.text_metadata = pickle.load(f)

        # Load image retrieval components
        print("  Loading image index...")
        self.image_index = faiss.read_index(image_index_path)
        with open(image_metadata_path, 'rb') as f:
            self.image_metadata = pickle.load(f)

        # Load embedding models
        print("  Loading embedding models...")
        self.text_embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        # Move to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model.to(self.device)

        # Initialize OpenAI client
        print("  Connecting to OpenAI...")
        self.client = OpenAI()
        self.llm_model = "gpt-3.5-turbo"

        print("Multi-Modal RAG ready!")
        print(f"  Text chunks: {self.text_index.ntotal}")
        print(f"  Images: {self.image_index.ntotal}")
        print(f"  LLM: gpt-3.5-turbo\n")

    def retrieve_text(self, query: str, k: int = 3) -> List[Dict]:
        """Retrieve relevant text chunks"""
        # Embed query
        query_embedding = self.text_embedding_model.encode([query])[0]
        query_vector = query_embedding.astype('float32').reshape(1, -1)

        # Search
        distances, indices = self.text_index.search(query_vector, k)

        # Format results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            chunk = self.text_metadata[idx]
            similarity = 1 / (1 + dist)

            results.append({
                'type': 'text',
                'score': similarity,
                'paper': chunk['paper_filename'],
                'chunk_id': chunk['chunk_id'],
                'text': chunk['chunk_text'],
                'tokens': chunk['chunk_tokens']
            })

        return results

    def retrieve_images(self, query: str, k: int = 2) -> List[Dict]:
        """Retrieve relevant images"""
        # Encode text query with CLIP
        inputs = self.clip_processor(text=[query], return_tensors="pt", padding=True)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        with torch.no_grad():
            text_outputs = self.clip_model.text_model(**inputs)
            text_embeds = text_outputs.pooler_output
            text_embeds = self.clip_model.text_projection(text_embeds)
            text_embeds = F.normalize(text_embeds, p=2, dim=1)

        query_vector = text_embeds.cpu().numpy().astype('float32')

        # Search
        distances, indices = self.image_index.search(query_vector, k)

        # Format results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            img = self.image_metadata[idx]
            similarity = 1 / (1 + dist)

            results.append({
                'type': 'image',
                'score': similarity,
                'paper': img['paper'],
                'page': img['page'],
                'filename': img['filename'],
                'path': img['path'],
                'size': f"{img['width']}x{img['height']}"
            })

        return results

    def retrieve_multimodal(
            self,
            query: str,
            text_k: int = 3,
            image_k: int = 2
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Retrieve both text and images

        Returns:
            (text_results, image_results)
        """
        text_results = self.retrieve_text(query, k=text_k)
        image_results = self.retrieve_images(query, k=image_k)

        return text_results, image_results

    def generate_answer(
            self,
            query: str,
            text_chunks: List[Dict],
            images: List[Dict],
            temperature: float = 0.3
    ) -> Dict:
        """Generate answer using both text and image context"""

        # Build text context
        text_context = "\n\n---\n\n".join([
            f"[Text Source {i + 1}: {chunk['paper'][:50]}]\n{chunk['text']}"
            for i, chunk in enumerate(text_chunks)
        ])

        # Build image context
        image_context = "\n".join([
            f"[Figure {i + 1}: {img['paper'][:50]}, Page {img['page']}, File: {img['filename']}]"
            for i, img in enumerate(images)
        ])

        # System prompt
        system_prompt = """You are an AI research assistant with access to both text excerpts and figures from research papers.

Guidelines:
- Answer questions based on provided text excerpts and available figures
- Reference text sources using [Text Source N]
- Reference figures using [Figure N] when relevant
- If a figure would help illustrate your answer, mention it explicitly
- Be precise and academic in tone
- If information isn't in the context, say so"""

        # User prompt
        user_prompt = f"""Text context from papers:

{text_context}

---

Available figures:
{image_context}

---

Question: {query}

Answer based on the text and mention relevant figures if they help illustrate the answer:"""

        # Call LLM
        start_time = time.time()

        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=500
        )

        answer = response.choices[0].message.content
        latency = time.time() - start_time

        return {
            'answer': answer,
            'text_sources': text_chunks,
            'image_sources': images,
            'latency': latency,
            'model': self.llm_model,
            'tokens_used': response.usage.total_tokens
        }

    def query(
            self,
            question: str,
            text_k: int = 3,
            image_k: int = 2,
            verbose: bool = True
    ) -> Dict:
        """
        Complete multi-modal RAG query

        Args:
            question: User question
            text_k: Number of text chunks to retrieve
            image_k: Number of images to retrieve
            verbose: Print progress
        """
        if verbose:
            print(f"Question: {question}")
            print("=" * 80)

        # Retrieve
        if verbose:
            print(f"\n[1/2] Retrieving context...")
            print(f"  Text chunks: top {text_k}")
            print(f"  Images: top {image_k}")

        text_results, image_results = self.retrieve_multimodal(
            question,
            text_k=text_k,
            image_k=image_k
        )

        if verbose:
            print(f"\nRetrieved:")
            print(
                f"  {len(text_results)} text chunks (scores: {text_results[0]['score']:.3f}-{text_results[-1]['score']:.3f})")
            print(
                f"  {len(image_results)} images (scores: {image_results[0]['score']:.3f}-{image_results[-1]['score']:.3f})")

        # Generate
        if verbose:
            print(f"\n[2/2] Generating answer with {self.llm_model}...")

        result = self.generate_answer(question, text_results, image_results)

        if verbose:
            print(f"Answer generated in {result['latency']:.2f}s")
            print(f"Tokens used: {result['tokens_used']}")

        return result

    def display_result(self, result: Dict):
        """Pretty print multi-modal result"""
        print("\n" + "=" * 80)
        print("ANSWER")
        print("=" * 80)
        print(result['answer'])

        print("\n" + "=" * 80)
        print("TEXT SOURCES")
        print("=" * 80)
        for i, source in enumerate(result['text_sources'], 1):
            print(f"[{i}] {source['paper']}")
            print(f"    Chunk {source['chunk_id'] + 1} | Score: {source['score']:.4f}")

        print("\n" + "=" * 80)
        print("FIGURE SOURCES")
        print("=" * 80)
        for i, img in enumerate(result['image_sources'], 1):
            print(f"[{i}] {img['paper']}")
            print(f"    Page {img['page']} | Score: {img['score']:.4f}")
            print(f"    File: {img['filename']}")
            print(f"    Path: {img['path']}")

        print("\n" + "=" * 80)
        print("METADATA")
        print("=" * 80)
        print(f"Model: {result['model']}")
        print(f"Latency: {result['latency']:.2f}s")
        print(f"Tokens: {result['tokens_used']}")


def demo():
    """Demo multi-modal RAG"""
    rag = MultiModalRAG()

    # Questions that benefit from images
    test_questions = [
        "Explain the transformer architecture. Are there any diagrams?",
        "How does the attention mechanism work visually?",
        "Show me examples of training loss curves in transformers",
    ]

    print("=" * 80)
    print("MULTI-MODAL RAG DEMO")
    print("Answering questions with text + image retrieval")
    print("=" * 80)

    for question in test_questions:
        result = rag.query(question, text_k=3, image_k=2, verbose=True)
        rag.display_result(result)
        print("\n\n")


def interactive():
    """Interactive multi-modal RAG"""
    rag = MultiModalRAG()

    print("=" * 80)
    print("PAPERMIND - Multi-Modal Interactive RAG")
    print("=" * 80)
    print("Ask questions about transformer papers (text + images)")
    print("Type 'quit' to exit\n")

    while True:
        question = input("Question: ").strip()

        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not question:
            continue

        try:
            result = rag.query(question, text_k=3, image_k=2, verbose=False)
            rag.display_result(result)
            print("\n")

        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.\n")


if __name__ == "__main__":
    # Run demo
    demo()

    # Start interactive
    print("\n\n")
    interactive()