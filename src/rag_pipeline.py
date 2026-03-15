"""
Complete RAG Pipeline: Retrieval + LLM Generation
The brain of PaperMind - answers questions using retrieved context
"""

import os
from dotenv import load_dotenv
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from typing import List, Dict
import time

# Load environment variables
load_dotenv()


class RAGPipeline:
    """End-to-end RAG system with retrieval and generation"""

    def __init__(
            self,
            index_path: str = "data/processed/faiss_index/faiss_index.bin",
            metadata_path: str = "data/processed/faiss_index/chunks_metadata.pkl",
            model_name: str = "all-MiniLM-L6-v2",
            llm_model: str = "gpt-3.5-turbo"
    ):
        print("Initializing RAG Pipeline...")

        # Load retrieval components
        print("  Loading FAISS index...")
        self.index = faiss.read_index(index_path)

        with open(metadata_path, 'rb') as f:
            self.chunks_metadata = pickle.load(f)

        print("  Loading embedding model...")
        self.embedding_model = SentenceTransformer(model_name)

        # Initialize OpenAI client
        print("  Connecting to OpenAI...")
        self.llm_model = llm_model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        print(f"✓ RAG Pipeline ready!")
        print(f"  - {self.index.ntotal} chunks indexed")
        print(f"  - LLM: {llm_model}\n")

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieve relevant chunks for query

        Args:
            query: User question
            k: Number of chunks to retrieve

        Returns:
            List of relevant chunks with metadata
        """
        # Embed query
        query_embedding = self.embedding_model.encode([query])[0]
        query_vector = query_embedding.astype('float32').reshape(1, -1)

        # Search
        distances, indices = self.index.search(query_vector, k)

        # Format results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            chunk = self.chunks_metadata[idx]
            similarity_score = 1 / (1 + dist)

            results.append({
                'score': similarity_score,
                'paper': chunk['paper_filename'],
                'chunk_id': chunk['chunk_id'],
                'text': chunk['chunk_text'],
                'tokens': chunk['chunk_tokens']
            })

        return results

    def generate_answer(
            self,
            query: str,
            context_chunks: List[Dict],
            temperature: float = 0.3
    ) -> Dict:
        """
        Generate answer using LLM with retrieved context

        Args:
            query: User question
            context_chunks: Retrieved relevant chunks
            temperature: LLM temperature (0=deterministic, 1=creative)

        Returns:
            Dict with answer, sources, and metadata
        """
        # Build context from chunks
        context = "\n\n---\n\n".join([
            f"[Source {i + 1}: {chunk['paper'][:50]}...]\n{chunk['text']}"
            for i, chunk in enumerate(context_chunks)
        ])

        # Build prompt
        system_prompt = """You are an AI research assistant. Answer questions based on the provided research paper excerpts.

Guidelines:
- Answer directly and concisely
- Cite sources using [Source N] notation
- If information isn't in the context, say "I don't have enough information"
- Focus on the most relevant information
- Be precise and academic in tone"""

        user_prompt = f"""Context from research papers:

{context}

---

Question: {query}

Answer based on the context above, citing sources:"""

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

        # Extract sources
        sources = [
            {
                'paper': chunk['paper'],
                'chunk_id': chunk['chunk_id'],
                'score': chunk['score']
            }
            for chunk in context_chunks
        ]

        return {
            'answer': answer,
            'sources': sources,
            'latency': latency,
            'model': self.llm_model,
            'tokens_used': response.usage.total_tokens
        }

    def query(self, question: str, k: int = 5, verbose: bool = True) -> Dict:
        """
        Complete RAG query: retrieve + generate

        Args:
            question: User question
            k: Number of chunks to retrieve
            verbose: Print progress

        Returns:
            Complete result with answer and sources
        """
        if verbose:
            print(f"Question: {question}")
            print("=" * 80)

        # Step 1: Retrieve
        if verbose:
            print(f"\n[1/2] Retrieving top {k} relevant chunks...")

        chunks = self.retrieve(question, k=k)

        if verbose:
            print(f"✓ Retrieved {len(chunks)} chunks")
            print(f"  Top score: {chunks[0]['score']:.4f}")
            print(f"  Papers: {set(c['paper'][:30] for c in chunks)}")

        # Step 2: Generate
        if verbose:
            print(f"\n[2/2] Generating answer with {self.llm_model}...")

        result = self.generate_answer(question, chunks)

        if verbose:
            print(f"✓ Answer generated in {result['latency']:.2f}s")
            print(f"  Tokens used: {result['tokens_used']}")

        return result

    def display_result(self, result: Dict):
        """Pretty print RAG result"""
        print("\n" + "=" * 80)
        print("ANSWER")
        print("=" * 80)
        print(result['answer'])

        print("\n" + "=" * 80)
        print("SOURCES")
        print("=" * 80)
        for i, source in enumerate(result['sources'], 1):
            print(f"[{i}] {source['paper']}")
            print(f"    Chunk {source['chunk_id'] + 1} | Relevance: {source['score']:.4f}")

        print("\n" + "=" * 80)
        print("METADATA")
        print("=" * 80)
        print(f"Model: {result['model']}")
        print(f"Latency: {result['latency']:.2f}s")
        print(f"Tokens: {result['tokens_used']}")


def demo():
    """Demo the RAG pipeline"""
    rag = RAGPipeline()

    test_questions = [
        "What is the attention mechanism in transformers?",
        "How do vision transformers process images?",
        "What are the key advantages of transformers over RNNs?",
    ]

    print("=" * 80)
    print("RAG PIPELINE DEMO")
    print("=" * 80)

    for question in test_questions:
        result = rag.query(question, k=3, verbose=True)
        rag.display_result(result)
        print("\n\n")


def interactive():
    """Interactive RAG session"""
    rag = RAGPipeline()

    print("=" * 80)
    print("PAPERMIND - Interactive RAG")
    print("=" * 80)
    print("Ask questions about transformer research papers")
    print("Type 'quit' to exit\n")

    while True:
        question = input("Question: ").strip()

        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not question:
            continue

        try:
            result = rag.query(question, k=3, verbose=False)
            rag.display_result(result)
            print("\n")

        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.\n")


if __name__ == "__main__":
    # Run demo first
    demo()

    # Then start interactive mode
    print("\n\n")
    interactive()