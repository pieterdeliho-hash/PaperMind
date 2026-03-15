"""
Text Chunking Module - Production Version
Uses recursive character splitting with token-aware chunking
"""

import json
from pathlib import Path
from typing import List, Dict
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextChunker:
    """Chunk text using industry-standard recursive splitting"""

    def __init__(
            self,
            chunk_size: int = 512,
            chunk_overlap: int = 102,  # 20% of 512
            encoding_name: str = "cl100k_base"  # GPT-3.5/4 tokenizer
    ):
        """
        Args:
            chunk_size: Target chunk size in TOKENS (not words)
            chunk_overlap: Number of overlapping tokens
            encoding_name: Tokenizer to use (cl100k_base = GPT-3.5/4)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding_name = encoding_name

        # Initialize tokenizer for accurate token counting
        self.tokenizer = tiktoken.get_encoding(encoding_name)

        # Create recursive splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self._count_tokens,
            separators=[
                "\n\n",  # Double newline (paragraphs)
                "\n",  # Single newline
                ". ",  # Sentence end
                "! ",
                "? ",
                ";",
                ",",
                " ",  # Words
                ""  # Characters (fallback)
            ]
        )

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        return len(self.tokenizer.encode(text))

    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk text using recursive character splitting

        Args:
            text: Input text

        Returns:
            List of text chunks
        """
        chunks = self.splitter.split_text(text)
        return chunks

    def chunk_papers(
            self,
            extracted_data_path: str,
            output_path: str
    ) -> Dict:
        """
        Chunk all papers from extraction output

        Args:
            extracted_data_path: Path to extracted_text.json
            output_path: Where to save chunked data

        Returns:
            Statistics about chunking
        """
        # Load extracted text
        with open(extracted_data_path, 'r', encoding='utf-8') as f:
            papers = json.load(f)

        chunked_papers = []
        stats = {
            "total_papers": len(papers),
            "total_chunks": 0,
            "avg_chunks_per_paper": 0,
            "avg_chunk_tokens": 0,
            "avg_chunk_chars": 0,
            "method": "recursive_character_splitting",
            "chunk_size_tokens": self.chunk_size,
            "overlap_tokens": self.chunk_overlap,
            "overlap_percentage": (self.chunk_overlap / self.chunk_size) * 100
        }

        total_tokens = 0
        total_chars = 0

        for paper in papers:
            if paper['status'] != 'success':
                continue

            text = paper['text']
            chunks = self.chunk_text(text)

            # Store chunks with metadata
            for i, chunk in enumerate(chunks):
                chunk_tokens = self._count_tokens(chunk)

                chunked_papers.append({
                    "paper_filename": paper['filename'],
                    "chunk_id": i,
                    "chunk_text": chunk,
                    "chunk_tokens": chunk_tokens,
                    "chunk_chars": len(chunk),
                    "total_chunks_in_paper": len(chunks)
                })

                total_tokens += chunk_tokens
                total_chars += len(chunk)

            stats["total_chunks"] += len(chunks)

        # Calculate averages
        if stats["total_papers"] > 0:
            stats["avg_chunks_per_paper"] = stats["total_chunks"] / stats["total_papers"]

        if stats["total_chunks"] > 0:
            stats["avg_chunk_tokens"] = total_tokens / stats["total_chunks"]
            stats["avg_chunk_chars"] = total_chars / stats["total_chunks"]

        # Save results
        output = {
            "metadata": stats,
            "chunks": chunked_papers
        }

        output_path = Path(output_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        return stats


if __name__ == "__main__":
    print("=" * 70)
    print("TEXT CHUNKING - Recursive Character Splitting")
    print("Industry Standard: 512 tokens, 20% overlap")
    print("=" * 70)

    # Standard configuration (512 tokens, 20% overlap)
    print("\n[1] Standard Config: 512 tokens, 102 overlap (20%)")
    chunker_standard = TextChunker(chunk_size=512, chunk_overlap=102)
    stats_standard = chunker_standard.chunk_papers(
        extracted_data_path="data/processed/extracted_text.json",
        output_path="data/processed/chunks_recursive_512.json"
    )

    print(f"  Total chunks: {stats_standard['total_chunks']}")
    print(f"  Avg chunks per paper: {stats_standard['avg_chunks_per_paper']:.1f}")
    print(f"  Avg chunk size: {stats_standard['avg_chunk_tokens']:.1f} tokens")
    print(f"  Avg chunk size: {stats_standard['avg_chunk_chars']:.0f} characters")

    # Alternative: Larger chunks for more context (optional test)
    print("\n[2] Larger Chunks: 1024 tokens, 205 overlap (20%)")
    chunker_large = TextChunker(chunk_size=1024, chunk_overlap=205)
    stats_large = chunker_large.chunk_papers(
        extracted_data_path="data/processed/extracted_text.json",
        output_path="data/processed/chunks_recursive_1024.json"
    )

    print(f"  Total chunks: {stats_large['total_chunks']}")
    print(f"  Avg chunks per paper: {stats_large['avg_chunks_per_paper']:.1f}")
    print(f"  Avg chunk size: {stats_large['avg_chunk_tokens']:.1f} tokens")
    print(f"  Avg chunk size: {stats_large['avg_chunk_chars']:.0f} characters")

    print("\n" + "=" * 70)
    print("Chunking complete! Using production-ready recursive splitting")
    print("=" * 70)