"""
Download additional papers across diverse transformer topics
Strategic expansion for better retrieval coverage
"""

import arxiv
from pathlib import Path
import time
from typing import List, Dict
import json


class StrategicPaperDownloader:
    """Download papers across multiple transformer topics"""

    def __init__(self, download_dir: str = "data/papers"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True, parents=True)
        self.downloaded_ids = self._get_existing_ids()

    def _get_existing_ids(self):
        """Get arxiv IDs of already downloaded papers"""
        existing = set()
        for pdf_file in self.download_dir.glob("*.pdf"):
            # Extract arxiv ID from filename (last part before .pdf)
            parts = pdf_file.stem.split('_')
            if len(parts) > 0:
                # Look for pattern like "1706.03762" or "2010.11929v2"
                arxiv_id = parts[-1].replace('.pdf', '')
                existing.add(arxiv_id)
        return existing

    def download_by_query(self, query: str, max_results: int = 5):
        """Download papers for a specific search query"""
        print(f"\nSearching: '{query}'")
        print(f"Target: {max_results} papers")
        print("-" * 70)

        search = arxiv.Search(
            query=query,
            max_results=max_results * 2,  # Get more to account for duplicates
            sort_by=arxiv.SortCriterion.Relevance
        )

        downloaded = 0
        skipped = 0

        for result in search.results():
            if downloaded >= max_results:
                break

            arxiv_id = result.entry_id.split('/')[-1]

            # Skip if already have this paper
            if any(arxiv_id.startswith(existing) for existing in self.downloaded_ids):
                skipped += 1
                continue

            # Create filename
            clean_title = "".join(c for c in result.title if c.isalnum() or c in (' ', '-', '_'))
            clean_title = clean_title[:50].strip()
            filename = f"{clean_title}_{arxiv_id}.pdf"
            filepath = self.download_dir / filename

            try:
                result.download_pdf(filename=str(filepath))
                print(f"✓ [{downloaded + 1}/{max_results}] {result.title[:60]}")

                self.downloaded_ids.add(arxiv_id)
                downloaded += 1
                time.sleep(3)

            except Exception as e:
                print(f"✗ Failed: {e}")

        print(f"Downloaded: {downloaded} | Skipped (duplicates): {skipped}")
        return downloaded

    def download_diverse_corpus(self):
        """Download papers across diverse transformer topics"""

        queries = [
            # Core transformer architectures
            ("transformer architecture", 5),
            ("BERT language model", 3),
            ("GPT language model", 3),

            # Vision transformers
            ("vision transformer image", 5),
            ("ViT image classification", 3),

            # Attention mechanisms
            ("self-attention mechanism", 4),
            ("multi-head attention", 3),
            ("cross-attention transformer", 3),

            # Applications
            ("transformer NLP", 4),
            ("transformer computer vision", 4),
            ("transformer time series", 3),

            # Recent advances
            ("efficient transformer", 3),
            ("linear attention", 3),
            ("sparse transformer", 3),
        ]

        print("=" * 70)
        print("STRATEGIC PAPER DOWNLOAD")
        print("=" * 70)
        print(f"Current papers: {len(self.downloaded_ids)}")
        print(f"Target queries: {len(queries)}")
        print(f"Estimated new papers: {sum(q[1] for q in queries)}")
        print("=" * 70)

        total_downloaded = 0

        for query, max_results in queries:
            downloaded = self.download_by_query(query, max_results)
            total_downloaded += downloaded

        print("\n" + "=" * 70)
        print("DOWNLOAD COMPLETE")
        print("=" * 70)
        print(f"Total papers now: {len(self.downloaded_ids)}")
        print(f"New papers added: {total_downloaded}")
        print("=" * 70)


if __name__ == "__main__":
    downloader = StrategicPaperDownloader()
    downloader.download_diverse_corpus()