"""
ArXiv Paper Downloader
Downloads research papers based on search queries
"""

import arxiv
from pathlib import Path
import time
from typing import List, Dict


class PaperDownloader:
    """Download papers from ArXiv"""

    def __init__(self, download_dir: str = "data/papers"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True, parents=True)

    def search_and_download(
            self,
            query: str,
            max_results: int = 10,
            sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance
    ) -> List[Dict]:
        """
        Search ArXiv and download papers

        Args:
            query: Search query (e.g., "transformer attention mechanism")
            max_results: Number of papers to download
            sort_by: How to sort results (Relevance, LastUpdatedDate, SubmittedDate)

        Returns:
            List of paper metadata
        """
        print(f"Searching ArXiv for: '{query}'")
        print(f"Downloading up to {max_results} papers...")
        print("=" * 60)

        # Search ArXiv
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort_by
        )

        papers_metadata = []

        for i, result in enumerate(search.results(), 1):
            print(f"\n[{i}/{max_results}] {result.title}")
            print(f"Authors: {', '.join([a.name for a in result.authors[:3]])}...")
            print(f"Published: {result.published.strftime('%Y-%m-%d')}")

            # Create clean filename
            # Remove special characters, limit length
            clean_title = "".join(c for c in result.title if c.isalnum() or c in (' ', '-', '_'))
            clean_title = clean_title[:50].strip()
            filename = f"{clean_title}_{result.entry_id.split('/')[-1]}.pdf"
            filepath = self.download_dir / filename

            try:
                # Download PDF
                result.download_pdf(filename=str(filepath))
                print(f"✓ Downloaded: {filename}")

                # Store metadata
                papers_metadata.append({
                    "title": result.title,
                    "authors": [a.name for a in result.authors],
                    "published": result.published.strftime('%Y-%m-%d'),
                    "summary": result.summary,
                    "arxiv_id": result.entry_id.split('/')[-1],
                    "pdf_url": result.pdf_url,
                    "filename": filename,
                    "filepath": str(filepath)
                })

                # Be nice to ArXiv servers
                time.sleep(3)

            except Exception as e:
                print(f"✗ Failed to download: {e}")

        print("\n" + "=" * 60)
        print(f"Successfully downloaded {len(papers_metadata)} papers")

        return papers_metadata

    def download_specific_papers(self, arxiv_ids: List[str]) -> List[Dict]:
        """
        Download specific papers by ArXiv ID

        Args:
            arxiv_ids: List of ArXiv IDs (e.g., ["1706.03762", "2010.11929"])
        """
        papers_metadata = []

        for arxiv_id in arxiv_ids:
            print(f"\nDownloading ArXiv ID: {arxiv_id}")

            try:
                search = arxiv.Search(id_list=[arxiv_id])
                result = next(search.results())

                clean_title = "".join(c for c in result.title if c.isalnum() or c in (' ', '-', '_'))
                clean_title = clean_title[:50].strip()
                filename = f"{clean_title}_{arxiv_id}.pdf"
                filepath = self.download_dir / filename

                result.download_pdf(filename=str(filepath))
                print(f"✓ Downloaded: {result.title}")

                papers_metadata.append({
                    "title": result.title,
                    "authors": [a.name for a in result.authors],
                    "published": result.published.strftime('%Y-%m-%d'),
                    "summary": result.summary,
                    "arxiv_id": arxiv_id,
                    "pdf_url": result.pdf_url,
                    "filename": filename,
                    "filepath": str(filepath)
                })

                time.sleep(3)

            except Exception as e:
                print(f"✗ Failed: {e}")

        return papers_metadata


if __name__ == "__main__":
    downloader = PaperDownloader()

    # Option 1: Search and download
    print("OPTION 1: Search-based download")
    print("=" * 60)
    papers = downloader.search_and_download(
        query="transformer attention mechanism",
        max_results=10,
        sort_by=arxiv.SortCriterion.Relevance
    )

    # Option 2: Download specific landmark papers (uncomment to use)
    print("\n\nOPTION 2: Download specific papers")
    print("=" * 60)
    landmark_papers = [
        "1706.03762",  # Attention Is All You Need (original Transformer)
        "1810.04805",  # BERT
        "2005.14165",  # GPT-3
        "2010.11929",  # ViT (Vision Transformer)
    ]
    papers = downloader.download_specific_papers(landmark_papers)

    # Save metadata
    import json

    metadata_path = Path("data/processed/papers_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Metadata saved to: {metadata_path}")