"""
PDF Text Extraction Module
Handles extracting text from academic papers
"""

import os
from pathlib import Path
from PyPDF2 import PdfReader
from typing import Dict, List
import json


class PDFExtractor:
    """Extract text from PDF files"""

    def __init__(self, pdf_dir: str, output_dir: str):
        """
        Args:
            pdf_dir: Directory containing PDF files
            output_dir: Directory to save extracted text
        """
        self.pdf_dir = Path(pdf_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def extract_text_from_pdf(self, pdf_path: Path) -> Dict:
        """
        Extract text from a single PDF

        Returns:
            Dict with 'filename', 'text', 'num_pages'
        """
        try:
            reader = PdfReader(str(pdf_path))

            # Extract text from all pages
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"

            return {
                "filename": pdf_path.name,
                "text": text,
                "num_pages": len(reader.pages),
                "status": "success"
            }

        except Exception as e:
            return {
                "filename": pdf_path.name,
                "text": "",
                "num_pages": 0,
                "status": "failed",
                "error": str(e)
            }

    def extract_all(self) -> List[Dict]:
        """Extract text from all PDFs in directory"""
        results = []

        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        print(f"Found {len(pdf_files)} PDF files")

        for pdf_path in pdf_files:
            print(f"Processing: {pdf_path.name}")
            result = self.extract_text_from_pdf(pdf_path)
            results.append(result)

        return results

    def save_results(self, results: List[Dict], output_filename: str = "extracted_text.json"):
        """Save extraction results to JSON"""
        output_path = self.output_dir / output_filename

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Saved results to: {output_path}")


# Example usage
if __name__ == "__main__":
    extractor = PDFExtractor(
        pdf_dir="data/papers",
        output_dir="data/processed"
    )

    results = extractor.extract_all()
    extractor.save_results(results)

    # Print summary
    successful = sum(1 for r in results if r['status'] == 'success')
    print(f"\n{'=' * 50}")
    print(f"Extraction complete!")
    print(f"Successful: {successful}/{len(results)}")
    print(f"{'=' * 50}")