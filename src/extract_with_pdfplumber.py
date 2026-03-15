"""Fallback extraction using pdfplumber for problematic PDFs"""
import pdfplumber
from pathlib import Path
import json


def extract_with_pdfplumber(pdf_path: str) -> str:
    """Try extraction with pdfplumber (better for complex layouts)"""
    text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    return text


# Example: Re-extract a specific problematic file
if __name__ == "__main__":
    # Replace with your problematic filename
    problem_file = "data/papers/YOUR_PROBLEM_FILE.pdf"

    if Path(problem_file).exists():
        text = extract_with_pdfplumber(problem_file)
        print(f"Extracted {len(text)} characters")
        print("\nFirst 500 chars:")
        print(text[:500])