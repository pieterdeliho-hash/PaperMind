"""
Extract images and figures from research papers
Prepares visual content for multi-modal retrieval
"""

import fitz  # PyMuPDF
from pathlib import Path
from PIL import Image
import io
import json
from typing import List, Dict
from tqdm import tqdm


class ImageExtractor:
    """Extract images from PDF research papers"""

    def __init__(
            self,
            pdf_dir: str = "data/papers",
            output_dir: str = "data/processed/images"
    ):
        self.pdf_dir = Path(pdf_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def extract_images_from_pdf(self, pdf_path: Path) -> List[Dict]:
        """
        Extract all images from a PDF file

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of image metadata dictionaries
        """
        images = []

        try:
            doc = fitz.open(pdf_path)

            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images()

                for img_index, img_info in enumerate(image_list):
                    try:
                        xref = img_info[0]
                        base_image = doc.extract_image(xref)

                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]

                        # Convert to PIL Image
                        image = Image.open(io.BytesIO(image_bytes))

                        # Filter out very small images (likely logos, icons)
                        width, height = image.size
                        if width < 100 or height < 100:
                            continue

                        # Filter out very large images (likely full-page scans)
                        if width > 3000 or height > 3000:
                            continue

                        # Create filename
                        paper_name = pdf_path.stem
                        img_filename = f"{paper_name}_page{page_num + 1}_img{img_index + 1}.{image_ext}"
                        img_path = self.output_dir / img_filename

                        # Save image
                        image.save(img_path)

                        # Store metadata
                        images.append({
                            "paper": pdf_path.name,
                            "page": page_num + 1,
                            "image_index": img_index + 1,
                            "filename": img_filename,
                            "width": width,
                            "height": height,
                            "format": image_ext,
                            "path": str(img_path)
                        })

                    except Exception as e:
                        # Skip problematic images
                        continue

            doc.close()

        except Exception as e:
            print(f"Error processing {pdf_path.name}: {e}")

        return images

    def extract_all_images(self) -> Dict:
        """
        Extract images from all PDFs

        Returns:
            Dictionary with extraction statistics and metadata
        """
        print("=" * 70)
        print("IMAGE EXTRACTION FROM PDFs")
        print("=" * 70)

        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        print(f"Found {len(pdf_files)} PDF files\n")

        all_images = []
        successful_papers = 0

        for pdf_path in tqdm(pdf_files, desc="Extracting images"):
            images = self.extract_images_from_pdf(pdf_path)

            if images:
                all_images.extend(images)
                successful_papers += 1

        # Save metadata
        metadata = {
            "total_papers": len(pdf_files),
            "papers_with_images": successful_papers,
            "total_images": len(all_images),
            "avg_images_per_paper": len(all_images) / successful_papers if successful_papers > 0 else 0,
            "images": all_images
        }

        metadata_path = self.output_dir / "images_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n{'=' * 70}")
        print("EXTRACTION COMPLETE")
        print("=" * 70)
        print(f"Total papers: {len(pdf_files)}")
        print(f"Papers with images: {successful_papers}")
        print(f"Total images extracted: {len(all_images)}")
        print(f"Avg images per paper: {metadata['avg_images_per_paper']:.1f}")
        print(f"Images saved to: {self.output_dir}")
        print(f"Metadata saved to: {metadata_path}")

        return metadata


if __name__ == "__main__":
    extractor = ImageExtractor()
    metadata = extractor.extract_all_images()

    # Show some examples
    print(f"\n{'=' * 70}")
    print("SAMPLE EXTRACTED IMAGES")
    print("=" * 70)

    for i, img in enumerate(metadata['images'][:5], 1):
        print(f"\n{i}. {img['filename']}")
        print(f"   Paper: {img['paper'][:50]}")
        print(f"   Page: {img['page']}")
        print(f"   Size: {img['width']}x{img['height']}")
        print(f"   Format: {img['format']}")